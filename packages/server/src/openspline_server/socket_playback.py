"""Same-origin demo playback with a bounded, acknowledged media window."""

import asyncio
import base64
import contextlib
import io
import time

from PIL import Image
from starlette.websockets import WebSocketDisconnect

PLAYBACK_WINDOW_SAMPLES = 48000


def jpeg(frame):
    output = io.BytesIO()
    Image.fromarray(frame).save(output, format="JPEG", quality=80)
    return base64.b64encode(output.getvalue()).decode("ascii")


class SocketPlayback:
    def __init__(self, session, websocket):
        self.session = session
        self.websocket = websocket
        self.events = asyncio.Queue(64)
        self.closed = False
        self.idle_sent = self.idle_played = 0
        self.idle_epoch = session.epoch
        self.next_idle_at = 0.0

    def acknowledge_idle(self, sequence, epoch):
        if epoch == self.idle_epoch and type(sequence) is int:
            self.idle_played = max(self.idle_played, min(sequence, self.idle_sent))

    def emit(self, event):
        # Control messages cannot be silently lost (especially interruption).
        if self.events.full():
            self.closed = True
        else:
            self.events.put_nowait(event)

    async def send(self):
        s = self.session
        await self.websocket.send_json({"type": "connected", "epoch": s.epoch})
        while not self.closed and not s.closed:
            while not self.events.empty():
                await self.websocket.send_json(self.events.get_nowait())
            timeline = s.timeline
            if self.idle_epoch != s.epoch:
                self.idle_epoch = s.epoch
                self.idle_sent = self.idle_played = 0
            # Feedback arrives after playback + output latency + network RTT.
            # A 200ms window starves even realtime generation over ordinary WANs.
            # Keep at most one second outstanding; paused viewers still backpressure.
            if (
                s.viewer_ready.is_set()
                and timeline.sent - timeline.played < PLAYBACK_WINDOW_SAMPLES
                and (timeline.current is not None or timeline.queue)
            ):
                epoch = s.epoch
                before = timeline.sent
                audio = timeline.audio(1920)
                samples = timeline.sent
                frame = await asyncio.to_thread(jpeg, timeline.frame.copy())
                if epoch != s.epoch:
                    continue
                await self.websocket.send_json(
                    {
                        "type": "media",
                        "epoch": epoch,
                        "samples": samples,
                        "audio": base64.b64encode(
                            audio[: samples - before].astype("<i2").tobytes()
                        ).decode("ascii"),
                        "image": frame,
                    }
                )
            elif (
                s.viewer_ready.is_set()
                and timeline.played >= timeline.submitted
                and self.idle_sent - self.idle_played < 25
                and time.monotonic() >= self.next_idle_at
                and timeline.advance_idle(1920)
            ):
                epoch = s.epoch
                frame = await asyncio.to_thread(jpeg, timeline.frame.copy())
                if epoch != s.epoch:
                    continue
                self.idle_sent += 1
                self.next_idle_at = time.monotonic() + 0.04
                await self.websocket.send_json(
                    {"type": "idle", "epoch": epoch, "sequence": self.idle_sent, "image": frame}
                )
            else:
                await asyncio.sleep(0.01)
        await self.websocket.close(code=1011)

    async def close(self):
        self.closed = True
        with contextlib.suppress(RuntimeError, WebSocketDisconnect, OSError):
            await self.websocket.close()
