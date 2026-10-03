from __future__ import annotations

import asyncio
import contextlib
import secrets
import time
from dataclasses import dataclass
from fractions import Fraction

import av
import numpy as np
from PIL import Image

from .audio import AudioConverter, EncodedDecoder
from .media import Segment, Timeline


@dataclass
class Input:
    kind: str
    data: object
    epoch: int
    future: asyncio.Future | None = None


class Session:
    def __init__(self, id, worker, image, settings):
        self.id, self.worker, self.image, self.settings = id, worker, image, settings
        self.publisher_token = secrets.token_urlsafe(32)
        self.viewer_token = secrets.token_urlsafe(32)
        self.token_expires = time.time() + settings.token_ttl
        self.created = self.touched = time.monotonic()
        self.epoch = 0
        self.closed = False
        self.publisher_connected = False
        self.pc = None
        self.channel = None
        self.native_sink = None
        self.socket_sink = None
        self.viewer_ready = asyncio.Event()
        self.events = asyncio.Queue(256)
        self.input = asyncio.Queue(8)
        self.converter = None
        self.format = None
        self.pending = np.empty(0, dtype=np.int16)
        self.history = np.zeros(worker.info["cache_samples"], dtype=np.float32)
        self.downsampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        self.model_pending = np.empty(0, dtype=np.float32)
        self.timeline = Timeline(np.asarray(Image.open(image).convert("RGB").resize((512, 512))))
        self.task = None
        self.close_lock = asyncio.Lock()
        self.metrics = {
            "blocks": 0,
            "inference_seconds": 0.0,
            "audio_seconds": 0.0,
            "idle_blocks": 0,
            "idle_inference_seconds": 0.0,
        }

    async def start(self):
        await self.worker.call("prepare", str(self.image), 0)
        self.task = asyncio.create_task(self._consume())

    def touch(self):
        self.touched = time.monotonic()

    def descriptor(self):
        return dict(
            id=self.id,
            url=self.settings.public_url,
            token=self.viewer_token,
            expires_at=self.token_expires,
            viewer_url=f"{self.settings.public_url}/view#{self.id}:{self.viewer_token}",
        )

    def emit(self, event):
        event = {"session_id": self.id, "epoch": self.epoch, **event}
        if not self.events.full():
            self.events.put_nowait(event)
        if self.socket_sink:
            self.socket_sink.emit(event)
        if self.channel and self.channel.readyState == "open":
            import json

            self.channel.send(json.dumps(event))

    def playback(self, samples, epoch):
        if self.closed or epoch != self.epoch:
            return
        # Never accept feedback for audio that has not left this service.
        self.timeline.played = max(self.timeline.played, min(int(samples), self.timeline.sent))
        self.timeline.changed.set()
        self.touch()
        self.emit({"type": "playback", "samples": self.timeline.played, "sample_rate": 48000})

    async def push(self, data, format):
        if self.closed:
            raise RuntimeError("Session closed")
        encoding = format.get("encoding", "pcm_s16le")
        rate = format.get("sample_rate", 24000)
        channels = format.get("channels", 1)
        if (
            encoding != "mp3"
            and len(data) > rate * channels * (4 if encoding == "pcm_f32le" else 2) // 4
        ):
            raise ValueError("Send PCM in packets of at most 250ms")
        if len(data) > 65536:
            raise ValueError("Audio packet exceeds 64 KiB")
        if self.format and format != self.format:
            raise ValueError("End the turn before changing audio format")
        self.format = format
        self.touch()
        await self.input.put(Input("audio", (data, format), self.epoch))

    async def end_turn(self):
        future = asyncio.get_running_loop().create_future()
        await self.input.put(Input("end", None, self.epoch, future))
        return await future

    async def drain(self, target, epoch, timeout=180):
        async def wait():
            while self.timeline.played < target:
                if self.closed:
                    raise RuntimeError("Session closed")
                if self.epoch != epoch:
                    return
                self.timeline.changed.clear()
                if self.timeline.played >= target:
                    break
                await self.timeline.changed.wait()

        await asyncio.wait_for(wait(), timeout)

    async def interrupt(self):
        self.epoch += 1
        self.timeline.clear(self.epoch)
        self.touch()
        while not self.input.empty():
            item = self.input.get_nowait()
            if item.future and not item.future.done():
                item.future.set_exception(RuntimeError("Turn interrupted"))
        self.converter = None
        self.format = None
        self.pending = np.empty(0, dtype=np.int16)
        self.model_pending = np.empty(0, dtype=np.float32)
        self.history.fill(0)
        self.downsampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        # Retain motion conditioning within this session. Resetting to the portrait
        # here causes a visible jump whenever the user interrupts.
        self.emit({"type": "interrupted"})

    def _downsample(self, audio):
        if not len(audio):
            return np.empty(0, dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 48000
        frame.time_base = Fraction(1, 48000)
        frames = self.downsampler.resample(frame)
        return (
            np.concatenate([f.to_ndarray().reshape(-1) for f in frames])
            if frames
            else np.empty(0, dtype=np.float32)
        )

    async def _append(self, audio, epoch, final=False):
        if epoch != self.epoch:
            return
        self.pending = np.concatenate([self.pending, audio])
        converted = self._downsample(audio) if len(audio) else np.empty(0, dtype=np.float32)
        self.model_pending = np.concatenate([self.model_pending, converted])
        if final:
            tail = self.downsampler.resample(None)
            if tail:
                self.model_pending = np.concatenate(
                    [self.model_pending, *[f.to_ndarray().reshape(-1) for f in tail]]
                )
        chunk = self.worker.info["chunk_samples"]
        playback_chunk = chunk * 3
        while len(self.model_pending) >= chunk or (final and len(self.pending)):
            if epoch != self.epoch:
                return
            valid = min(len(self.pending), playback_chunk)
            if not valid:
                break
            pcm = self.pending[:valid].copy()
            self.pending = self.pending[valid:]
            model = self.model_pending[:chunk]
            self.model_pending = self.model_pending[chunk:]
            model = np.pad(model, (0, max(0, chunk - len(model))))
            self.history = np.concatenate([self.history, model])[-len(self.history) :]
            frames, elapsed = await self.worker.call("infer", self.history.copy())
            if epoch != self.epoch:
                return
            self.metrics["blocks"] += 1
            self.metrics["inference_seconds"] += elapsed
            self.metrics["audio_seconds"] += valid / 48000
            await self.timeline.put(Segment(pcm, frames, self.worker.info["fps"], epoch))
            self.emit(
                {
                    "type": "speaking",
                    "inference_ms": elapsed * 1000,
                    "generated_samples": self.timeline.submitted,
                }
            )

    async def _idle(self):
        timeline = self.timeline
        if not self.viewer_ready.is_set() or not self.input.empty() or timeline.idle_next is not None:
            return
        epoch = self.epoch
        chunk = self.worker.info["chunk_samples"]
        self.history = np.concatenate([self.history, np.zeros(chunk, dtype=np.float32)])[
            -len(self.history) :
        ]
        frames, elapsed = await self.worker.call("infer", self.history.copy())
        if epoch != self.epoch or self.closed:
            return
        self.metrics["idle_blocks"] += 1
        self.metrics["idle_inference_seconds"] += elapsed
        timeline.put_idle(frames, self.worker.info["fps"], epoch)

    async def _consume(self):
        try:
            while not self.closed:
                # Use one inference loop for speech and silence: never overlap GPU
                # calls, and always consume queued speech before generating more idle.
                try:
                    item = await asyncio.wait_for(self.input.get(), 0.04)
                except asyncio.TimeoutError:
                    await self._idle()
                    continue
                if item.epoch != self.epoch:
                    if item.future and not item.future.done():
                        item.future.set_exception(RuntimeError("Turn interrupted"))
                    continue
                try:
                    if item.kind == "audio":
                        raw, fmt = item.data
                        if self.converter is None:
                            self.converter = (
                                EncodedDecoder("mp3")
                                if fmt.get("encoding") == "mp3"
                                else AudioConverter(**fmt)
                            )
                        await self._append(self.converter.push(raw), item.epoch)
                    else:
                        tail = (
                            self.converter.flush()
                            if self.converter
                            else np.empty(0, dtype=np.int16)
                        )
                        await self._append(tail, item.epoch, final=True)
                        if item.epoch != self.epoch:
                            if item.future and not item.future.done():
                                item.future.set_exception(RuntimeError("Turn interrupted"))
                            continue
                        self.converter = None
                        self.format = None
                        self.downsampler = av.AudioResampler(
                            format="flt", layout="mono", rate=16000
                        )
                        if not item.future.done():
                            item.future.set_result((self.timeline.submitted, self.epoch))
                        self.emit({"type": "turn_end", "target_samples": self.timeline.submitted})
                except Exception as exc:
                    if item.future and not item.future.done():
                        item.future.set_exception(exc)
                    raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.emit({"type": "error", "code": "inference", "message": str(exc)})
            self.closed = True
            self.timeline.changed.set()
            while not self.input.empty():
                item = self.input.get_nowait()
                if item.future and not item.future.done():
                    item.future.set_exception(exc)

    async def close(self):
        self.closed = True
        self.timeline.changed.set()
        self.viewer_ready.set()
        if self.task:
            # Do not abandon an in-flight process RPC: release must not consume its response.
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        if self.pc:
            await self.pc.close()
        if self.native_sink:
            await self.native_sink.close()
        if self.socket_sink:
            await self.socket_sink.close()
