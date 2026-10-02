"""Optional native room sink; no WebRTC round-trip back through the agent process."""

import asyncio
import time

from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, DataStreamAudioReceiver


class LiveKitSink:
    def __init__(self, session):
        self.session = session
        self.room = rtc.Room()
        self.tasks = []
        self.started = False
        self.turn_base = 0
        self.frames = 0
        self.receiving = False

    async def start(self, url, token, sender_identity):
        await self.room.connect(url, token)
        self.audio = rtc.AudioSource(48000, 1, queue_size_ms=100)
        self.video = rtc.VideoSource(512, 512)
        self.sync = rtc.AVSynchronizer(
            audio_source=self.audio, video_source=self.video, video_fps=25, video_queue_size_ms=100
        )
        await self.room.local_participant.publish_track(
            rtc.LocalAudioTrack.create_audio_track("avatar_audio", self.audio)
        )
        await self.room.local_participant.publish_track(
            rtc.LocalVideoTrack.create_video_track("avatar_video", self.video)
        )
        self.receiver = DataStreamAudioReceiver(self.room, sender_identity=sender_identity)
        await self.receiver.start()

        @self.receiver.on("clear_buffer")
        def clear():
            self.tasks.append(asyncio.create_task(self.interrupt()))

        self.tasks.extend([asyncio.create_task(self._input()), asyncio.create_task(self._output())])
        for task in self.tasks:
            task.add_done_callback(self._failed)
        self.session.viewer_ready.set()
        self.session.emit({"type": "viewer_ready"})

    def _failed(self, task):
        if not task.cancelled() and task.exception():
            import logging

            logging.getLogger(__name__).error("Native transport failed", exc_info=task.exception())
            self.session.closed = True
            self.session.timeline.changed.set()

    async def interrupt(self):
        played = max(
            0, (self.session.timeline.sent - self.turn_base) / 48000 - self.audio.queued_duration
        )
        await self.sync.clear_queue()
        await self.session.interrupt()
        self.receiver.notify_playback_finished(playback_position=played, interrupted=True)
        self.turn_base = 0
        self.started = False
        self.receiving = False

    async def _input(self):
        async for frame in self.receiver:
            self.session.touch()
            if isinstance(frame, AudioSegmentEnd):
                if not self.receiving:
                    continue
                self.receiving = False
                before = self.session.epoch
                try:
                    target, epoch = await self.session.end_turn()
                except RuntimeError:
                    if before != self.session.epoch:
                        continue
                    raise
                if epoch != before:
                    continue
                await self.session.drain(target, epoch)
                if epoch == self.session.epoch:
                    self.receiver.notify_playback_finished(
                        playback_position=(target - self.turn_base) / 48000, interrupted=False
                    )
                    self.turn_base = target
                    self.started = False
            else:
                self.receiving = True
                data = bytes(frame.data)
                step = frame.sample_rate * frame.num_channels * 2 // 10
                for i in range(0, len(data), step):
                    await self.session.push(
                        data[i : i + step],
                        {"sample_rate": frame.sample_rate, "channels": frame.num_channels},
                    )

    async def _output(self):
        origin = time.monotonic()
        tick = 0
        while True:
            await asyncio.sleep(max(0, origin + tick * 0.02 - time.monotonic()))
            timeline = self.session.timeline
            before = timeline.sent
            pcm = timeline.audio(960)
            await self.sync.push(rtc.AudioFrame(pcm.tobytes(), 48000, 1, 960))
            if timeline.sent > before and not self.started:
                self.started = True
                self.receiver.notify_playback_started()
            if tick % 2 == 0:
                await self.sync.push(
                    rtc.VideoFrame(512, 512, rtc.VideoBufferType.RGB24, timeline.frame.tobytes())
                )
            if timeline.sent > before:
                asyncio.get_running_loop().call_later(
                    self.audio.queued_duration,
                    self.session.playback,
                    timeline.sent,
                    self.session.epoch,
                )
            self.session.touch()
            tick += 1

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if hasattr(self, "receiver"):
            await self.receiver.aclose()
        if hasattr(self, "sync"):
            await self.sync.aclose()
        if hasattr(self, "audio"):
            await self.audio.aclose()
        if hasattr(self, "video"):
            await self.video.aclose()
        await self.room.disconnect()
