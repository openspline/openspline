"""Place after TTS/realtime output and before transport.output()."""

import asyncio

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputImageRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from openspline import Openspline


class AvatarProcessor(FrameProcessor):
    def __init__(self, portrait, quality=None, client=None, **kwargs):
        super().__init__(**kwargs)
        self.client = client or Openspline()
        self.portrait = portrait
        self.quality = quality
        self.avatar = None
        self.task = None

    async def _output(self):
        import av

        resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
        media = await self.avatar.media()
        async for kind, frame in media:
            if kind == "audio":
                for out in resampler.resample(frame):
                    await self.push_frame(
                        TTSAudioRawFrame(
                            audio=out.to_ndarray().tobytes(), sample_rate=48000, num_channels=1
                        )
                    )
                media.acknowledge()
            else:
                out = OutputImageRawFrame(
                    image=frame.to_ndarray(format="rgb24").tobytes(),
                    size=(frame.width, frame.height),
                    format="RGB",
                )
                if frame.pts is not None:
                    out.pts = int(frame.pts * frame.time_base * 1e9)
                await self.push_frame(out)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, StartFrame):
            self.avatar = await self.client.avatar(self.portrait, self.quality).__aenter__()
            await self.push_frame(frame, direction)
            self.task = asyncio.create_task(self._output())
            return
        if isinstance(frame, TTSAudioRawFrame):
            await self.avatar.send_audio(
                frame.audio, sample_rate=frame.sample_rate, channels=frame.num_channels
            )
            return
        if isinstance(frame, TTSStoppedFrame):
            await self.avatar.end_turn()
        if isinstance(frame, InterruptionFrame):
            await self.avatar.interrupt()
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._close()
        await self.push_frame(frame, direction)

    async def _close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        if self.avatar:
            await self.avatar.close()
            self.avatar = None

    async def cleanup(self):
        await self._close()
        await super().cleanup()
