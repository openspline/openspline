"""Audio drives the shared playout clock; video samples that same position."""

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

import av
import numpy as np
from aiortc import MediaStreamTrack
from PIL import Image


@dataclass
class Segment:
    audio: np.ndarray
    video: np.ndarray
    fps: int
    epoch: int


class Timeline:
    def __init__(self, portrait):
        self.portrait = portrait
        self.queue = deque()
        self.current = None
        self.offset = 0
        self.frame = portrait
        self.slots = asyncio.Semaphore(3)
        self.epoch = 0
        self.submitted = 0
        self.sent = 0
        self.played = 0
        self.changed = asyncio.Event()
        self.origin = None
        self.idle_current = None
        self.idle_next = None
        self.idle_offset = 0
        self.idle_batches = 0
        self.idle_loop = None
        self.idle_reverse = False
        self.idle_source = None
        self.idle_elapsed = 0
        self.idle_rendered_tick = -1

    def clear_idle(self):
        self.idle_current = self.idle_next = None
        self.idle_offset = 0
        self.idle_batches = 0
        self.idle_loop = None
        self.idle_reverse = False
        self.idle_source = None
        self.idle_elapsed = 0
        self.idle_rendered_tick = -1

    def put_idle(self, frames, fps, epoch):
        if epoch == self.epoch:
            self.idle_next = (frames, fps)
            self.idle_batches += 1
            # The first batch settles the mouth after speech. Reuse the second
            # batch so fresh random diffusion noise cannot slowly deform a face.
            if self.idle_batches == 2:
                self.idle_loop = (frames, fps)

    def _idle_frame(self, frame):
        t = self.idle_elapsed / 48000
        height, width = frame.shape[:2]
        unit = min(height, width) / 512
        zoom = 1.012 + 0.002 * math.sin(t * 0.75)
        x = 1.2 * unit * math.sin(t * 0.7)
        y = 1.1 * unit * math.sin(t * 0.9)
        # Resample the original frame, never the previously animated result.
        # This provides continuous, gentle motion even if GPU inference is late.
        image = Image.fromarray(frame).transform(
            (width, height),
            Image.Transform.AFFINE,
            (
                1 / zoom,
                0,
                width * (1 - 1 / zoom) / 2 - x,
                0,
                1 / zoom,
                height * (1 - 1 / zoom) / 2 - y,
            ),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(image)

    def advance_idle(self, samples):
        if self.idle_current is None:
            if self.idle_next is None:
                if self.idle_loop is not None:
                    frames, fps = self.idle_loop
                    self.idle_reverse = not self.idle_reverse
                    self.idle_current = (frames[::-1] if self.idle_reverse else frames, fps)
                    self.idle_offset = 0
            else:
                self.idle_current, self.idle_next = self.idle_next, None
                self.idle_offset = 0
        if self.idle_current is not None:
            frames, fps = self.idle_current
            self.idle_source = frames[min(len(frames) - 1, self.idle_offset * fps // 48000)]
            self.idle_offset += samples
            if self.idle_offset * fps >= len(frames) * 48000:
                self.idle_current = None
        if self.idle_source is None:
            self.idle_source = self.frame
        self.idle_elapsed += samples
        tick = (self.idle_elapsed - 1) // 1920
        if tick != self.idle_rendered_tick:
            self.frame = self._idle_frame(self.idle_source)
            self.idle_rendered_tick = tick
        return True

    async def put(self, segment):
        await self.slots.acquire()
        if segment.epoch != self.epoch:
            self.slots.release()
            return
        self.clear_idle()
        self.queue.append(segment)
        self.submitted += len(segment.audio)

    def clear(self, epoch):
        self.epoch = epoch
        while self.queue:
            self.queue.popleft()
            self.slots.release()
        self.current = None
        self.offset = 0
        # Keep the last displayed pose until new idle or speech frames arrive.
        self.clear_idle()
        self.submitted = self.sent = self.played = 0
        self.changed.set()

    def audio(self, n):
        out = np.zeros(n, dtype=np.int16)
        written = 0
        while written < n:
            if self.current is None:
                if not self.queue:
                    if not written:
                        self.advance_idle(n)
                    break
                self.current = self.queue.popleft()
                self.slots.release()
                self.offset = 0
            seg = self.current
            count = min(n - written, len(seg.audio) - self.offset)
            out[written : written + count] = seg.audio[self.offset : self.offset + count]
            index = min(len(seg.video) - 1, int(self.offset * seg.fps / 48000))
            self.frame = seg.video[index]
            written += count
            self.offset += count
            self.sent += count
            if self.offset == len(seg.audio):
                self.current = None
        return out

    async def pace(self, seconds):
        if self.origin is None:
            self.origin = time.monotonic()
        await asyncio.sleep(max(0, self.origin + seconds - time.monotonic()))


class AudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, timeline):
        super().__init__()
        self.timeline = timeline
        self.pts = 0

    async def recv(self):
        await self.timeline.pace(self.pts / 48000)
        data = self.timeline.audio(960)
        frame = av.AudioFrame.from_ndarray(data.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 48000
        frame.pts = self.pts
        frame.time_base = Fraction(1, 48000)
        self.pts += 960
        return frame


class VideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, timeline):
        super().__init__()
        self.timeline = timeline
        self.pts = 0

    async def recv(self):
        await self.timeline.pace(self.pts / 90000)
        frame = av.VideoFrame.from_ndarray(self.timeline.frame, format="rgb24")
        frame.pts = self.pts
        frame.time_base = Fraction(1, 90000)
        self.pts += 3600
        return frame
