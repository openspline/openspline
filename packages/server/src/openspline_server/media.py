"""Audio drives the shared playout clock; video samples that same position."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

import av
import numpy as np
from aiortc import MediaStreamTrack


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

    async def put(self, segment):
        await self.slots.acquire()
        if segment.epoch != self.epoch:
            self.slots.release()
            return
        self.queue.append(segment)
        self.submitted += len(segment.audio)

    def clear(self, epoch):
        self.epoch = epoch
        while self.queue:
            self.queue.popleft()
            self.slots.release()
        self.current = None
        self.offset = 0
        self.frame = self.portrait
        self.submitted = self.sent = self.played = 0
        self.changed.set()

    def audio(self, n):
        out = np.zeros(n, dtype=np.int16)
        written = 0
        while written < n:
            if self.current is None:
                if not self.queue:
                    self.frame = self.portrait
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
