"""Vocode output device for the pinned current AudioChunk worker API."""

import asyncio
from collections import deque

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import ChunkState


class AvatarOutputDevice(AbstractOutputDevice):
    """Use inside an open avatar context; its viewer owns output playback."""

    def __init__(self, avatar, sampling_rate=24000, max_pending_seconds=10):
        super().__init__(sampling_rate, AudioEncoding.LINEAR16)
        self._input_queue = asyncio.Queue(maxsize=64)
        self.avatar = avatar
        self.pending = deque()
        self.samples = 0
        self.played = 0
        self.limit = int(max_pending_seconds * 48000)
        self.changed = asyncio.Event()
        self.unsubscribe = avatar.on(self._feedback)
        self.tasks = set()

    def _feedback(self, event):
        if event.get("type") == "playback":
            self.played = event["samples"]
            while self.pending and self.pending[0][0] <= self.played:
                _, event = self.pending.popleft()
                chunk = event.payload
                if event.is_interrupted():
                    chunk.state = ChunkState.INTERRUPTED
                    chunk.on_interrupt()
                else:
                    chunk.state = ChunkState.PLAYED
                    chunk.on_play()
            self.changed.set()
        elif event.get("type") == "interrupted":
            self._clear()

    def _clear(self):
        while self.pending:
            _, event = self.pending.popleft()
            event.payload.state = ChunkState.INTERRUPTED
            event.payload.on_interrupt()
        self.samples = self.played = 0
        self.changed.set()

    async def _run_loop(self):
        while True:
            event = await self._input_queue.get()
            if event.is_interrupted():
                event.payload.state = ChunkState.INTERRUPTED
                event.payload.on_interrupt()
                continue
            while self.samples - self.played > self.limit:
                self.changed.clear()
                await self.changed.wait()
            self.samples += round(len(event.payload.data) / 2 / self.sampling_rate * 48000)
            self.pending.append((self.samples, event))
            await self.avatar.send_audio(event.payload.data, sample_rate=self.sampling_rate)
            # Vocode output chunks have no turn boundary. Flush each chunk, retaining
            # model history. This favors correctness for short utterances over throughput.
            await self.avatar.end_turn(drain=False)

    def interrupt(self):
        self._clear()
        while not self._input_queue.empty():
            event = self._input_queue.get_nowait()
            event.payload.state = ChunkState.INTERRUPTED
            event.payload.on_interrupt()
        task = asyncio.create_task(self.avatar.interrupt())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def terminate(self):
        self.unsubscribe()
        self._clear()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await super().terminate()
