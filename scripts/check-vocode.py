import asyncio

from openspline.integrations.vocode import AvatarOutputDevice
from vocode.streaming.output_device.audio_chunk import AudioChunk, ChunkState
from vocode.streaming.utils.worker import InterruptibleEvent


class Avatar:
    def on(self, cb):
        self.callback = cb
        return lambda: None

    async def send_audio(self, *args, **kwargs):
        pass

    async def end_turn(self, **kwargs):
        pass

    async def interrupt(self):
        self.callback({"type": "interrupted"})


async def main():
    a = Avatar()
    out = AvatarOutputDevice(a)
    task = out.start()
    chunk = AudioChunk(bytes(4800))
    played = []
    chunk.on_play = lambda: played.append(True)
    out.consume_nonblocking(InterruptibleEvent(chunk))
    await asyncio.sleep(0.02)
    assert chunk.state == ChunkState.UNPLAYED
    a.callback({"type": "playback", "samples": 4800})
    assert played and chunk.state == ChunkState.PLAYED
    other = AudioChunk(bytes(4800))
    out.consume_nonblocking(InterruptibleEvent(other))
    await asyncio.sleep(0.02)
    out.interrupt()
    await asyncio.sleep(0.02)
    assert other.state == ChunkState.INTERRUPTED
    await out.terminate()
    await asyncio.gather(task, return_exceptions=True)
    print("Vocode actual AudioChunk playback and interruption contract passed")


asyncio.run(main())
