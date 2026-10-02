"""Use a prerecorded chunk to verify the Vocode output-device contract."""

import asyncio
import sys
import wave

from openspline import Openspline
from openspline.integrations.vocode import AvatarOutputDevice
from vocode.streaming.output_device.audio_chunk import AudioChunk
from vocode.streaming.utils.worker import InterruptibleEvent


async def main():
    with wave.open(sys.argv[2]) as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Use mono PCM16 WAV")
        rate = wav.getframerate()
        data = wav.readframes(wav.getnframes())
    async with Openspline().avatar(sys.argv[1]) as avatar:
        print(avatar.viewer_url, flush=True)
        await avatar.wait_for_viewer()
        output = AvatarOutputDevice(avatar, sampling_rate=rate)
        task = output.start()
        done = asyncio.Event()
        chunk = AudioChunk(data)
        chunk.on_play = done.set
        try:
            output.consume_nonblocking(InterruptibleEvent(chunk))
            await asyncio.wait_for(done.wait(), 180)
        finally:
            await output.terminate()
            await asyncio.gather(task, return_exceptions=True)


asyncio.run(main())
