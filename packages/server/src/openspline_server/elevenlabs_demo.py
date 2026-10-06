"""Backend-only ElevenLabs Agents bridge for the microphone demo."""

import asyncio
import base64
import json
from fractions import Fraction
from urllib.parse import urlencode

from .provider_errors import avatar_operation

API_URL = "https://api.elevenlabs.io/v1/convai/conversation"
WS_URL = "wss://api.elevenlabs.io/v1/convai/conversation"


class ElevenLabsDemoError(ValueError):
    """A safe, actionable error that can be shown in the demo."""


def audio_rate(audio_format):
    if audio_format == "ulaw_8000":
        return 8000
    if audio_format not in {f"pcm_{rate}" for rate in (8000, 16000, 22050, 24000, 44100, 48000)}:
        raise ElevenLabsDemoError("Configure your ElevenLabs agent to use PCM or ulaw_8000 audio.")
    return int(audio_format[4:])


class MicrophoneAudio:
    """Resample the browser's 24 kHz PCM16 continuously, preserving filter state."""

    def __init__(self, audio_format):
        import av

        self.rate = audio_rate(audio_format)
        self.position = 0
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=self.rate)
        self.encoder = None
        if audio_format == "ulaw_8000":
            self.encoder = av.CodecContext.create("pcm_mulaw", "w")
            self.encoder.sample_rate = self.rate
            self.encoder.layout = "mono"
            self.encoder.format = "s16"

    def push(self, raw):
        import av
        import numpy as np

        frame = av.AudioFrame.from_ndarray(
            np.frombuffer(raw, dtype="<i2").reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = 24000
        frame.pts = self.position
        frame.time_base = Fraction(1, 24000)
        self.position += len(raw) // 2
        frames = self.resampler.resample(frame)
        if self.encoder:
            return b"".join(
                bytes(packet) for item in frames for packet in self.encoder.encode(item)
            )
        return b"".join(item.to_ndarray().astype("<i2").tobytes() for item in frames)


async def run_elevenlabs_demo(session, microphone, feed, api_key, agent_id):
    import httpx
    import numpy as np
    from websockets.asyncio.client import connect

    from .demo import run_duplex

    url = f"{WS_URL}?{urlencode({'agent_id': agent_id})}"
    if api_key:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                f"{API_URL}/get-signed-url",
                params={"agent_id": agent_id},
                headers={"xi-api-key": api_key},
            )
            response.raise_for_status()
            url = response.json()["signed_url"]

    async with connect(url, max_size=4 * 1024 * 1024, open_timeout=20) as conn:
        await conn.send(json.dumps({"type": "conversation_initiation_client_data"}))
        ready = asyncio.Event()
        converter = None
        output_format, output_rate = None, None
        interrupted_id, pending = -1, False

        async def finish():
            nonlocal pending
            if pending:
                await avatar_operation(session.end_turn())
                pending = False

        async def receive():
            nonlocal converter, output_format, output_rate, interrupted_id, pending
            async for message in conn:
                event = json.loads(message)
                kind = event.get("type")
                if kind == "conversation_initiation_metadata":
                    metadata = event["conversation_initiation_metadata_event"]
                    output_format = metadata["agent_output_audio_format"]
                    output_rate = audio_rate(output_format)
                    converter = MicrophoneAudio(metadata["user_input_audio_format"])
                    ready.set()
                    session.emit({"type": "ready"})
                elif kind == "ping":
                    await conn.send(
                        json.dumps({"type": "pong", "event_id": event["ping_event"]["event_id"]})
                    )
                elif kind == "audio":
                    if not ready.is_set():
                        raise ElevenLabsDemoError(
                            "ElevenLabs sent audio before its format was negotiated. Start a new session."
                        )
                    audio = event["audio_event"]
                    if audio["event_id"] <= interrupted_id:
                        continue
                    raw = base64.b64decode(audio["audio_base_64"], validate=True)
                    if output_format == "ulaw_8000":
                        data = np.bitwise_not(np.frombuffer(raw, dtype=np.uint8)).astype(np.int32)
                        samples = (((data & 15) << 3) + 132) << ((data >> 4) & 7)
                        raw = (
                            np.where(data & 128, 132 - samples, samples - 132)
                            .astype("<i2")
                            .tobytes()
                        )
                    if raw:
                        await avatar_operation(feed(raw, output_rate))
                        pending = True
                    if audio.get("is_final"):
                        await finish()
                elif kind == "agent_response_complete":
                    if event["agent_response_complete_event"]["event_id"] > interrupted_id:
                        await finish()
                elif kind == "interruption":
                    interrupted_id = max(interrupted_id, event["interruption_event"]["event_id"])
                    pending = False
                    await avatar_operation(session.interrupt())
                elif kind == "client_tool_call":
                    call = event["client_tool_call"]
                    if call.get("expects_response", True):
                        await conn.send(
                            json.dumps(
                                {
                                    "type": "client_tool_result",
                                    "tool_call_id": call["tool_call_id"],
                                    "is_error": True,
                                    "result": "The avatar demo cannot execute client tools. Use an application with a client tool handler.",
                                }
                            )
                        )
                elif kind == "client_error":
                    raise ElevenLabsDemoError(
                        "ElevenLabs rejected the conversation. Check your agent settings, required dynamic variables, and access permissions."
                    )

        async def send(raw):
            converted = converter.push(raw)
            if converted:
                await conn.send(
                    json.dumps({"user_audio_chunk": base64.b64encode(converted).decode()})
                )

        async def input_audio():
            await asyncio.wait_for(ready.wait(), 20)
            await microphone(send)

        await run_duplex(receive(), input_audio())
