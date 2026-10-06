"""ElevenLabs Agents WebSocket events, without a provider SDK dependency."""

import base64
import struct

from .events import EventAdapter, value


def audio_rate(audio_format):
    if audio_format == "ulaw_8000":
        return 8000
    if audio_format not in {f"pcm_{rate}" for rate in (8000, 16000, 22050, 24000, 44100, 48000)}:
        raise ValueError("Unsupported ElevenLabs audio format; use PCM or ulaw_8000.")
    return int(audio_format[4:])


def decode_audio(encoded, audio_format):
    raw = base64.b64decode(encoded, validate=True)
    if audio_format == "ulaw_8000":
        samples = []
        for byte in raw:
            byte = ~byte & 255
            sample = (((byte & 15) << 3) + 132) << ((byte >> 4) & 7)
            samples.append(132 - sample if byte & 128 else sample - 132)
        return struct.pack(f"<{len(samples)}h", *samples)
    return raw


class ElevenLabsAgents(EventAdapter):
    """Forward decoded WebSocket events; the application owns pings, tools, and input.

    Include the initiation metadata event to select the negotiated audio format.
    If attaching after initiation, pass that format explicitly.
    """

    def __init__(self, avatar, *, audio_format="pcm_16000"):
        super().__init__(avatar)
        self.audio_format = audio_format
        self.sample_rate = audio_rate(audio_format)
        self.interrupted_id = -1
        self.pending = False

    async def handle(self, event):
        kind = value(event, "type")
        if kind == "conversation_initiation_metadata":
            metadata = value(event, "conversation_initiation_metadata_event")
            self.audio_format = value(metadata, "agent_output_audio_format")
            self.sample_rate = audio_rate(self.audio_format)
        elif kind == "interruption":
            interruption = value(event, "interruption_event")
            self.interrupted_id = max(self.interrupted_id, value(interruption, "event_id", -1))
            self.pending = False
            await self.avatar.interrupt()
        elif kind == "audio":
            audio = value(event, "audio_event")
            if value(audio, "event_id", 0) <= self.interrupted_id:
                return
            raw = decode_audio(value(audio, "audio_base_64"), self.audio_format)
            if raw:
                await self.avatar.send_audio(raw, sample_rate=self.sample_rate)
                self.pending = True
            if value(audio, "is_final", False):
                await self._finish()
        elif kind == "agent_response_complete":
            complete = value(event, "agent_response_complete_event")
            if value(complete, "event_id", 0) > self.interrupted_id:
                await self._finish()

    async def _finish(self):
        if self.pending:
            await self.avatar.end_turn(drain=False)
            self.pending = False
