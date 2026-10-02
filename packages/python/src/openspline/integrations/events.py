import base64
import re


def value(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


class EventAdapter:
    def __init__(self, avatar):
        self.avatar = avatar

    async def wrap(self, events):
        """Preserve every provider event for the application's existing event loop."""
        async for event in events:
            await self.handle(event)
            yield event


class OpenAIRealtime(EventAdapter):
    def __init__(self, avatar, connection=None):
        super().__init__(avatar)
        self.connection = connection
        self.item = None
        self.content_index = 0
        self.played = 0
        self.generated = 0
        self.base = 0
        self.unsubscribe = avatar.on(self._playback)

    def _playback(self, event):
        if event.get("type") == "playback":
            self.played = event["samples"] / 48000

    async def handle(self, event):
        kind = value(event, "type")
        if kind == "response.output_audio.delta":
            item = value(event, "item_id")
            if item != self.item:
                self.item = item
                self.base = self.generated
            self.content_index = value(event, "content_index", 0)
            raw = base64.b64decode(value(event, "delta"))
            self.generated += len(raw) / 48000
            await self.avatar.send_audio(raw, sample_rate=24000)
        elif kind == "response.output_audio.done":
            await self.avatar.end_turn(drain=False)
        elif kind == "input_audio_buffer.speech_started":
            heard = max(0, self.played - self.base)
            await self.avatar.interrupt()
            if self.connection and self.item:
                await self.connection.send(
                    {
                        "type": "conversation.item.truncate",
                        "item_id": self.item,
                        "content_index": self.content_index,
                        "audio_end_ms": int(heard * 1000),
                    }
                )
            self.item = None
            self.generated = self.played = self.base = 0

    def close(self):
        self.unsubscribe()


class OpenAIAgents(EventAdapter):
    """Use .tracker in RealtimeRunner.run(model_config={"playback_tracker": ...})."""

    def __init__(self, avatar, playback_tracker=None):
        from agents.realtime import RealtimePlaybackTracker

        super().__init__(avatar)
        self.tracker = playback_tracker or RealtimePlaybackTracker()
        self.spans = []
        self.generated = 0
        self.played = 0
        self.unsubscribe = avatar.on(self._playback)

    def _playback(self, event):
        if event.get("type") != "playback":
            return
        new = event["samples"] / 48
        for start, end, item, index in self.spans:
            duration = max(0, min(new, end) - max(self.played, start))
            if duration:
                self.tracker.on_play_ms(item, index, duration)
        self.played = max(self.played, new)
        self.spans = [span for span in self.spans if span[1] > self.played]

    async def handle(self, event):
        kind = value(event, "type")
        if kind == "audio":
            audio = value(event, "audio")
            raw = value(audio, "data", audio)
            duration = len(raw) / 48
            self.spans.append(
                (
                    self.generated,
                    self.generated + duration,
                    value(event, "item_id"),
                    value(event, "content_index", 0),
                )
            )
            self.generated += duration
            await self.avatar.send_audio(raw, sample_rate=24000)
        elif kind == "audio_end":
            await self.avatar.end_turn(drain=False)
        elif kind == "audio_interrupted":
            await self.avatar.interrupt()
            self.spans = []
            self.generated = self.played = 0

    def close(self):
        self.unsubscribe()


class GeminiLive(EventAdapter):
    async def handle(self, event):
        content = value(event, "server_content", value(event, "serverContent"))
        if not content:
            return
        if value(content, "interrupted", False):
            await self.avatar.interrupt()
            return
        turn = value(content, "model_turn", value(content, "modelTurn"))
        for part in value(turn, "parts", []) or []:
            blob = value(part, "inline_data", value(part, "inlineData"))
            mime = value(blob, "mime_type", value(blob, "mimeType", ""))
            if blob and mime.startswith("audio/"):
                raw = value(blob, "data")
                raw = base64.b64decode(raw) if isinstance(raw, str) else raw
                match = re.search(r"rate=(\d+)", mime)
                await self.avatar.send_audio(raw, sample_rate=int(match[1]) if match else 24000)
        if value(content, "turn_complete", value(content, "turnComplete", False)):
            await self.avatar.end_turn(drain=False)


class GoogleADK(EventAdapter):
    async def handle(self, event):
        if value(event, "interrupted", False):
            await self.avatar.interrupt()
            return
        for part in value(value(event, "content"), "parts", []) or []:
            blob = value(part, "inline_data")
            mime = value(blob, "mime_type", "")
            if blob and mime.startswith("audio/"):
                match = re.search(r"rate=(\d+)", mime)
                await self.avatar.send_audio(
                    value(blob, "data"), sample_rate=int(match[1]) if match else 24000
                )
        if value(event, "turn_complete", False):
            await self.avatar.end_turn(drain=False)
