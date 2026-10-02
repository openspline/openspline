import base64
from types import SimpleNamespace as NS

from openspline.integrations import GeminiLive, GoogleADK, OpenAIRealtime


class Avatar:
    def __init__(self):
        self.calls = []

    def on(self, cb):
        self.cb = cb
        return lambda: None

    async def send_audio(self, data, **fmt):
        self.calls.append(("audio", data, fmt))

    async def end_turn(self, **kwargs):
        self.calls.append(("end",))

    async def interrupt(self):
        self.calls.append(("interrupt",))


async def test_openai_events_and_truncation():
    a = Avatar()
    sent = []

    class Conn:
        async def send(self, event):
            sent.append(event)

    adapter = OpenAIRealtime(a, Conn())
    await adapter.handle(
        {
            "type": "response.output_audio.delta",
            "item_id": "item",
            "content_index": 0,
            "delta": base64.b64encode(bytes(4800)).decode(),
        }
    )
    a.cb({"type": "playback", "samples": 2400})
    await adapter.handle({"type": "input_audio_buffer.speech_started"})
    assert a.calls[-1] == ("interrupt",)
    assert sent[0]["audio_end_ms"] == 50


async def test_gemini_all_parts_and_interrupt():
    a = Avatar()
    adapter = GeminiLive(a)
    await adapter.handle(
        NS(
            server_content=NS(
                model_turn=NS(
                    parts=[
                        NS(inline_data=NS(data=b"ab", mime_type="audio/pcm;rate=24000")),
                        NS(inline_data=NS(data=b"cd", mime_type="audio/pcm;rate=16000")),
                    ]
                ),
                turn_complete=True,
                interrupted=False,
            )
        )
    )
    assert len(a.calls) == 3 and a.calls[1][2]["sample_rate"] == 16000
    await adapter.handle({"serverContent": {"interrupted": True}})
    assert a.calls[-1] == ("interrupt",)


async def test_adk_preserves_events():
    a = Avatar()
    adapter = GoogleADK(a)
    events = [
        NS(content=NS(parts=[NS(inline_data=NS(data=b"ab", mime_type="audio/pcm;rate=24000"))])),
        NS(turn_complete=True),
        NS(tool="unchanged"),
    ]

    async def source():
        for e in events:
            yield e

    assert [e async for e in adapter.wrap(source())] == events
    assert a.calls[-1] == ("end",)
