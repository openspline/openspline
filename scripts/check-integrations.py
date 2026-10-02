"""Run inside the separately locked framework environment."""

import asyncio

from agents.realtime.events import RealtimeAudio, RealtimeAudioEnd
from agents.realtime.model_events import RealtimeModelAudioEvent
from google.adk.events import Event
from google.genai import types
from openspline.integrations import GeminiLive, GoogleADK, OpenAIAgents
from openspline.integrations.livekit import AvatarSession
from openspline.integrations.pipecat import AvatarProcessor
from pipecat.frames.frames import InterruptionFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection


class Avatar:
    def __init__(self):
        self.calls = []

    def on(self, cb):
        self.callback = cb
        return lambda: None

    async def send_audio(self, data, **fmt):
        self.calls.append(("audio", bytes(data), fmt))

    async def end_turn(self, **kwargs):
        self.calls.append(("end",))

    async def interrupt(self):
        self.calls.append(("interrupt",))


async def main():
    a = Avatar()
    adapter = OpenAIAgents(a)
    event = RealtimeAudio(
        audio=RealtimeModelAudioEvent(
            data=bytes(4800), response_id="r", item_id="i", content_index=0
        ),
        item_id="i",
        content_index=0,
        info=None,
    )
    await adapter.handle(event)
    a.callback({"type": "playback", "samples": 2400})
    assert adapter.tracker.get_state()["elapsed_ms"] == 50
    a.callback({"type": "playback", "samples": 4800})
    assert adapter.tracker.get_state()["elapsed_ms"] == 100
    await adapter.handle(RealtimeAudioEnd(info=None, item_id="i", content_index=0))
    google = GeminiLive(a)
    await google.handle(
        types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=b"\0\0", mime_type="audio/pcm;rate=24000")
                        )
                    ],
                ),
                turn_complete=True,
            )
        )
    )
    await GoogleADK(a).handle(
        Event(
            author="agent",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=b"\0\0", mime_type="audio/pcm;rate=24000")
                    )
                ],
            ),
        )
    )
    from pipecat.clocks.system_clock import SystemClock
    from pipecat.processors.frame_processor import FrameProcessorSetup
    from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

    manager = TaskManager()
    manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    processor = AvatarProcessor("unused.png")
    processor.avatar = a
    await processor.setup(FrameProcessorSetup(clock=SystemClock(), task_manager=manager))

    async def push(frame, direction=None):
        pass

    processor.push_frame = push
    await processor.process_frame(
        TTSAudioRawFrame(audio=b"\0\0", sample_rate=24000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await processor.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    assert a.calls[-1] == ("interrupt",)
    processor.avatar = None
    await asyncio.sleep(0.01)
    await processor.cleanup()
    plugin = AvatarSession("unused.png")
    assert plugin.provider == "openspline"
    print("OpenAI Agents, Gemini, Google ADK, Pipecat and LiveKit SDK contracts passed")


asyncio.run(main())
