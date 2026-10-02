import asyncio
import os
import sys

from google.adk.agents import Agent, LiveRequestQueue, RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from openspline import Openspline
from openspline.integrations import GoogleADK


async def main():
    sessions = InMemorySessionService()
    await sessions.create_session(app_name="openspline", user_id="demo", session_id="demo")
    runner = Runner(
        app_name="openspline",
        agent=Agent(
            name="guide",
            model=os.getenv("GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"),
            instruction="Be brief.",
        ),
        session_service=sessions,
    )
    queue = LiveRequestQueue()
    async with Openspline().avatar(sys.argv[1]) as avatar:
        print(avatar.viewer_url, flush=True)
        await avatar.wait_for_viewer()
        queue.send_content(
            types.Content(role="user", parts=[types.Part(text="Introduce yourself briefly.")])
        )
        try:
            events = runner.run_live(
                user_id="demo",
                session_id="demo",
                live_request_queue=queue,
                run_config=RunConfig(
                    streaming_mode=StreamingMode.BIDI, response_modalities=["AUDIO"]
                ),
            )
            async for event in GoogleADK(avatar).wrap(events):
                if event.turn_complete:
                    break
            await avatar.end_turn()
        finally:
            queue.close()


asyncio.run(main())
