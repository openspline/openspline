import asyncio
import os
import sys

from google import genai
from google.genai import types
from openspline import Openspline
from openspline.integrations import GeminiLive


async def main():
    async with Openspline().avatar(sys.argv[1]) as avatar:
        print(avatar.viewer_url, flush=True)
        await avatar.wait_for_viewer()
        async with genai.Client().aio.live.connect(
            model=os.getenv("GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"),
            config=types.LiveConnectConfig(response_modalities=["AUDIO"]),
        ) as session:
            await session.send_client_content(
                turns=types.Content(
                    role="user", parts=[types.Part(text="Introduce yourself briefly.")]
                ),
                turn_complete=True,
            )
            async for event in GeminiLive(avatar).wrap(session.receive()):
                if event.server_content and event.server_content.turn_complete:
                    break
            await avatar.end_turn()


asyncio.run(main())
