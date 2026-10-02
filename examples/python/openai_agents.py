import asyncio
import sys

from agents.realtime import RealtimeAgent, RealtimeRunner
from openspline import Openspline
from openspline.integrations import OpenAIAgents


async def main():
    async with Openspline().avatar(sys.argv[1]) as avatar:
        print(avatar.viewer_url, flush=True)
        await avatar.wait_for_viewer()
        adapter = OpenAIAgents(avatar)
        runner = RealtimeRunner(RealtimeAgent(name="Guide", instructions="Keep responses brief."))
        async with await runner.run(model_config={"playback_tracker": adapter.tracker}) as session:
            await session.send_message("Introduce yourself in two sentences.")
            try:
                async for event in adapter.wrap(session):
                    if event.type == "audio_end":
                        break
            finally:
                adapter.close()
            await avatar.end_turn()


asyncio.run(main())
