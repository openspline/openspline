"""Text prompted realtime speech. The same adapter handles a microphone agent."""

import asyncio
import os
import sys

from openai import AsyncOpenAI
from openspline import Openspline
from openspline.integrations import OpenAIRealtime


async def main():
    async with Openspline().avatar(sys.argv[1]) as avatar:
        print(avatar.viewer_url, flush=True)
        await avatar.wait_for_viewer()
        async with AsyncOpenAI().realtime.connect(
            model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
        ) as connection:
            adapter = OpenAIRealtime(avatar, connection)
            await connection.session.update(
                session={"type": "realtime", "output_modalities": ["audio"]}
            )
            await connection.response.create(
                response={"instructions": "Introduce yourself in two sentences."}
            )
            try:
                async for event in adapter.wrap(connection):
                    if event.type == "error":
                        raise RuntimeError(event.error.message)
                    if event.type == "response.done":
                        break
            finally:
                adapter.close()
            await avatar.end_turn()


asyncio.run(main())
