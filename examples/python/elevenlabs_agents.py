"""Play one ElevenLabs agent response. Use the web demo for microphone conversations."""

import asyncio
import json
import os
import sys
from urllib.parse import urlencode

import httpx
from openspline import Openspline
from openspline.integrations import ElevenLabsAgents
from websockets import connect


async def main():
    agent_id = os.environ["ELEVENLABS_AGENT_ID"]
    url = "wss://api.elevenlabs.io/v1/convai/conversation?" + urlencode({"agent_id": agent_id})
    if api_key := os.getenv("ELEVENLABS_API_KEY"):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url",
                params={"agent_id": agent_id},
                headers={"xi-api-key": api_key},
            )
            response.raise_for_status()
            url = response.json()["signed_url"]

    async with Openspline().avatar(sys.argv[1]) as avatar:
        print(avatar.viewer_url, flush=True)
        await avatar.wait_for_viewer()
        async with connect(url, max_size=4 * 1024 * 1024) as connection:
            adapter = ElevenLabsAgents(avatar)
            await connection.send(json.dumps({"type": "conversation_initiation_client_data"}))

            async def receive_response():
                heard_audio = False
                async for message in connection:
                    event = json.loads(message)
                    await adapter.handle(event)
                    if event["type"] == "ping":
                        await connection.send(
                            json.dumps(
                                {"type": "pong", "event_id": event["ping_event"]["event_id"]}
                            )
                        )
                    elif event["type"] == "conversation_initiation_metadata":
                        await connection.send(
                            json.dumps(
                                {"type": "user_message", "text": "Introduce yourself briefly."}
                            )
                        )
                    elif event["type"] == "client_error":
                        raise RuntimeError(
                            "ElevenLabs rejected the conversation. Check your agent configuration."
                        )
                    elif event["type"] == "client_tool_call":
                        raise RuntimeError("Add your client tool handler before using this agent.")
                    elif event["type"] == "audio":
                        heard_audio = True
                        if event["audio_event"].get("is_final"):
                            return
                    elif event["type"] == "agent_response_complete" and heard_audio:
                        return
                raise RuntimeError("ElevenLabs closed before completing a response.")

            await asyncio.wait_for(receive_response(), 60)
            await avatar.end_turn()


asyncio.run(main())
