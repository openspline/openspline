"""Optional server-owned voice demo. Provider credentials never enter the browser."""

import asyncio
import base64
import os
import re
import secrets

from fastapi import WebSocket, WebSocketDisconnect


async def run_demo(ws: WebSocket, session, provider, remove):
    await ws.accept()
    authenticated = False
    try:
        auth = await asyncio.wait_for(ws.receive_json(), 5)
        if not session or not secrets.compare_digest(
            auth.get("token", ""), session.publisher_token
        ):
            await ws.close(4401)
            return
        if session.publisher_connected:
            await ws.close(4409)
            return
        session.publisher_connected = True
        authenticated = True

        async def output():
            while True:
                await ws.send_json(await session.events.get())

        output_task = asyncio.create_task(output())
        session.emit({"type": "ready"})

        async def microphone(send):
            while True:
                raw = await ws.receive_bytes()
                session.touch()
                if len(raw) > 48000:
                    raise ValueError("Microphone packet too large")
                await send(raw)

        async def feed(raw, rate=24000):
            epoch = session.epoch
            for offset in range(0, len(raw), rate // 5):
                if epoch != session.epoch:
                    return
                await session.push(raw[offset : offset + rate // 5], {"sample_rate": rate})

        async def openai():
            from openai import AsyncOpenAI

            async with AsyncOpenAI().realtime.connect(
                model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
            ) as conn:
                await conn.session.update(
                    session={
                        "type": "realtime",
                        "instructions": "You are a friendly voice assistant. Keep answers brief.",
                        "audio": {
                            "input": {"format": {"type": "audio/pcm", "rate": 24000}},
                            "output": {
                                "format": {"type": "audio/pcm", "rate": 24000},
                                "voice": "marin",
                            },
                        },
                    }
                )

                async def send(raw):
                    await conn.input_audio_buffer.append(audio=base64.b64encode(raw).decode())

                task = asyncio.create_task(microphone(send))
                item = None
                generated = base = 0
                index = 0
                try:
                    async for event in conn:
                        if event.type == "response.output_audio.delta":
                            if event.item_id != item:
                                item = event.item_id
                                base = generated
                            index = event.content_index
                            raw = base64.b64decode(event.delta)
                            generated += len(raw) / 48
                            await feed(raw)
                        elif event.type == "response.output_audio.done":
                            await session.end_turn()
                        elif event.type == "input_audio_buffer.speech_started":
                            played = max(0, session.timeline.played / 48 - base)
                            await session.interrupt()
                            if item:
                                await conn.send(
                                    {
                                        "type": "conversation.item.truncate",
                                        "item_id": item,
                                        "content_index": index,
                                        "audio_end_ms": int(played),
                                    }
                                )
                            item = None
                            generated = base = 0
                        elif event.type == "error":
                            session.emit({"type": "error", "message": event.error.message})
                        if task.done():
                            await task
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        async def gemini():
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            async with client.aio.live.connect(
                model=os.getenv(
                    "GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"
                ),
                config={
                    "response_modalities": ["AUDIO"],
                    "system_instruction": "Be friendly and concise.",
                },
            ) as conn:

                async def send(raw):
                    await conn.send_realtime_input(
                        audio=types.Blob(data=raw, mime_type="audio/pcm;rate=24000")
                    )

                task = asyncio.create_task(microphone(send))
                try:
                    while True:
                        async for event in conn.receive():
                            content = event.server_content
                            if not content:
                                continue
                            if content.interrupted:
                                await session.interrupt()
                                continue
                            for part in content.model_turn.parts if content.model_turn else []:
                                if part.inline_data and part.inline_data.mime_type.startswith(
                                    "audio/"
                                ):
                                    match = re.search(r"rate=(\d+)", part.inline_data.mime_type)
                                    await feed(
                                        part.inline_data.data, int(match[1]) if match else 24000
                                    )
                            if content.turn_complete:
                                await session.end_turn()
                        if task.done():
                            await task
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    await client.aio.aclose()

        try:
            await asyncio.wait_for(session.viewer_ready.wait(), 60)
            if provider == "openai":
                await openai()
            elif provider == "gemini":
                await gemini()
            else:
                raise ValueError("Unknown demo provider")
        except (ImportError, KeyError):
            await ws.send_json(
                {
                    "type": "error",
                    "message": "Install the optional provider SDK and configure its server-side API key.",
                }
            )
        finally:
            output_task.cancel()
            await asyncio.gather(output_task, return_exceptions=True)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if authenticated:
            session.publisher_connected = False
            await remove(session.id)
