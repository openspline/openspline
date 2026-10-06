"""Voice demo with per-connection credentials and a server environment fallback."""

import asyncio
import base64
import logging
import os
import re
import secrets

from fastapi import WebSocket, WebSocketDisconnect

from .elevenlabs_demo import ElevenLabsDemoError, run_elevenlabs_demo
from .provider_errors import (
    PROVIDER_NAMES,
    avatar_operation,
    connection_failure,
    log_connection_failure,
)

logger = logging.getLogger(__name__)
OPENAI_CONNECTION_ATTEMPTS = 3
PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
}


def configured_providers():
    # The browser only needs to know whether a fallback exists, never its value.
    providers = {
        provider: {"configured": bool(os.getenv(variable, "").strip())}
        for provider, variable in PROVIDER_API_KEYS.items()
    }
    providers["elevenlabs"]["agent_configured"] = bool(os.getenv("ELEVENLABS_AGENT_ID", "").strip())
    return providers


def demo_api_key(provider, supplied):
    if provider not in PROVIDER_API_KEYS:
        raise ValueError("Choose OpenAI Realtime, Gemini Live, or ElevenLabs Agents.")
    if supplied is not None and (not isinstance(supplied, str) or len(supplied) > 4096):
        raise ValueError("Enter a valid provider API key.")
    variable = PROVIDER_API_KEYS[provider]
    key = (supplied or "").strip() or os.getenv(variable, "").strip()
    if not key and provider != "elevenlabs":
        raise ValueError(f"Enter a {PROVIDER_NAMES[provider]} API key or set {variable} in .env.")
    return key


def demo_agent_id(supplied):
    if supplied is not None and (not isinstance(supplied, str) or len(supplied) > 256):
        raise ValueError("Enter a valid ElevenLabs agent ID.")
    agent_id = (supplied or "").strip() or os.getenv("ELEVENLABS_AGENT_ID", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", agent_id):
        raise ValueError("Enter an ElevenLabs agent ID or set ELEVENLABS_AGENT_ID in .env.")
    return agent_id


async def run_duplex(receive, microphone):
    """Either side closing ends the bridge, including an otherwise idle microphone."""
    tasks = [asyncio.create_task(receive), asyncio.create_task(microphone)]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            await task
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class OpenAIDemoEvents:
    def __init__(self, session, connection, feed):
        self.session, self.connection, self.feed = session, connection, feed
        self.item = self.provider_session_id = None
        self.generated = self.base = self.index = 0
        self.warned = self.audio_pending = False

    def provider_error(self, error, event_id=None, response_id=None):
        error_type = getattr(error, "type", None)
        code = getattr(error, "code", None)
        temporary = error_type == "server_error" or code == "server_error"
        message = (
            "OpenAI Realtime had a temporary server error. Please try speaking again."
            if temporary
            else "OpenAI Realtime: "
            + (
                getattr(error, "message", None)
                or code
                or error_type
                or "The response failed. Please try speaking again."
            )
        )
        # Keep provider failures separate from inference/playback errors. A
        # Realtime error event normally leaves its connection open for more turns.
        logger.warning(
            "OpenAI Realtime error: avatar_session=%s provider_session=%s event_id=%s response_id=%s type=%s code=%s",
            self.session.id,
            self.provider_session_id,
            event_id,
            response_id,
            error_type,
            code,
        )
        self.session.emit(
            {
                "type": "warning",
                "provider": "openai",
                "code": code or error_type,
                "message": message,
                "provider_session_id": self.provider_session_id,
                "provider_event_id": event_id,
            }
        )
        self.warned = True

    async def handle(self, event):
        if event.type in {"session.created", "session.updated"}:
            self.provider_session_id = getattr(event.session, "id", self.provider_session_id)
        elif event.type == "response.output_audio.delta":
            if event.item_id != self.item:
                self.item = event.item_id
                self.base = self.generated
            self.index = event.content_index
            raw = base64.b64decode(event.delta)
            self.generated += len(raw) / 48
            self.audio_pending = True
            await avatar_operation(self.feed(raw))
            if self.warned:
                self.warned = False
                self.session.emit(
                    {
                        "type": "provider_status",
                        "provider": "openai",
                        "message": "OpenAI Realtime resumed. Speak naturally; interrupt at any time.",
                    }
                )
        elif event.type == "response.output_audio.done":
            await avatar_operation(self.session.end_turn())
            self.audio_pending = False
        elif event.type == "input_audio_buffer.speech_started":
            played = min(
                self.generated - self.base, max(0, self.session.timeline.played / 48 - self.base)
            )
            await avatar_operation(self.session.interrupt())
            if self.item:
                await self.connection.send(
                    {
                        "type": "conversation.item.truncate",
                        "item_id": self.item,
                        "content_index": self.index,
                        "audio_end_ms": int(played),
                    }
                )
            self.item = None
            self.generated = self.base = 0
            self.audio_pending = False
        elif event.type == "error":
            self.provider_error(event.error, event.event_id)
        elif event.type == "response.done" and event.response.status == "failed":
            # A failure can be delivered only in response.done, without error.
            # Flush any partial audio once, then keep listening for another turn.
            if self.audio_pending:
                await avatar_operation(self.session.end_turn())
                self.audio_pending = False
            details = event.response.status_details
            self.provider_error(getattr(details, "error", None), event.event_id, event.response.id)


async def run_openai_demo(session, microphone, feed, api_key):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    active_connection = send_failure = None

    async def send(raw):
        nonlocal active_connection
        conn, failure = active_connection, send_failure
        # Continue draining the microphone while connecting/retrying. Replaying
        # queued speech into a fresh conversation can duplicate a user's request.
        if conn is None:
            return
        try:
            await conn.input_audio_buffer.append(audio=base64.b64encode(raw).decode())
        except Exception as exc:
            if conn is active_connection:
                active_connection = None
                if not failure.done():
                    failure.set_result(exc)

    async def connect():
        nonlocal active_connection, send_failure
        connected_before = False
        for attempt in range(1, OPENAI_CONNECTION_ATTEMPTS + 1):
            send_failure = asyncio.get_running_loop().create_future()
            stage = "connect"
            try:
                async with client.realtime.connect(
                    model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
                ) as conn:
                    stage = "configure"
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
                    active_connection = conn
                    stage = "stream"
                    if not connected_before:
                        # Only enable the browser's microphone after the provider
                        # is connected; otherwise the first spoken words get lost.
                        session.emit({"type": "ready"})
                    if attempt > 1:
                        session.emit(
                            {
                                "type": "provider_status",
                                "provider": "openai",
                                "message": (
                                    "OpenAI Realtime reconnected with a new conversation. Please repeat your last message."
                                    if connected_before
                                    else "OpenAI Realtime connected. Speak naturally; interrupt at any time."
                                ),
                            }
                        )
                    connected_before = True
                    events = OpenAIDemoEvents(session, conn, feed)

                    async def receive():
                        async for event in conn:
                            await events.handle(event)

                    async def failed_send():
                        raise await send_failure

                    await run_duplex(receive(), failed_send())
                    return
            except Exception as exc:
                active_connection = None
                failure = connection_failure(exc, "openai")
                if not failure["retryable"] or attempt == OPENAI_CONNECTION_ATTEMPTS:
                    raise
                log_connection_failure(exc, "openai", session, stage=stage, attempt=attempt)
                await avatar_operation(session.interrupt())
                session.emit(
                    {
                        "type": "warning",
                        "provider": "openai",
                        "code": "provider_reconnecting",
                        "message": f"{failure['message']} Reconnecting ({attempt}/{OPENAI_CONNECTION_ATTEMPTS - 1})…",
                    }
                )
                await asyncio.sleep(0.5 * 2 ** (attempt - 1))
            finally:
                active_connection = None
                send_failure.cancel()

    try:
        await run_duplex(connect(), microphone(send))
    finally:
        await client.close()


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

        try:
            api_key = demo_api_key(provider, auth.get("api_key"))
            agent_id = demo_agent_id(auth.get("agent_id")) if provider == "elevenlabs" else None
        except ValueError as exc:
            await ws.send_json(
                {"type": "error", "code": "configuration", "fatal": True, "message": str(exc)}
            )
            return

        async def output():
            while True:
                await ws.send_json(await session.events.get())

        output_task = asyncio.create_task(output())

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

        async def gemini():
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            async with client.aio.live.connect(
                model=os.getenv(
                    "GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"
                ),
                config={
                    "response_modalities": ["AUDIO"],
                    "system_instruction": "Be friendly and concise.",
                },
            ) as conn:
                session.emit({"type": "ready"})

                async def send(raw):
                    await conn.send_realtime_input(
                        audio=types.Blob(data=raw, mime_type="audio/pcm;rate=24000")
                    )

                async def receive():
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

                try:
                    await run_duplex(receive(), microphone(send))
                finally:
                    await client.aio.aclose()

        try:
            try:
                await asyncio.wait_for(session.viewer_ready.wait(), 60)
            except asyncio.TimeoutError:
                await ws.send_json(
                    {
                        "type": "error",
                        "code": "playback_timeout",
                        "fatal": True,
                        "message": "Avatar playback did not connect. Enable playback in the browser and start a new session.",
                    }
                )
                return
            if provider == "openai":
                await run_openai_demo(session, microphone, feed, api_key)
            elif provider == "gemini":
                await gemini()
            elif provider == "elevenlabs":
                await run_elevenlabs_demo(session, microphone, feed, api_key, agent_id)
            else:
                raise ValueError("Unknown demo provider")
            await ws.send_json(
                {
                    "type": "error",
                    "code": "provider_closed",
                    "provider": provider,
                    "message": f"{PROVIDER_NAMES.get(provider, provider)} connection closed. Start a new session.",
                    "fatal": True,
                }
            )
        except ElevenLabsDemoError as exc:
            await ws.send_json(
                {"type": "error", "code": "configuration", "fatal": True, "message": str(exc)}
            )
        except ImportError:
            await ws.send_json(
                {
                    "type": "error",
                    "code": "configuration",
                    "fatal": True,
                    "message": "Install the optional SDK for the selected voice provider.",
                }
            )
        finally:
            output_task.cancel()
            await asyncio.gather(output_task, return_exceptions=True)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        failure = log_connection_failure(exc, provider, session, stage="demo")
        message = failure["message"] + " Start a new session to try again."
        try:
            await ws.send_json(
                {
                    "type": "error",
                    "code": "provider_connection",
                    "provider": provider,
                    "message": message,
                    "fatal": True,
                }
            )
        except Exception:
            pass
    finally:
        if authenticated:
            session.publisher_connected = False
            await remove(session.id)
