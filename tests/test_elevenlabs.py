import asyncio
import base64
import json
import struct
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import WebSocketDisconnect
from openspline.integrations import ElevenLabsAgents
from openspline_server.demo import demo_agent_id, demo_api_key, run_demo
from openspline_server.elevenlabs_demo import MicrophoneAudio
from test_demo_provider import make_session
from test_integrations import Avatar
from websockets.asyncio.server import serve


def audio(raw=b"ab", event_id=1, final=False):
    return {
        "type": "audio",
        "audio_event": {
            "audio_base_64": base64.b64encode(raw).decode(),
            "event_id": event_id,
            "is_final": final,
        },
    }


def metadata(output="pcm_16000", input="pcm_16000"):
    return {
        "type": "conversation_initiation_metadata",
        "conversation_initiation_metadata_event": {
            "conversation_id": "conv_test",
            "agent_output_audio_format": output,
            "user_input_audio_format": input,
        },
    }


def complete(event_id=1):
    return {
        "type": "agent_response_complete",
        "agent_response_complete_event": {"event_id": event_id},
    }


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 44100, 48000])
async def test_adapter_uses_metadata_and_flushes_audio_once(rate):
    avatar = Avatar()
    adapter = ElevenLabsAgents(avatar)
    await adapter.handle(metadata(f"pcm_{rate}"))
    await adapter.handle(audio())
    await adapter.handle(
        {
            "type": "agent_response",
            "agent_response_event": {"agent_response": "text is not audio completion"},
        }
    )
    assert avatar.calls == [("audio", b"ab", {"sample_rate": rate})]
    await adapter.handle(audio(b"cd", final=True))
    await adapter.handle(complete())
    assert avatar.calls[-2:] == [("audio", b"cd", {"sample_rate": rate}), ("end",)]
    adapter.avatar.end_turn = AsyncMock()
    await adapter.handle(audio(event_id=2))
    await adapter.handle(complete(2))
    adapter.avatar.end_turn.assert_awaited_once_with(drain=False)


async def test_adapter_decodes_mulaw_and_drops_interrupted_audio():
    avatar = Avatar()
    adapter = ElevenLabsAgents(avatar, audio_format="ulaw_8000")
    await adapter.handle(audio(bytes([0, 128, 255, 127])))
    assert avatar.calls[0] == (
        "audio",
        struct.pack("<4h", -32124, 32124, 0, 0),
        {"sample_rate": 8000},
    )
    await adapter.handle({"type": "interruption", "interruption_event": {"event_id": 3}})
    await adapter.handle(audio(event_id=3, final=True))
    await adapter.handle(complete(3))
    assert len(avatar.calls) == 2 and avatar.calls[-1] == ("interrupt",)
    await adapter.handle(audio(event_id=4, final=True))
    assert avatar.calls[-1] == ("end",)


async def test_adapter_wrap_preserves_tool_transcript_and_ping_events():
    events = [{"type": kind} for kind in ("client_tool_call", "user_transcript", "ping")]

    async def source():
        for event in events:
            yield event

    avatar = Avatar()
    assert [event async for event in ElevenLabsAgents(avatar).wrap(source())] == events
    assert not avatar.calls
    with pytest.raises(ValueError, match="audio format"):
        ElevenLabsAgents(avatar, audio_format="mp3_44100_128")


@pytest.mark.parametrize(
    "fmt,rate,width",
    [
        ("pcm_16000", 16000, 2),
        ("pcm_24000", 24000, 2),
        ("pcm_44100", 44100, 2),
        ("ulaw_8000", 8000, 1),
    ],
)
def test_microphone_conversion_preserves_rate_and_state(fmt, rate, width):
    converter = MicrophoneAudio(fmt)
    packet = bytes(4800)  # 100ms at the browser's 24kHz rate
    parts = [converter.push(packet) for _ in range(10)]
    samples = len(b"".join(parts)) // width
    assert rate - 100 < samples <= rate
    assert len(parts[-1]) == rate // 10 * width
    assert set(b"".join(parts)) == ({255} if width == 1 else {0})


def test_public_private_credentials_and_agent_validation(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_AGENT_ID", raising=False)
    assert demo_api_key("elevenlabs", None) == ""
    for value in (123, {}, "x" * 4097):
        with pytest.raises(ValueError, match="API key"):
            demo_api_key("elevenlabs", value)
    for value in (None, " ", 123, {}, "https://example.com", "x" * 257):
        with pytest.raises(ValueError, match="agent ID"):
            demo_agent_id(value)
    monkeypatch.setenv("ELEVENLABS_API_KEY", " saved-key ")
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", " agent_saved ")
    assert demo_api_key("elevenlabs", None) == "saved-key"
    assert demo_api_key("elevenlabs", " entered-key ") == "entered-key"
    assert demo_agent_id(None) == "agent_saved"
    assert demo_agent_id(" agent_entered ") == "agent_entered"


async def test_demo_rejects_missing_agent_before_waiting_for_playback(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_AGENT_ID", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    session, remove = make_session(), AsyncMock()
    session.viewer_ready.clear()
    socket = NS(
        accept=AsyncMock(),
        receive_json=AsyncMock(return_value={"token": session.publisher_token}),
        send_json=AsyncMock(),
    )
    await asyncio.wait_for(run_demo(socket, session, "elevenlabs", remove), 1)
    assert socket.send_json.call_args.args[0]["code"] == "configuration"
    assert "agent ID" in socket.send_json.call_args.args[0]["message"]
    remove.assert_awaited_once_with(session.id)


async def test_private_auth_failure_is_safe_and_releases_session(monkeypatch, caplog):
    def reject(request):
        return httpx.Response(403, json={"detail": "private-key-and-provider-payload"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(reject), **kwargs),
    )
    session, remove = make_session(), AsyncMock()
    socket = NS(
        accept=AsyncMock(),
        receive_json=AsyncMock(
            return_value={
                "token": session.publisher_token,
                "agent_id": "agent_test",
                "api_key": "private-test-key",
            }
        ),
        send_json=AsyncMock(),
    )
    await asyncio.wait_for(run_demo(socket, session, "elevenlabs", remove), 2)
    error = socket.send_json.call_args.args[0]
    assert error["fatal"] and error["code"] == "provider_connection"
    assert "ElevenLabs" in error["message"] and "403" in error["message"]
    assert "private-test-key" not in str(error) + caplog.text
    assert "private-key-and-provider-payload" not in str(error) + caplog.text
    assert not session.publisher_connected
    remove.assert_awaited_once_with(session.id)


@pytest.mark.parametrize(
    "event",
    [
        metadata("mp3_44100_128"),
        {"type": "client_error", "error_event": {"message": "private-provider-payload"}},
    ],
)
async def test_provider_configuration_error_is_actionable_and_closes_connection(monkeypatch, event):
    from openspline_server import elevenlabs_demo

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    async def provider(conn):
        await conn.recv()
        await conn.send(json.dumps(event))
        await conn.wait_closed()

    async with serve(provider, "127.0.0.1", 0) as server:
        monkeypatch.setattr(
            elevenlabs_demo, "WS_URL", f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        )
        session, remove = make_session(), AsyncMock()
        socket = NS(
            accept=AsyncMock(),
            receive_json=AsyncMock(
                return_value={"token": session.publisher_token, "agent_id": "agent_test"}
            ),
            send_json=AsyncMock(),
        )
        await asyncio.wait_for(run_demo(socket, session, "elevenlabs", remove), 3)
        error = socket.send_json.call_args.args[0]
        assert error["fatal"] and error["code"] == "configuration"
        assert "ElevenLabs" in error["message"]
        assert "private-provider-payload" not in str(error)
        remove.assert_awaited_once_with(session.id)


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("output_format", ["pcm_16000", "ulaw_8000"])
async def test_demo_duplex_audio_auth_ping_interrupt_and_cleanup(
    monkeypatch, caplog, private, output_format
):
    from openspline_server import elevenlabs_demo

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_AGENT_ID", raising=False)
    finished = asyncio.Event()
    received = []

    async def provider(conn):
        received.append(conn.request.path)
        assert json.loads(await conn.recv()) == {"type": "conversation_initiation_client_data"}
        await conn.send(json.dumps({"type": "ping", "ping_event": {"event_id": 0}}))
        assert json.loads(await conn.recv()) == {"type": "pong", "event_id": 0}
        await conn.send(json.dumps(metadata(output_format, "pcm_24000")))
        mic = json.loads(await conn.recv())
        assert base64.b64decode(mic["user_audio_chunk"]) == bytes(4800)
        for event in [
            audio(),
            {"type": "interruption", "interruption_event": {"event_id": 1}},
            audio(event_id=1, final=True),
            complete(1),
            audio(b"cd", event_id=2, final=True),
            complete(2),
            {
                "type": "client_tool_call",
                "client_tool_call": {"tool_call_id": "tool_test", "expects_response": True},
            },
            {"type": "ping", "ping_event": {"event_id": 9}},
        ]:
            await conn.send(json.dumps(event))
        tool = json.loads(await conn.recv())
        assert (
            tool["type"] == "client_tool_result"
            and tool["is_error"]
            and tool["tool_call_id"] == "tool_test"
        )
        assert json.loads(await conn.recv()) == {"type": "pong", "event_id": 9}
        finished.set()
        await conn.wait_closed()

    async with serve(provider, "127.0.0.1", 0) as server:
        base = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        monkeypatch.setattr(elevenlabs_demo, "WS_URL", base)
        auth_requests = []

        def sign(request):
            auth_requests.append(request)
            assert request.headers["xi-api-key"] == "private-test-key"
            assert request.url.params["agent_id"] == "agent_test"
            return httpx.Response(
                200, json={"signed_url": base + "/?conversation_signature=private-signature"}
            )

        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kwargs: real_client(transport=httpx.MockTransport(sign), **kwargs),
        )
        session, remove = make_session(), AsyncMock()
        browser_events, mic_queue = asyncio.Queue(), asyncio.Queue()

        async def mic():
            packet = await mic_queue.get()
            if isinstance(packet, Exception):
                raise packet
            return packet

        auth = {"token": session.publisher_token, "agent_id": "agent_test"}
        if private:
            auth["api_key"] = "private-test-key"
        socket = NS(
            accept=AsyncMock(),
            receive_json=AsyncMock(return_value=auth),
            send_json=AsyncMock(side_effect=browser_events.put_nowait),
            receive_bytes=mic,
        )
        task = asyncio.create_task(run_demo(socket, session, "elevenlabs", remove))
        try:
            assert (await asyncio.wait_for(browser_events.get(), 3))["type"] == "ready"
            await mic_queue.put(bytes(4800))
            await asyncio.wait_for(finished.wait(), 3)
            assert session.push.await_count == 2  # stale audio after interruption was dropped
            session.interrupt.assert_awaited_once()
            session.end_turn.assert_awaited_once()
            if output_format == "pcm_16000":
                assert session.push.call_args.args == (b"cd", {"sample_rate": 16000})
            else:
                raw, fmt = session.push.call_args.args
                assert len(raw) == 4 and fmt == {"sample_rate": 8000}
            await mic_queue.put(WebSocketDisconnect())
            await asyncio.wait_for(task, 3)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert len(auth_requests) == int(private)
        assert ("conversation_signature=" in received[0]) == private
        if not private:
            assert "agent_id=agent_test" in received[0]
        remove.assert_awaited_once_with(session.id)
        assert not session.publisher_connected
        assert "private-test-key" not in str(socket.send_json.call_args_list) + caplog.text
        assert "private-signature" not in str(socket.send_json.call_args_list) + caplog.text
