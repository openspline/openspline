import asyncio
import base64
import os
import sys
from http import HTTPStatus
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocketDisconnect
from openspline_server.demo import OpenAIDemoEvents, run_demo, run_duplex
from openspline_server.provider_errors import connection_failure, log_connection_failure
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response


@pytest.fixture(autouse=True)
def demo_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_AGENT_ID", raising=False)


def server_error():
    return NS(
        type="error",
        event_id="evt_test",
        error=NS(
            type="server_error",
            code="server_error",
            message="The server had an error while processing your request. Please retry.",
        ),
    )


def audio():
    return NS(
        type="response.output_audio.delta",
        item_id="item_test",
        content_index=0,
        delta=base64.b64encode(bytes(4800)).decode(),
    )


def make_session():
    events = asyncio.Queue()
    ready = asyncio.Event()
    ready.set()
    session = NS(
        id="avatar_test",
        publisher_token="publisher_test",
        publisher_connected=False,
        events=events,
        viewer_ready=ready,
        epoch=0,
        timeline=NS(played=0),
        end_turn=AsyncMock(),
        interrupt=AsyncMock(),
        push=AsyncMock(),
        touch=lambda: None,
    )
    session.emit = events.put_nowait
    return session


def test_demo_provider_status_never_returns_saved_keys(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from openspline_server.app import create_app
    from openspline_server.config import Settings

    monkeypatch.setenv("OPENAI_API_KEY", "private-saved-key")
    monkeypatch.setenv("GEMINI_API_KEY", "  ")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "private-eleven-key")
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "agent_private")
    with TestClient(create_app(Settings(backend="test", runtime_dir=str(tmp_path)))) as client:
        response = client.get("/v1/demo/providers")
        assert response.json() == {
            "openai": {"configured": True},
            "gemini": {"configured": False},
            "elevenlabs": {"configured": True, "agent_configured": True},
        }
        assert response.headers["cache-control"] == "no-store"
        assert "private-saved-key" not in response.text
        assert "private-eleven-key" not in response.text
        assert "agent_private" not in response.text


@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.parametrize(
    "fallback,supplied,expected",
    [
        (None, "entered-key", "entered-key"),
        ("saved-key", "  entered-key  ", "entered-key"),
        ("saved-key", None, "saved-key"),
        ("saved-key", "  ", "saved-key"),
    ],
)
async def test_demo_passes_session_key_to_selected_sdk_without_changing_environment(
    monkeypatch, caplog, provider, fallback, supplied, expected
):
    variable = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[provider]
    if fallback is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, fallback)
    received = []

    def client(*, api_key):
        received.append(api_key)
        raise WebSocketDisconnect()

    monkeypatch.setitem(sys.modules, "openai", NS(AsyncOpenAI=client))
    genai = NS(Client=client, types=NS())
    monkeypatch.setitem(sys.modules, "google", NS(genai=genai))
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    session, remove = make_session(), AsyncMock()
    auth = {"token": session.publisher_token}
    if supplied is not None:
        auth["api_key"] = supplied
    socket = NS(
        accept=AsyncMock(), receive_json=AsyncMock(return_value=auth), send_json=AsyncMock()
    )
    await run_demo(socket, session, provider, remove)
    assert received == [expected]
    assert os.getenv(variable) == fallback
    assert expected not in str(socket.send_json.call_args_list) + caplog.text
    remove.assert_awaited_once_with(session.id)
    assert not session.publisher_connected


@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.parametrize("supplied", [None, "  ", 123, {"key": "private"}, "x" * 4097])
async def test_demo_rejects_missing_or_invalid_keys_before_waiting_for_playback(
    monkeypatch, provider, supplied
):
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("GEMINI_API_KEY")
    session, remove = make_session(), AsyncMock()
    session.viewer_ready.clear()
    socket = NS(
        accept=AsyncMock(),
        receive_json=AsyncMock(
            return_value={"token": session.publisher_token, "api_key": supplied}
        ),
        send_json=AsyncMock(),
    )
    await asyncio.wait_for(run_demo(socket, session, provider, remove), 1)
    error = socket.send_json.call_args.args[0]
    assert error["fatal"] and error["code"] == "configuration"
    assert "API key" in error["message"]
    assert "private" not in error["message"]
    remove.assert_awaited_once_with(session.id)


async def test_demo_authenticates_before_accepting_provider_credentials():
    session, remove = make_session(), AsyncMock()
    socket = NS(
        accept=AsyncMock(),
        receive_json=AsyncMock(return_value={"token": "wrong", "api_key": 123}),
        send_json=AsyncMock(),
        close=AsyncMock(),
    )
    await run_demo(socket, session, "openai", remove)
    socket.close.assert_awaited_once_with(4401)
    socket.send_json.assert_not_called()
    remove.assert_not_called()


async def test_openai_error_keeps_playback_alive_and_later_audio_recovers(caplog):
    session, conn, feed = make_session(), NS(send=AsyncMock()), AsyncMock()
    handler = OpenAIDemoEvents(session, conn, feed)
    await handler.handle(NS(type="session.created", session=NS(id="sess_test")))
    await handler.handle(server_error())
    event = session.events.get_nowait()
    assert event["type"] == "warning" and event["provider"] == "openai"
    assert "try speaking again" in event["message"]
    assert "provider_session=sess_test" in caplog.text and "event_id=evt_test" in caplog.text
    session.interrupt.assert_not_called()
    await handler.handle(audio())
    feed.assert_awaited_once_with(bytes(4800))
    assert session.events.get_nowait()["type"] == "provider_status"
    await handler.handle(NS(type="response.output_audio.done"))
    session.end_turn.assert_awaited_once()
    conn.send.assert_not_called()  # Do not replay a failed request or duplicate its speech.


@pytest.mark.parametrize("audio_done", [False, True])
async def test_failed_response_flushes_partial_audio_once_and_reports_warning(audio_done):
    session = make_session()
    handler = OpenAIDemoEvents(session, NS(send=AsyncMock()), AsyncMock())
    await handler.handle(audio())
    if audio_done:
        await handler.handle(NS(type="response.output_audio.done"))
    await handler.handle(
        NS(
            type="response.done",
            event_id="evt_done",
            response=NS(
                id="resp_failed",
                status="failed",
                status_details=NS(error=NS(type="server_error", code="server_error")),
            ),
        )
    )
    session.end_turn.assert_awaited_once()
    assert session.events.get_nowait()["type"] == "warning"
    await handler.handle(audio())
    assert session.events.get_nowait()["type"] == "provider_status"


async def test_duplex_cancels_idle_provider_on_microphone_disconnect():
    cancelled = asyncio.Event()

    async def provider():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def microphone():
        raise WebSocketDisconnect()

    with pytest.raises(WebSocketDisconnect):
        await asyncio.wait_for(run_duplex(provider(), microphone()), 1)
    assert cancelled.is_set()


async def test_demo_provider_warning_is_delivered_without_teardown_then_disconnect_cleans_up(
    monkeypatch,
):
    source, sent, inbound = asyncio.Queue(), asyncio.Queue(), asyncio.Queue()
    session = make_session()

    class Connection:
        session = NS(update=AsyncMock())
        input_audio_buffer = NS(append=AsyncMock())
        send = AsyncMock()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def __aiter__(self):
            while True:
                yield await source.get()

    connection = Connection()
    monkeypatch.setitem(
        sys.modules,
        "openai",
        NS(
            AsyncOpenAI=lambda **kwargs: NS(
                realtime=NS(connect=lambda **kwargs: connection), close=AsyncMock()
            )
        ),
    )

    async def receive():
        value = await inbound.get()
        if value is None:
            raise WebSocketDisconnect()
        return value

    socket = NS(
        accept=AsyncMock(),
        receive_json=AsyncMock(return_value={"token": session.publisher_token}),
        receive_bytes=receive,
        send_json=sent.put,
        close=AsyncMock(),
    )
    remove = AsyncMock()
    task = asyncio.create_task(run_demo(socket, session, "openai", remove))
    try:
        assert (await asyncio.wait_for(sent.get(), 1))["type"] == "ready"
        await source.put(server_error())
        warning = await asyncio.wait_for(sent.get(), 1)
        assert warning["type"] == "warning"
        remove.assert_not_called()
        assert session.publisher_connected
        await source.put(audio())
        assert (await asyncio.wait_for(sent.get(), 1))["type"] == "provider_status"
        session.push.assert_awaited_once()
        await inbound.put(None)
        await asyncio.wait_for(task, 1)
        remove.assert_awaited_once_with(session.id)
        assert not session.publisher_connected
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_error_handling_with_installed_openai_event_types():
    pytest.importorskip("openai")
    from openai.types.realtime import RealtimeErrorEvent, ResponseDoneEvent, SessionCreatedEvent

    handler = OpenAIDemoEvents(make_session(), NS(send=AsyncMock()), AsyncMock())
    await handler.handle(
        SessionCreatedEvent.model_validate(
            {
                "type": "session.created",
                "event_id": "evt_created",
                "session": {"type": "realtime", "id": "sess_test"},
            }
        )
    )
    await handler.handle(
        RealtimeErrorEvent.model_validate(
            {
                "type": "error",
                "event_id": "evt_error",
                "error": {"type": "server_error", "code": "server_error", "message": "Retry"},
            }
        )
    )
    await handler.handle(
        ResponseDoneEvent.model_validate(
            {
                "type": "response.done",
                "event_id": "evt_done",
                "response": {
                    "id": "resp_test",
                    "status": "failed",
                    "status_details": {"type": "failed", "error": {"type": "server_error"}},
                },
            }
        )
    )
    events = [handler.session.events.get_nowait(), handler.session.events.get_nowait()]
    assert all(
        event["type"] == "warning" and event["provider_session_id"] == "sess_test"
        for event in events
    )


@pytest.mark.parametrize(
    "status,text",
    [(503, "temporarily unavailable"), (401, "access was denied"), (429, "rate or quota limit")],
)
async def test_provider_connection_failure_is_actionable_and_releases_session(
    monkeypatch, caplog, status, text
):
    session, sent = make_session(), []

    class ProviderFailure(Exception):
        status_code = status

    def fail(**kwargs):
        raise ProviderFailure("private provider diagnostics")

    monkeypatch.setitem(
        sys.modules,
        "openai",
        NS(AsyncOpenAI=lambda **kwargs: NS(realtime=NS(connect=fail), close=AsyncMock())),
    )
    socket = NS(
        accept=AsyncMock(),
        receive_json=AsyncMock(return_value={"token": session.publisher_token}),
        receive_bytes=asyncio.Event().wait,
        send_json=AsyncMock(side_effect=sent.append),
        close=AsyncMock(),
    )
    remove = AsyncMock()
    await run_demo(socket, session, "openai", remove)
    assert sent[-1]["type"] == "error" and sent[-1]["fatal"]
    assert text in sent[-1]["message"]
    assert "private provider diagnostics" not in str(sent) + caplog.text
    remove.assert_awaited_once_with(session.id)
    assert not session.publisher_connected


@pytest.mark.parametrize("status", [401, 403, 404, 429, 503])
def test_real_websocket_handshake_errors_expose_http_status(status):
    failure = connection_failure(
        InvalidStatus(Response(status, "private text", Headers())), "openai"
    )
    assert failure["status"] == status
    assert str(status) in failure["message"]
    assert failure["retryable"] is (status == 503)
    assert "private text" not in str(failure)


@pytest.mark.parametrize(
    "received,sent,heartbeat,retryable",
    [
        (1011, 1011, False, True),
        (1008, 1008, False, False),
        (None, 1011, True, True),
        (None, None, False, True),
    ],
)
def test_close_diagnostics_include_codes_without_private_reason(
    caplog, received, sent, heartbeat, retryable
):
    exc = ConnectionClosedError(
        Close(received, "private provider payload") if received else None,
        Close(sent, "keepalive ping timeout" if heartbeat else "private provider payload")
        if sent
        else None,
        True if received and sent else None,
    )
    details = log_connection_failure(exc, "openai", NS(id="test"), stage="stream")
    assert details["received_close_code"] == received
    assert details["sent_close_code"] == sent
    assert details["heartbeat_timeout"] == heartbeat
    assert details["retryable"] == retryable
    assert "private provider payload" not in str(details) + caplog.text
    assert "received_close=" in caplog.text


async def test_avatar_failure_is_not_reported_or_retried_as_provider_disconnect():
    session = make_session()
    feed = AsyncMock(side_effect=RuntimeError("private inference details"))
    handler = OpenAIDemoEvents(session, NS(send=AsyncMock()), feed)
    with pytest.raises(Exception) as caught:
        await handler.handle(audio())
    details = connection_failure(caught.value, "openai")
    assert "Avatar audio processing failed" in details["message"]
    assert not details["retryable"]


async def test_real_openai_sdk_reconnects_after_close_on_speech_without_replaying_audio(
    monkeypatch,
):
    """Real SDK + local WebSocket server, including close on the first mic packet."""
    pytest.importorskip("openai")
    import json

    from websockets.asyncio.server import serve

    connections, received = [], []
    source, sent = asyncio.Queue(), asyncio.Queue()
    session, remove = make_session(), AsyncMock()
    second_connected, second_audio = asyncio.Event(), asyncio.Event()

    async def provider(conn):
        index = len(connections)
        connections.append(conn)
        config = json.loads(await conn.recv())
        assert config["type"] == "session.update"
        assert config["session"]["audio"]["output"]["voice"] == "marin"
        if index:
            second_connected.set()
        async for raw in conn:
            event = json.loads(raw)
            if event["type"] != "input_audio_buffer.append":
                continue
            received.append((index, base64.b64decode(event["audio"])))
            if not index:
                await conn.close(1011, "private provider details")
                return
            second_audio.set()
            await conn.send(
                json.dumps(
                    {
                        "type": "response.output_audio.delta",
                        "event_id": "evt_audio",
                        "response_id": "resp_test",
                        "output_index": 0,
                        "item_id": "item_test",
                        "content_index": 0,
                        "delta": base64.b64encode(bytes(4800)).decode(),
                    }
                )
            )

    async def microphone():
        value = await source.get()
        if value is None:
            raise WebSocketDisconnect()
        return value

    socket = NS(
        accept=AsyncMock(),
        close=AsyncMock(),
        send_json=sent.put,
        receive_json=AsyncMock(return_value={"token": session.publisher_token}),
        receive_bytes=microphone,
    )
    async with serve(provider, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("OPENAI_API_KEY", "test-only")
        monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
        task = asyncio.create_task(run_demo(socket, session, "openai", remove))
        try:
            assert (await asyncio.wait_for(sent.get(), 2))["type"] == "ready"
            # Ready must mean the provider can receive the user's first words.
            await source.put(b"first turn")
            warning = await asyncio.wait_for(sent.get(), 3)
            assert warning["code"] == "provider_reconnecting"
            assert "1011" in warning["message"]
            assert "private provider details" not in str(warning)
            assert session.publisher_connected
            remove.assert_not_called()
            session.interrupt.assert_awaited_once()
            # Mic packets sent during reconnection must be drained and discarded.
            await source.put(b"stale microphone audio")
            await asyncio.wait_for(second_connected.wait(), 3)
            status = await asyncio.wait_for(sent.get(), 2)
            assert status["type"] == "provider_status"
            assert "new conversation" in status["message"]
            await source.put(b"repeated turn")
            await asyncio.wait_for(second_audio.wait(), 2)
            for _ in range(100):
                if session.push.await_count:
                    break
                await asyncio.sleep(0.01)
            session.push.assert_awaited_once_with(bytes(4800), {"sample_rate": 24000})
            assert received == [(0, b"first turn"), (1, b"repeated turn")]
            await source.put(None)
            await asyncio.wait_for(task, 2)
            remove.assert_awaited_once_with(session.id)
            assert not session.publisher_connected
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("status,attempts", [(401, 1), (503, 3)])
@pytest.mark.parametrize("api_key", [None, "entered-test-only"])
async def test_real_openai_handshake_retry_limits_and_cleanup(
    monkeypatch, status, attempts, api_key
):
    pytest.importorskip("openai")
    from websockets.asyncio.server import serve

    accepted = []

    async def reject(conn, request):
        assert request.headers["Authorization"] == f"Bearer {api_key or 'test-only'}"
        accepted.append(True)
        return conn.respond(HTTPStatus(status), "private response body")

    async def provider(conn):
        pytest.fail("Handshake should be rejected")

    session, sent, remove = make_session(), [], AsyncMock()
    socket = NS(
        accept=AsyncMock(),
        close=AsyncMock(),
        send_json=AsyncMock(side_effect=sent.append),
        receive_json=AsyncMock(return_value={"token": session.publisher_token, "api_key": api_key}),
        receive_bytes=asyncio.Event().wait,
    )
    async with serve(provider, "127.0.0.1", 0, process_request=reject) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("OPENAI_API_KEY", "test-only")
        monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
        await asyncio.wait_for(run_demo(socket, session, "openai", remove), 5)
    assert len(accepted) == attempts
    assert sent[-1]["fatal"] and str(status) in sent[-1]["message"]
    assert "private response body" not in str(sent)
    remove.assert_awaited_once_with(session.id)


async def test_disconnect_during_reconnect_cancels_retry_and_releases_session(monkeypatch):
    import openspline_server.demo as demo

    attempts, sent = [], asyncio.Queue()
    source = asyncio.Queue()
    session, remove, closed = make_session(), AsyncMock(), AsyncMock()

    def connect(**kwargs):
        attempts.append(True)
        raise ConnectionClosedError(None, None)

    monkeypatch.setitem(
        sys.modules,
        "openai",
        NS(
            AsyncOpenAI=lambda **kwargs: NS(
                realtime=NS(connect=connect),
                close=closed,
            )
        ),
    )
    original_sleep = asyncio.sleep
    retry_waiting = asyncio.Event()

    async def wait_in_retry(delay):
        retry_waiting.set()
        await asyncio.Event().wait()

    # Patch only the demo module's sleep, without changing asyncio globally.
    monkeypatch.setattr(demo, "asyncio", NS(**{**vars(asyncio), "sleep": wait_in_retry}))

    async def microphone():
        await source.get()
        raise WebSocketDisconnect()

    socket = NS(
        accept=AsyncMock(),
        close=AsyncMock(),
        send_json=sent.put,
        receive_json=AsyncMock(return_value={"token": session.publisher_token}),
        receive_bytes=microphone,
    )
    task = asyncio.create_task(run_demo(socket, session, "openai", remove))
    try:
        await asyncio.wait_for(retry_waiting.wait(), 1)
        assert session.publisher_connected
        await source.put(None)
        await asyncio.wait_for(task, 1)
        await original_sleep(0)
        assert len(attempts) == 1
        closed.assert_awaited_once()
        remove.assert_awaited_once_with(session.id)
        assert not session.publisher_connected
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_microphone_send_failure_ends_blocked_receive_and_releases_session(monkeypatch):
    import openspline_server.demo as demo

    session, sent, remove = make_session(), [], AsyncMock()
    connected, cancelled = asyncio.Event(), asyncio.Event()

    class Connection:
        session = NS(update=AsyncMock())
        input_audio_buffer = NS(append=AsyncMock(side_effect=ConnectionClosedError(None, None)))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def __aiter__(self):
            connected.set()
            try:
                await asyncio.Event().wait()
                yield
            finally:
                cancelled.set()

    monkeypatch.setattr(demo, "OPENAI_CONNECTION_ATTEMPTS", 1)
    monkeypatch.setitem(
        sys.modules,
        "openai",
        NS(
            AsyncOpenAI=lambda **kwargs: NS(
                realtime=NS(connect=lambda **kwargs: Connection()),
                close=AsyncMock(),
            )
        ),
    )
    received = False

    async def microphone():
        nonlocal received
        await connected.wait()
        if received:
            await asyncio.Event().wait()
        received = True
        return bytes(4800)

    socket = NS(
        accept=AsyncMock(),
        close=AsyncMock(),
        send_json=AsyncMock(side_effect=sent.append),
        receive_json=AsyncMock(return_value={"token": session.publisher_token}),
        receive_bytes=microphone,
    )
    await asyncio.wait_for(run_demo(socket, session, "openai", remove), 1)
    assert cancelled.is_set()
    assert sent[-1]["fatal"] and "1006" in sent[-1]["message"]
    remove.assert_awaited_once_with(session.id)
