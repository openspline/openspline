import asyncio
import base64
import sys
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocketDisconnect
from openspline_server.demo import OpenAIDemoEvents, run_demo, run_duplex


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
        NS(AsyncOpenAI=lambda: NS(realtime=NS(connect=lambda **kwargs: connection))),
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
        sys.modules, "openai", NS(AsyncOpenAI=lambda: NS(realtime=NS(connect=fail)))
    )
    socket = NS(
        accept=AsyncMock(),
        receive_json=AsyncMock(return_value={"token": session.publisher_token}),
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
