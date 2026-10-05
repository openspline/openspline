"""Safe diagnostics for optional voice providers; never log payloads or headers."""

import asyncio
import logging
import socket
import ssl
import traceback

from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidProxy, ProxyError

logger = logging.getLogger(__name__)
PROVIDER_NAMES = {"openai": "OpenAI Realtime", "gemini": "Gemini Live"}


class AvatarProcessingError(Exception):
    """An avatar operation failed, rather than the provider connection."""


async def avatar_operation(awaitable):
    try:
        return await awaitable
    except Exception as exc:
        raise AvatarProcessingError() from exc


def connection_failure(exc, provider):
    name = PROVIDER_NAMES.get(provider, provider)
    chain, current = [], exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    status = next(
        (
            value
            for error in chain
            for value in (
                getattr(error, "status_code", None),
                getattr(getattr(error, "response", None), "status_code", None),
            )
            if isinstance(value, int)
        ),
        None,
    )
    closed = next((error for error in chain if isinstance(error, ConnectionClosed)), None)
    received = closed.rcvd.code if closed and closed.rcvd else None
    sent = closed.sent.code if closed and closed.sent else None
    # Arbitrary close reasons may contain credentials or request content. Only
    # recognize the library's fixed heartbeat reason; report numeric codes otherwise.
    heartbeat = bool(closed and closed.sent and closed.sent.reason == "keepalive ping timeout")
    retryable = False
    if any(isinstance(error, AvatarProcessingError) for error in chain):
        message = "Avatar audio processing failed. Check the server's inference logs."
    elif any(isinstance(error, (InvalidProxy, ProxyError)) for error in chain):
        message = f"{name} could not connect through the server's proxy. Check its proxy settings."
    elif status in {401, 403}:
        message = f"{name} access was denied (HTTP {status}). Check the provider API key and model access on the server."
    elif status == 429:
        message = f"{name} reached a rate or quota limit (HTTP 429). Check the provider account and retry later."
    elif status in {400, 404}:
        message = f"{name} rejected the connection (HTTP {status}). Check the configured model and API base URL."
    elif status is not None and status >= 500:
        message = f"{name} is temporarily unavailable (HTTP {status})."
        retryable = True
    elif closed:
        retryable = received in {None, 1001, 1006, 1011, 1012, 1013}
        if heartbeat:
            message = (
                f"{name} heartbeat timed out. The connection stopped receiving timely replies."
            )
        elif received == 1011:
            message = f"{name} closed the connection with a server error (WebSocket 1011)."
        elif received == 1008:
            message = f"{name} rejected the session (WebSocket 1008). Check the provider API key, model access, and session limits."
        elif received is not None:
            message = f"{name} closed the connection (WebSocket {received})."
        else:
            message = f"{name} connection was lost without a close frame (WebSocket 1006). Check the server's network and proxy."
    elif any(isinstance(error, ssl.SSLError) for error in chain):
        message = f"{name} TLS connection failed. Check the server's certificates and HTTPS proxy."
    elif any(isinstance(error, socket.gaierror) for error in chain):
        message = f"{name} hostname lookup failed. Check the server's DNS and provider base URL."
        retryable = True
    elif any(isinstance(error, (TimeoutError, asyncio.TimeoutError)) for error in chain):
        message = f"{name} connection timed out. Check the server's network and proxy."
        retryable = True
    elif any(isinstance(error, ConnectionError) for error in chain):
        message = f"{name} could not maintain its network connection. Check the server's network and proxy."
        retryable = True
    elif any(isinstance(error, InvalidHandshake) for error in chain):
        message = f"{name} WebSocket handshake failed. Check the API base URL and proxy's WebSocket support."
    else:
        message = f"{name} failed ({type(exc).__name__}). Check the matching voice demo error in the server logs."
    return {
        "message": message,
        "retryable": retryable,
        "status": status,
        "received_close_code": received,
        "sent_close_code": sent,
        "heartbeat_timeout": heartbeat,
        "exception": type(exc).__name__,
    }


def log_connection_failure(exc, provider, session, *, stage, attempt=None):
    failure = connection_failure(exc, provider)
    # Frame locations identify application bugs without printing exception text,
    # source lines, locals, provider payloads, URLs, or authorization headers.
    locations = []
    error, seen = exc, set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        locations.extend(
            f"{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(error.__traceback__)
        )
        error = error.__cause__
    logger.warning(
        "Voice demo connection failed: provider=%s avatar_session=%s stage=%s attempt=%s "
        "exception=%s status=%s received_close=%s sent_close=%s heartbeat_timeout=%s frames=%s",
        provider,
        getattr(session, "id", None),
        stage,
        attempt,
        failure["exception"],
        failure["status"],
        failure["received_close_code"],
        failure["sent_close_code"],
        failure["heartbeat_timeout"],
        ">".join(locations),
    )
    return failure
