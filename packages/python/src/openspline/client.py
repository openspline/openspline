"""Lightweight async client. No CUDA, NumPy, or framework imports."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import json
import uuid
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import httpx
from websockets.legacy.client import connect

from .config import OpensplineConfig, Quality
from .errors import ConnectionError, error_from


class Openspline:
    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        *,
        quality: Quality | None = None,
        viewer_timeout: float | None = None,
        config: OpensplineConfig | Mapping | None = None,
    ):
        if config is None:
            values = {}
        elif isinstance(config, OpensplineConfig):
            values = {field.name: getattr(config, field.name) for field in fields(config)}
        elif isinstance(config, Mapping):
            values = dict(config)
        else:
            raise TypeError("config must be an OpensplineConfig or mapping")
        overrides = {
            key: value
            for key, value in {
                "url": url,
                "api_key": api_key,
                "timeout": timeout,
                "quality": quality,
                "viewer_timeout": viewer_timeout,
            }.items()
            if value is not None
        }
        self.config = OpensplineConfig(**(values | overrides))
        self.url = self.config.url
        self.api_key = self.config.api_key
        self.timeout = self.config.timeout

    def avatar(self, portrait, quality: Quality | None = None):
        quality = self.config.quality if quality is None else quality
        if quality not in {"low", "high"}:
            raise ValueError("quality must be low or high")
        return AvatarSession(self, portrait, quality)


class AvatarSession:
    def __init__(self, client, portrait, quality):
        self.client, self.portrait, self.quality = client, portrait, quality
        self.id = None
        self.viewer_url = None
        self.session = None
        self.ws = None
        self.http = None
        self.pending = {}
        self.callbacks = []
        self.epoch = 0
        self.viewer_ready = asyncio.Event()
        self.reader = None
        self.heartbeat = None
        self._dirty = False
        self._closed = False
        self._send_lock = asyncio.Lock()
        self._media = None

    async def __aenter__(self):
        self.http = httpx.AsyncClient(base_url=self.client.url, timeout=self.client.timeout)
        try:
            if isinstance(self.portrait, (str, Path)):
                raw = Path(self.portrait).read_bytes()
            else:
                raw = bytes(self.portrait)
            response = await self.http.post(
                "/v1/sessions",
                headers=(
                    {"Authorization": "Bearer " + self.client.api_key}
                    if self.client.api_key
                    else {}
                ),
                files={"portrait": ("portrait", raw)},
                data={"quality": self.quality},
            )
            self._check(response)
            body = response.json()
            self.id = body["id"]
            self.token = body["publisher_token"]
            self.session = {k: v for k, v in body.items() if k != "publisher_token"}
            self.viewer_url = body["viewer_url"]
            url = (
                self.client.url.replace("https://", "wss://").replace("http://", "ws://")
                + f"/v1/sessions/{self.id}/audio"
            )
            self.ws = await connect(url, max_size=1024 * 1024, open_timeout=30)
            await self.ws.send(json.dumps({"token": self.token}))
            self.reader = asyncio.create_task(self._receive())
            self.heartbeat = asyncio.create_task(self._heartbeat())
            return self
        except BaseException:
            await self.close()
            raise

    async def __aexit__(self, typ, exc, tb):
        try:
            if typ is None and self._dirty:
                await self.end_turn()
        finally:
            await self.close()

    @staticmethod
    def _check(response):
        if response.is_error:
            try:
                error = response.json().get("error", {})
                message = error.get("message", response.text)
                code = error.get("code", "request")
            except (ValueError, AttributeError):
                message = response.text
                code = "request"
            raise error_from(code, message)

    def on(self, callback):
        self.callbacks.append(callback)
        return lambda: self.callbacks.remove(callback)

    async def _receive(self):
        failure = ConnectionError("Session connection closed", "connection")
        try:
            async for raw in self.ws:
                event = json.loads(raw)
                self.epoch = max(self.epoch, event.get("epoch", self.epoch))
                if event.get("type") == "viewer_ready" or event.get("viewer_ready"):
                    self.viewer_ready.set()
                if event.get("type") == "viewer_disconnected":
                    self.viewer_ready.clear()
                id = event.get("id")
                future = self.pending.get(id)
                if event.get("type") == "error":
                    error = error_from(
                        event.get("code", "error"), event.get("message", "Session error")
                    )
                    if future and not future.done():
                        future.set_exception(error)
                    elif not id:
                        failure = error
                        for f in self.pending.values():
                            if not f.done():
                                f.set_exception(error)
                elif future and not future.done():
                    future.set_result(event)
                for cb in list(self.callbacks):
                    try:
                        value = cb(event)
                        if inspect.isawaitable(value):
                            asyncio.create_task(value)
                    except Exception:
                        import logging

                        logging.getLogger(__name__).exception("Session event callback failed")
        except Exception as exc:
            failure = ConnectionError(str(exc), "connection")
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(failure)

    async def _heartbeat(self):
        try:
            while True:
                await asyncio.sleep(15)
                await self._request("ping")
        except (asyncio.CancelledError, Exception):
            return

    async def _request(self, type, **data):
        if self._closed or not self.ws:
            raise ConnectionError("Session is not connected", "connection")
        id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[id] = future
        try:
            async with self._send_lock:
                await self.ws.send(json.dumps({"type": type, "id": id, **data}))
            return await asyncio.wait_for(future, self.client.timeout)
        finally:
            self.pending.pop(id, None)

    async def wait_for_viewer(self, timeout=None):
        await asyncio.wait_for(
            self.viewer_ready.wait(),
            self.client.config.viewer_timeout if timeout is None else timeout,
        )

    async def send_audio(self, data, *, sample_rate=24000, channels=1, encoding="pcm_s16le"):
        if (
            sample_rate not in {8000, 16000, 22050, 24000, 32000, 44100, 48000}
            or channels not in {1, 2}
            or encoding not in {"pcm_s16le", "pcm_f32le", "mp3"}
        ):
            raise ValueError("Invalid audio format")
        raw = bytes(data)
        size = (
            16384
            if encoding == "mp3"
            else sample_rate * channels * (4 if encoding == "pcm_f32le" else 2) // 10
        )
        epoch = self.epoch
        for i in range(0, len(raw), size):
            if epoch != self.epoch:
                return
            await self._request(
                "audio",
                epoch=epoch,
                data=base64.b64encode(raw[i : i + size]).decode(),
                format={"sample_rate": sample_rate, "channels": channels, "encoding": encoding},
            )
            self._dirty = True

    async def stream(self, source, **format):
        await self.wait_for_viewer()
        try:
            async for packet in source:
                await self.send_audio(packet, **format)
            await self.end_turn()
        except BaseException:
            with contextlib.suppress(Exception):
                await self.interrupt()
            raise

    async def end_turn(self, drain=True):
        result = await self._request("end_turn", drain=drain)
        self._dirty = False
        return result

    async def interrupt(self):
        # A separate request can overtake a backpressured audio WebSocket.
        response = await self.http.post(
            f"/v1/sessions/{self.id}/interrupt", headers=self._headers()
        )
        self._check(response)
        self.epoch = response.json()["epoch"]
        self._dirty = False

    def _headers(self):
        return {"Authorization": "Bearer " + self.token}

    async def viewer(self):
        response = await self.http.post(f"/v1/sessions/{self.id}/viewer", headers=self._headers())
        self._check(response)
        self.session = response.json()
        self.viewer_url = self.session["viewer_url"]
        return self.session

    async def play_file(self, path):
        await self.wait_for_viewer()
        with Path(path).open("rb") as f:
            response = await self.http.post(
                f"/v1/sessions/{self.id}/file",
                headers=self._headers(),
                files={"audio": (Path(path).name, f)},
            )
        self._check(response)
        self._dirty = True
        await self.end_turn()

    async def media(self):
        from .media import MediaReceiver

        if self._media is None:
            self._media = MediaReceiver(self)
            await self._media.start()
        return self._media

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self._media:
            await self._media.close()
        if self.heartbeat:
            self.heartbeat.cancel()
        # Explicit delete is idempotent and releases the worker even if WebSocket setup failed.
        if self.id and self.http:
            with contextlib.suppress(Exception):
                await self.http.delete(f"/v1/sessions/{self.id}", headers=self._headers())
        if self.ws:
            await self.ws.close()
        if self.reader:
            self.reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.reader
        if self.http:
            await self.http.aclose()
