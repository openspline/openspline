from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import secrets
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

import av
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .config import Settings
from .media import AudioTrack, VideoTrack
from .session import Session
from .worker import CapacityError, WorkerPool

STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None):
    settings = settings or Settings()
    pool = WorkerPool(settings)
    sessions = {}

    async def remove(id):
        s = sessions.pop(id, None)
        if s:
            async with s.close_lock:
                await s.close()
                await pool.release(s.worker, s.id)
                shutil.rmtree(Path(settings.runtime_dir) / "sessions" / s.id, ignore_errors=True)

    async def reaper():
        while True:
            await asyncio.sleep(5)
            await pool.recover_idle()
            for id, s in list(sessions.items()):
                now = time.monotonic()
                if (
                    s.closed
                    or now - s.created > settings.session_ttl
                    or now - s.touched > settings.idle_timeout
                ):
                    await remove(id)

    @asynccontextmanager
    async def lifespan(app):
        Path(settings.runtime_dir).mkdir(parents=True, exist_ok=True)
        await pool.start()
        task = asyncio.create_task(reaper())
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await asyncio.gather(*(remove(id) for id in list(sessions)))
        await pool.close()

    app = FastAPI(title="openspline", version="0.1.0", lifespan=lifespan)
    app.state.pool = pool
    app.state.sessions = sessions
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.exception_handler(HTTPException)
    async def error(request, exc):
        detail = (
            exc.detail
            if isinstance(exc.detail, dict)
            else {"code": "request", "message": exc.detail}
        )
        return JSONResponse({"error": detail}, status_code=exc.status_code)

    def get(id):
        s = sessions.get(id)
        if not s or s.closed:
            raise HTTPException(404, "Unknown or closed session")
        return s

    def authorize(request, s, viewer=False):
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        expected = s.viewer_token if viewer else s.publisher_token
        if not secrets.compare_digest(token, expected):
            raise HTTPException(401, "Invalid session token")
        if viewer and time.time() > s.token_expires:
            raise HTTPException(401, "Playback token expired; request a new descriptor")
        s.touch()

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        workers = [
            {
                "id": w.config.id,
                "quality": w.config.quality,
                "ready": w.ready,
                "occupied": w.owner is not None,
            }
            for w in pool.workers
        ]
        return JSONResponse(
            {"workers": workers, "backend": settings.backend},
            status_code=200 if any(w.ready for w in pool.workers) else 503,
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        lines = ["# TYPE openspline_worker_occupied gauge"]
        for w in pool.workers:
            label = json.dumps(w.config.id)
            lines += [
                f"openspline_worker_occupied{{worker={label}}} {int(w.owner is not None)}",
                f"openspline_worker_ready{{worker={label}}} {int(w.ready)}",
            ]
            for key in ("gpu_memory_bytes", "gpu_peak_allocated_bytes"):
                if key in w.info:
                    lines.append(f"openspline_{key}{{worker={label}}} {w.info[key]}")
        lines.append(f"openspline_sessions {len(sessions)}")
        for s in sessions.values():
            lines.append(
                f'openspline_queue_seconds{{session="{s.id}"}} {(s.timeline.submitted - s.timeline.played) / 48000}'
            )
            lines.append(f'openspline_queue_packets{{session="{s.id}"}} {s.input.qsize()}')
            for k, v in s.metrics.items():
                lines.append(f'openspline_{k}{{session="{s.id}"}} {v}')
        return "\n".join(lines) + "\n"

    @app.post("/v1/sessions", status_code=201)
    async def create(portrait: UploadFile = File(...), quality: str = Form("low")):
        if quality not in {"low", "high"}:
            raise HTTPException(422, "quality must be low or high")
        raw = await portrait.read(10 * 1024 * 1024 + 1)
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(413, "Portrait limit is 10 MiB")
        try:
            picture = Image.open(io.BytesIO(raw))
            if picture.width * picture.height > 16_000_000:
                raise ValueError("Portrait limit is 16 megapixels")
            picture = picture.convert("RGB")
        except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, str(exc)) from exc
        id = secrets.token_hex(16)
        try:
            worker = await pool.acquire(quality, id)
        except CapacityError as exc:
            raise HTTPException(429, {"code": "capacity", "message": str(exc)}) from exc
        except ValueError as exc:
            raise HTTPException(422, {"code": "configuration", "message": str(exc)}) from exc
        folder = Path(settings.runtime_dir) / "sessions" / id
        folder.mkdir(parents=True)
        image = folder / "portrait.jpg"
        picture.save(image)
        s = Session(id, worker, image, settings)
        try:
            await s.start()
        except Exception as exc:
            await pool.release(worker, id)
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(
                503,
                {"code": "inference", "message": "Avatar preparation failed; check worker health"},
            ) from exc
        sessions[id] = s
        return {
            **s.descriptor(),
            "publisher_token": s.publisher_token,
            "quality": quality,
            "fps": worker.info["fps"],
        }

    @app.get("/v1/sessions/{id}")
    async def status(id: str, request: Request):
        s = get(id)
        authorize(request, s)
        return {
            "id": id,
            "viewer_ready": s.viewer_ready.is_set(),
            "epoch": s.epoch,
            "metrics": s.metrics,
        }

    @app.post("/v1/sessions/{id}/viewer")
    async def descriptor(id: str, request: Request):
        s = get(id)
        authorize(request, s)
        s.viewer_token = secrets.token_urlsafe(32)
        s.token_expires = time.time() + settings.token_ttl
        return s.descriptor()

    @app.delete("/v1/sessions/{id}", status_code=204)
    async def close(id: str, request: Request):
        s = sessions.get(id)
        if s:
            authorize(request, s)
            await remove(id)

    @app.post("/v1/sessions/{id}/interrupt")
    async def interrupt(id: str, request: Request):
        s = get(id)
        authorize(request, s)
        await s.interrupt()
        return {"epoch": s.epoch}

    @app.get("/v1/sessions/{id}/portrait")
    async def portrait(id: str, request: Request):
        s = get(id)
        authorize(request, s, True)
        return FileResponse(s.image, headers={"Cache-Control": "no-store"})

    @app.get("/v1/sessions/{id}/ice")
    async def ice(id: str, request: Request):
        s = get(id)
        authorize(request, s, True)
        return {"iceServers": settings.ice_servers}

    @app.post("/v1/sessions/{id}/offer")
    async def offer(id: str, request: Request):
        s = get(id)
        authorize(request, s, True)
        if s.pc or s.native_sink:
            raise HTTPException(409, "This session already has a playback destination")
        body = await request.json()
        if len(body.get("sdp", "")) > 65536 or body.get("type") != "offer":
            raise HTTPException(422, "Invalid SDP offer")
        pc = RTCPeerConnection(
            RTCConfiguration(iceServers=[RTCIceServer(**x) for x in settings.ice_servers])
        )
        s.pc = pc
        pc.addTrack(AudioTrack(s.timeline))
        pc.addTrack(VideoTrack(s.timeline))

        async def clock():
            while pc.connectionState not in {"closed", "failed"}:
                await asyncio.sleep(0.1)
                if s.channel and s.channel.readyState == "open":
                    s.channel.send(
                        json.dumps({"type": "clock", "epoch": s.epoch, "samples": s.timeline.sent})
                    )

        clock_task = asyncio.create_task(clock())

        @pc.on("datachannel")
        def channel(ch):
            if s.channel:
                return
            s.channel = ch

            @ch.on("message")
            def message(raw):
                try:
                    event = json.loads(raw)
                    if event.get("type") == "ready":
                        s.viewer_ready.set()
                        s.emit({"type": "viewer_ready"})
                    elif event.get("type") == "played":
                        s.playback(event["samples"], event["epoch"])
                    elif event.get("type") == "ping":
                        s.touch()
                except (ValueError, KeyError, TypeError):
                    pass

        @pc.on("connectionstatechange")
        async def state():
            if pc.connectionState in {"failed", "closed"}:
                s.viewer_ready.clear()
                clock_task.cancel()
                if s.pc is pc:
                    s.pc = None
                    s.channel = None
                if pc.connectionState != "closed":
                    await pc.close()
                if not s.closed:
                    s.emit({"type": "viewer_disconnected"})

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type="offer"))
            await pc.setLocalDescription(await pc.createAnswer())
            return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        except Exception:
            await pc.close()
            s.pc = None
            raise

    @app.websocket("/v1/sessions/{id}/audio")
    async def audio(ws: WebSocket, id: str):
        await ws.accept()
        s = sessions.get(id)
        try:
            auth = await asyncio.wait_for(ws.receive_json(), 5)
            if (
                not s
                or s.closed
                or not secrets.compare_digest(auth.get("token", ""), s.publisher_token)
            ):
                await ws.close(4401)
                return
            if s.publisher_connected:
                await ws.close(4409)
                return
            s.publisher_connected = True
            s.touch()

            async def output():
                while True:
                    await ws.send_json(await s.events.get())

            sender = asyncio.create_task(output())
            s.emit({"type": "ready", "viewer_ready": s.viewer_ready.is_set()})
            pending = set()

            async def finish(event):
                try:
                    target, epoch = await s.end_turn()
                    if event.get("drain", True):
                        await s.drain(target, epoch)
                    s.emit({"type": "ack", "id": event["id"], "target_samples": target})
                except Exception as exc:
                    s.emit(
                        {
                            "type": "error",
                            "id": event["id"],
                            "code": "playback",
                            "message": str(exc),
                        }
                    )

            try:
                while True:
                    event = await ws.receive_json()
                    s.touch()
                    kind = event.get("type")
                    try:
                        if kind == "audio":
                            if event.get("epoch", s.epoch) != s.epoch:
                                s.emit({"type": "ack", "id": event.get("id"), "discarded": True})
                                continue
                            if len(event.get("data", "")) > 90000:
                                raise ValueError("Audio packet too large")
                            await s.push(
                                base64.b64decode(event["data"], validate=True),
                                event.get("format", {}),
                            )
                        elif kind == "end_turn":
                            if pending:
                                raise ValueError("A turn is already ending")
                            task = asyncio.create_task(finish(event))
                            pending.add(task)
                            task.add_done_callback(pending.discard)
                            continue
                        elif kind == "interrupt":
                            await s.interrupt()
                        elif kind == "ping":
                            pass
                        else:
                            raise ValueError("Unknown message type")
                        s.emit({"type": "ack", "id": event.get("id")})
                    except (ValueError, KeyError, RuntimeError) as exc:
                        s.emit(
                            {
                                "type": "error",
                                "id": event.get("id"),
                                "code": "request",
                                "message": str(exc),
                            }
                        )
            finally:
                sender.cancel()
                for task in pending:
                    task.cancel()
                await asyncio.gather(sender, *pending, return_exceptions=True)
                s.publisher_connected = False
                await remove(id)
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass

    @app.post("/v1/sessions/{id}/file")
    async def file(id: str, request: Request, audio: UploadFile = File(...)):
        s = get(id)
        authorize(request, s)
        try:
            await asyncio.wait_for(s.viewer_ready.wait(), 60)
        except asyncio.TimeoutError:
            raise HTTPException(408, "Connect a playback destination before uploading audio")
        data = await audio.read(64 * 1024 * 1024 + 1)
        if len(data) > 64 * 1024 * 1024:
            raise HTTPException(413, "Audio file limit is 64 MiB")
        try:
            container = av.open(io.BytesIO(data))
            resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
            total = 0
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    samples = out.to_ndarray().reshape(-1)
                    total += len(samples)
                    if total > 48000 * 600:
                        raise ValueError("Audio file limit is ten minutes")
                    for start in range(0, len(samples), 4800):
                        await s.push(
                            samples[start : start + 4800].tobytes(), {"sample_rate": 48000}
                        )
            for out in resampler.resample(None):
                await s.push(out.to_ndarray().tobytes(), {"sample_rate": 48000})
            container.close()
        except (av.error.FFmpegError, ValueError) as exc:
            raise HTTPException(422, "Invalid or oversized audio file") from exc
        return {"samples": total}

    @app.post("/v1/sessions/{id}/livekit")
    async def attach_livekit(id: str, request: Request):
        s = get(id)
        authorize(request, s)
        if s.pc or s.native_sink:
            raise HTTPException(409, "Session already has a playback destination")
        try:
            from .livekit import LiveKitSink
        except ImportError as exc:
            raise HTTPException(503, "Install the server LiveKit extra") from exc
        body = await request.json()
        if not all(
            isinstance(body.get(k), str) and body[k] for k in ("url", "token", "sender_identity")
        ):
            raise HTTPException(422, "url, token and sender_identity are required")
        s.native_sink = LiveKitSink(s)
        try:
            await s.native_sink.start(body["url"], body["token"], body["sender_identity"])
        except Exception:
            await s.native_sink.close()
            s.native_sink = None
            raise HTTPException(502, "Could not connect avatar worker to LiveKit")
        return {"ready": True}

    @app.websocket("/v1/sessions/{id}/demo/{provider}")
    async def demo(ws: WebSocket, id: str, provider: str):
        from .demo import run_demo

        await run_demo(ws, sessions.get(id), provider, remove)

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/")
        async def index():
            return FileResponse(STATIC / "index.html")

        @app.get("/view")
        async def view():
            return FileResponse(STATIC / "view.html")

    return app
