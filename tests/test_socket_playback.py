import asyncio
import base64
import contextlib
import heapq
import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from openspline_server.app import create_app
from openspline_server.config import Settings
from openspline_server.media import Segment, Timeline
from openspline_server.socket_playback import PLAYBACK_WINDOW_SAMPLES, SocketPlayback
from PIL import Image
from starlette.websockets import WebSocketDisconnect


def portrait():
    data = io.BytesIO()
    Image.new("RGB", (64, 64), "blue").save(data, format="PNG")
    return data.getvalue()


def test_playback_auth_ownership_file_and_release(tmp_path):
    with TestClient(create_app(Settings(backend="test", runtime_dir=str(tmp_path)))) as client:
        s = client.post("/v1/sessions", files={"portrait": ("p.png", portrait())}).json()
        path = f"/v1/sessions/{s['id']}"
        with client.websocket_connect(path + "/playback") as ws:
            ws.send_json({"token": s["publisher_token"]})
            with pytest.raises(WebSocketDisconnect) as error:
                ws.receive_json()
            assert error.value.code == 4401
        with client.websocket_connect(path + "/playback") as ws:
            ws.send_json({"token": s["token"]})
            assert ws.receive_json()["type"] == "connected"
            headers = {"Authorization": "Bearer " + s["token"]}
            assert client.post(path + "/offer", headers=headers, json={}).status_code == 409
            with client.websocket_connect(path + "/playback") as duplicate:
                duplicate.send_json({"token": s["token"]})
                with pytest.raises(WebSocketDisconnect) as error:
                    duplicate.receive_json()
                assert error.value.code == 4409
            ws.send_json({"type": "ready"})
            assert ws.receive_json()["type"] == "playback_ready"
            audio = io.BytesIO()
            with wave.open(audio, "wb") as wav:
                wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                wav.writeframes(np.ones(5000, dtype="<i2").tobytes())
            headers = {"Authorization": "Bearer " + s["publisher_token"]}
            assert (
                client.post(
                    path + "/file", headers=headers, files={"audio": ("a.wav", audio.getvalue())}
                ).status_code
                == 200
            )
            with client.websocket_connect(path + "/audio") as publisher:
                publisher.send_json({"token": s["publisher_token"]})
                while publisher.receive_json()["type"] != "ready":
                    pass
                publisher.send_json({"type": "end_turn", "id": "end"})
                played = 0
                while played < 5000:
                    event = ws.receive_json()
                    if event["type"] == "media":
                        pcm = np.frombuffer(base64.b64decode(event["audio"]), dtype="<i2")
                        assert np.all(pcm == 1)
                        played += len(pcm)
                        assert event["samples"] == played
                        assert Image.open(io.BytesIO(base64.b64decode(event["image"]))).size == (
                            512,
                            512,
                        )
                        ws.send_json({"type": "played", "samples": played, "epoch": event["epoch"]})
                while publisher.receive_json().get("id") != "end":
                    pass
                assert played == 5000  # Final 40ms packet is trimmed, never padded for playback.
                ws.send_json(
                    {"type": "playback_stats", "epoch": 0, "buffer_ms": 160, "underruns": 2}
                )
                ws.send_json({"type": "ready"})
                while ws.receive_json()["type"] != "playback_ready":
                    pass
                metrics = client.get("/metrics").text
                assert f'openspline_playback_buffer_seconds{{session="{s["id"]}"}} 0.16' in metrics
                assert f'openspline_playback_underruns{{session="{s["id"]}"}} 2' in metrics
                assert client.delete(path, headers=headers).status_code == 204
        assert client.get("/readyz").json()["workers"][0]["occupied"] is False


async def test_slow_playback_is_bounded_and_interrupt_discards_encoding():
    timeline = Timeline(np.zeros((16, 16, 3), dtype=np.uint8))
    ready = asyncio.Event()
    ready.set()
    session = SimpleNamespace(timeline=timeline, viewer_ready=ready, epoch=0, closed=False)
    packets = []

    class Socket:
        async def send_json(self, event):
            packets.append(event)

        async def close(self, **kwargs):
            pass

    await timeline.put(
        Segment(np.zeros(96000, dtype=np.int16), np.zeros((50, 16, 16, 3), dtype=np.uint8), 25, 0)
    )
    sink = SocketPlayback(session, Socket())
    task = asyncio.create_task(sink.send())
    try:
        await asyncio.sleep(0.2)
        assert timeline.sent == PLAYBACK_WINDOW_SAMPLES
        assert len([p for p in packets if p["type"] == "media"]) == 25
        await asyncio.sleep(0.1)
        assert (
            timeline.sent == PLAYBACK_WINDOW_SAMPLES
        )  # No feedback: no extra generation consumed.
        session.epoch = 1
        timeline.clear(1)
        sink.emit({"type": "interrupted", "epoch": 1})
        await timeline.put(
            Segment(np.zeros(120, dtype=np.int16), np.zeros((1, 16, 16, 3), dtype=np.uint8), 25, 1)
        )
        await asyncio.sleep(0.1)
        assert packets[-2]["type"] == "interrupted"
        assert packets[-1]["epoch"] == 1
        assert packets[-1]["samples"] == 120
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("round_trip", [0.02, 0.1, 0.2, 0.4])
async def test_wan_feedback_does_not_create_audio_gaps(monkeypatch, round_trip):
    """Deterministic WAN clock: real sender and timeline, delayed played feedback."""
    from openspline_server import socket_playback

    timeline = Timeline(np.zeros((16, 16, 3), dtype=np.uint8))
    ready = asyncio.Event()
    ready.set()
    session = SimpleNamespace(timeline=timeline, viewer_ready=ready, epoch=0, closed=False)
    await timeline.put(
        Segment(np.zeros(96000, np.int16), np.zeros((50, 16, 16, 3), np.uint8), 25, 0)
    )
    now = end = gaps = 0.0
    packets = 0
    feedback = []

    async def advance(seconds):
        nonlocal now
        now += seconds
        assert now < 10, "Playback stalled"
        while feedback and feedback[0][0] <= now:
            _, samples = heapq.heappop(feedback)
            timeline.played = max(timeline.played, samples)
        if timeline.played == 96000:
            session.closed = True

    async def encode(*args):
        return "jpeg"

    monkeypatch.setattr(
        socket_playback,
        "asyncio",
        SimpleNamespace(Queue=asyncio.Queue, sleep=advance, to_thread=encode),
    )

    class Network:
        async def send_json(self, event):
            nonlocal end, gaps, packets
            if event["type"] != "media":
                return
            assert timeline.sent - timeline.played <= PLAYBACK_WINDOW_SAMPLES
            arrival = now + round_trip / 2
            at = end if end >= arrival + 0.005 else arrival + 0.16
            if packets:
                gaps += max(0, at - end)
            packets += 1
            end = at + len(base64.b64decode(event["audio"])) / 2 / 48000
            heapq.heappush(feedback, (end + 0.04 + round_trip / 2, event["samples"]))

        async def close(self, **kwargs):
            pass

    await SocketPlayback(session, Network()).send()
    assert packets == 50
    assert gaps == 0
