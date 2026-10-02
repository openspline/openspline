import base64
import io

from fastapi.testclient import TestClient
from openspline_server.app import create_app
from openspline_server.config import Settings, WorkerConfig
from PIL import Image


def image(color="blue"):
    b = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(b, format="PNG")
    return b.getvalue()


def test_auth_capacity_tokens_and_release(tmp_path):
    with TestClient(
        create_app(Settings(backend="test", runtime_dir=str(tmp_path), api_key="test-key"))
    ) as client:
        assert client.get("/readyz").status_code == 200
        assert (
            client.post("/v1/sessions", files={"portrait": ("p.png", image())}).status_code == 401
        )
        auth = {"Authorization": "Bearer test-key"}
        r = client.post("/v1/sessions", headers=auth, files={"portrait": ("p.png", image())})
        assert r.status_code == 201, r.text
        s = r.json()
        assert "publisher_token" in s
        assert (
            client.post(
                "/v1/sessions", headers=auth, files={"portrait": ("p.png", image())}
            ).status_code
            == 429
        )
        assert (
            client.post(
                f"/v1/sessions/{s['id']}/interrupt",
                headers={"Authorization": "Bearer " + s["token"]},
            ).status_code
            == 401
        )
        assert (
            client.delete(
                f"/v1/sessions/{s['id']}",
                headers={"Authorization": "Bearer " + s["publisher_token"]},
            ).status_code
            == 204
        )
        assert (
            client.post(
                "/v1/sessions", headers=auth, files={"portrait": ("p.png", image("red"))}
            ).status_code
            == 201
        )


def test_socket_short_utterance_and_epoch(tmp_path):
    with TestClient(create_app(Settings(backend="test", runtime_dir=str(tmp_path)))) as client:
        s = client.post("/v1/sessions", files={"portrait": ("p.png", image())}).json()
        with client.websocket_connect(f"/v1/sessions/{s['id']}/audio") as ws:
            ws.send_json({"token": s["publisher_token"]})
            assert ws.receive_json()["type"] == "ready"
            ws.send_json(
                {
                    "type": "audio",
                    "id": "a",
                    "data": base64.b64encode(bytes(4800)).decode(),
                    "format": {"sample_rate": 24000},
                }
            )
            assert ws.receive_json()["id"] == "a"
            ws.send_json({"type": "end_turn", "id": "e", "drain": False})
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg.get("id") == "e":
                    break
            assert messages[-1]["target_samples"] == 4800
            ws.send_json({"type": "interrupt", "id": "i"})
            assert ws.receive_json()["type"] == "interrupted"
            assert ws.receive_json()["epoch"] == 1


def test_multiple_workers_and_quality(tmp_path):
    settings = Settings(
        backend="test",
        runtime_dir=str(tmp_path),
        workers=[WorkerConfig("a", "low", (0,)), WorkerConfig("b", "high", (1, 2))],
    )
    with TestClient(create_app(settings)) as client:
        for quality in ["low", "high"]:
            assert (
                client.post(
                    "/v1/sessions",
                    data={"quality": quality},
                    files={"portrait": ("p.png", image())},
                ).status_code
                == 201
            )
        assert (
            client.post("/v1/sessions", files={"portrait": ("p.png", image())}).status_code == 429
        )
