import json
from pathlib import Path

from fastapi.testclient import TestClient
from openspline_server.app import create_app
from openspline_server.config import Settings
from test_protocol import image


def test_wire_fixture(tmp_path):
    fixture = json.loads(Path("protocol/fixtures.json").read_text())
    with TestClient(create_app(Settings(backend="test", runtime_dir=str(tmp_path)))) as client:
        session = client.post("/v1/sessions", files={"portrait": ("p.png", image())}).json()
        with client.websocket_connect(f"/v1/sessions/{session['id']}/audio") as ws:
            ws.send_json({"token": session["publisher_token"]})
            ws.receive_json()
            ws.send_json(fixture["audio"])
            assert ws.receive_json()["id"] == "audio-1"
            ws.send_json(fixture["end_turn"])
            while (event := ws.receive_json()).get("id") != "end-1":
                pass
            assert event["target_samples"] == 4
