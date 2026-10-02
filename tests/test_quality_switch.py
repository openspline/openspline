import asyncio
import io
import time

import pytest
from fastapi.testclient import TestClient
from openspline_server.app import create_app
from openspline_server.config import Settings, WorkerConfig
from openspline_server.worker import CapacityError, WorkerPool
from PIL import Image


def wait(client, response):
    assert response.status_code == 202, response.text
    path = "/v1/demo/quality/" + response.json()["id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(path).json()
        if job["state"] in {"ready", "error"}:
            return job
        time.sleep(0.01)
    pytest.fail("Quality change never completed")


def portrait():
    data = io.BytesIO()
    Image.new("RGB", (64, 64), "blue").save(data, format="PNG")
    return data.getvalue()


def test_demo_switches_both_directions_on_one_worker(tmp_path):
    app = create_app(Settings(backend="test", runtime_dir=str(tmp_path)))
    with TestClient(app) as client:
        worker = app.state.pool.workers[0]
        assert client.get("/readyz").json()["demo_qualities"] == ["low", "high"]
        pid = worker.process.pid
        for quality in ["high", "low"]:
            job = wait(client, client.post("/v1/demo/quality", json={"quality": quality}))
            assert job["state"] == "ready", job
            assert (
                worker.ready and worker.config.quality == quality and worker.config.devices == (0,)
            )
            assert worker.owner is None and worker.process.pid != pid
            pid = worker.process.pid
            # Re-selecting the loaded quality keeps the model process alive.
            response = client.post("/v1/demo/quality", json={"quality": quality})
            assert response.status_code == 200 and response.json()["state"] == "ready"
            assert worker.process.pid == pid
            response = client.post(
                "/v1/sessions", data={"quality": quality}, files={"portrait": ("p.png", portrait())}
            )
            assert response.status_code == 201, response.text
            session = response.json()
            assert (
                client.post(
                    "/v1/demo/quality", json={"quality": "low" if quality == "high" else "high"}
                ).status_code
                == 429
            )
            assert worker.process.pid == pid and worker.owner == session["id"]
            client.delete(
                f"/v1/sessions/{session['id']}",
                headers={"Authorization": "Bearer " + session["publisher_token"]},
            )


@pytest.mark.parametrize("stage", ["check", "load"])
def test_failed_quality_change_preserves_previous_service(tmp_path, monkeypatch, stage):
    app = create_app(Settings(backend="test", runtime_dir=str(tmp_path)))
    with TestClient(app) as client:
        worker = app.state.pool.workers[0]
        pid = worker.process.pid
        if stage == "check":

            async def fail(*args):
                raise RuntimeError("Model files missing")

            monkeypatch.setattr("openspline_server.quality.prepare_quality", fail)
        else:
            original = worker.start

            async def fail():
                if worker.config.quality == "high":
                    worker.ready = False
                    worker.error = "CUDA out of memory"
                else:
                    await original()

            monkeypatch.setattr(worker, "start", fail)
        job = wait(client, client.post("/v1/demo/quality", json={"quality": "high"}))
        assert job["state"] == "error"
        assert ("Model files missing" if stage == "check" else "CUDA out of memory") in job["error"]
        assert worker.ready and worker.config.quality == "low" and worker.owner is None
        if stage == "check":
            assert worker.process.pid == pid
        assert (
            client.post("/v1/sessions", files={"portrait": ("p.png", portrait())}).status_code
            == 201
        )


async def test_switch_reservation_is_atomic_and_shutdown_releases_it(tmp_path, monkeypatch):
    pool = WorkerPool(Settings(backend="test", runtime_dir=str(tmp_path)))
    await pool.start()
    started = asyncio.Event()

    async def check_models(*args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("openspline_server.quality.prepare_quality", check_models)
    try:
        results = await asyncio.gather(
            pool.reserve_quality_switch("high", "a"),
            pool.reserve_quality_switch("high", "b"),
            return_exceptions=True,
        )
        assert sum(isinstance(result, CapacityError) for result in results) == 1
        worker = pool.workers[0]
        task = asyncio.create_task(
            pool.switch_quality(worker, "high", worker.owner, lambda state: None)
        )
        await started.wait()
        with pytest.raises(CapacityError):
            await pool.acquire("low", "session")
        await pool.recover_idle()
        assert not getattr(worker, "recovering", None)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker.owner is None and worker.config.quality == "low" and worker.process is None
    finally:
        await pool.close()


def test_multi_worker_demo_exposes_only_configured_profiles(tmp_path):
    settings = Settings(
        backend="test",
        runtime_dir=str(tmp_path),
        workers=[WorkerConfig("a", devices=(0,)), WorkerConfig("b", devices=(1,))],
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/readyz").json()["demo_qualities"] == ["low"]
        assert client.post("/v1/demo/quality", json={"quality": "high"}).status_code == 422
        assert client.post("/v1/demo/quality", json={"quality": []}).status_code == 422
