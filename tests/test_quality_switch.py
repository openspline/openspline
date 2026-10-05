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
@pytest.mark.parametrize("group", [None, (0, 1)])
def test_failed_quality_change_preserves_previous_service(tmp_path, monkeypatch, stage, group):
    app = create_app(Settings(backend="test", runtime_dir=str(tmp_path), demo_high_devices=group))
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
                    assert worker.config.devices == (group or (0,))
                    worker.ready = False
                    worker.error = "CUDA out of memory"
                else:
                    await original()

            monkeypatch.setattr(worker, "start", fail)
        job = wait(client, client.post("/v1/demo/quality", json={"quality": "high"}))
        assert job["state"] == "error"
        assert ("Model files missing" if stage == "check" else "CUDA out of memory") in job["error"]
        assert worker.ready and worker.config.quality == "low" and worker.owner is None
        assert worker.config.devices == (0,)
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
        assert not client.get("/v1/demo/gpus").json()["editable"]
        assert (
            client.post("/v1/demo/quality", json={"quality": "low", "devices": [0]}).status_code
            == 422
        )
        assert client.post("/v1/demo/quality", json={"quality": "high"}).status_code == 422
        assert client.post("/v1/demo/quality", json={"quality": []}).status_code == 422


@pytest.mark.parametrize("initial", ["low", "high"])
def test_demo_switch_keeps_the_full_high_gpu_group(tmp_path, initial):
    settings = Settings(
        backend="test",
        runtime_dir=str(tmp_path),
        workers=[WorkerConfig("pair", initial, (0,) if initial == "low" else (0, 1))],
        demo_high_devices=(0, 1) if initial == "low" else None,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/readyz").json()["demo_devices"] == {"low": [0], "high": [0, 1]}
        worker = app.state.pool.workers[0]
        for quality in ["low", "high", "low", "high"]:
            result = client.post("/v1/demo/quality", json={"quality": quality})
            job = wait(client, result) if result.status_code == 202 else result.json()
            assert job["state"] == "ready"
            expected = (0, 1) if quality == "high" else (0,)
            assert worker.config.devices == expected
            ready = client.get("/readyz").json()["workers"][0]
            assert ready["devices"] == list(expected)
            assert ready["inference_ranks"] == 1  # CPU test backend, not a GPU benchmark.


@pytest.mark.parametrize(
    "workers,group",
    [
        ([WorkerConfig()], (0, 0)),
        ([WorkerConfig()], (1, 2)),
        ([WorkerConfig("a"), WorkerConfig("b", devices=(1,))], (0, 1)),
        ([WorkerConfig(quality="high")], (0, 1)),
    ],
)
def test_invalid_demo_groups_fail_at_startup(workers, group):
    with pytest.raises(ValueError):
        Settings(workers=workers, demo_high_devices=group)


def test_demo_selects_single_gpu_or_pair_and_changes_same_quality(tmp_path):
    settings = Settings(backend="test", runtime_dir=str(tmp_path), demo_high_devices=(0, 1))
    saved = []
    settings.save_demo_selection = lambda devices, quality: saved.append((devices, quality))
    app = create_app(settings)
    with TestClient(app) as client:
        options = client.get("/v1/demo/gpus").json()
        assert options["editable"] and options["persistent"]
        assert [d["id"] for d in options["gpus"]] == [0, 1]
        assert (
            client.post("/v1/demo/quality", json={"quality": "low", "devices": [0, 1]}).status_code
            == 422
        )
        worker = app.state.pool.workers[0]
        for quality, devices in [("high", [1]), ("high", [0, 1]), ("low", [0])]:
            pid = worker.process.pid
            job = wait(
                client,
                client.post("/v1/demo/quality", json={"quality": quality, "devices": devices}),
            )
            assert job["state"] == "ready", job
            assert worker.config.devices == tuple(devices)
            assert worker.process.pid != pid
        assert saved == [([1], "high"), ([0, 1], "high"), ([0, 1], "low")]
        assert client.get("/v1/demo/gpus").json()["selected"]["high"] == [0, 1]
        pid = worker.process.pid
        job = wait(client, client.post("/v1/demo/quality", json={"quality": "low", "devices": [0]}))
        assert job["state"] == "ready" and worker.process.pid == pid


@pytest.mark.parametrize("devices", [[], [0, 0], [True], [2], "0,1", ["0"], [-1]])
def test_demo_rejects_invalid_gpu_selection_before_unloading(tmp_path, devices):
    app = create_app(Settings(backend="test", runtime_dir=str(tmp_path)))
    with TestClient(app) as client:
        pid = app.state.pool.workers[0].process.pid
        assert (
            client.post(
                "/v1/demo/quality", json={"quality": "high", "devices": devices}
            ).status_code
            == 422
        )
        assert app.state.pool.workers[0].process.pid == pid


@pytest.mark.parametrize("failure", ["load", "save"])
def test_failed_gpu_change_rolls_back_selection(tmp_path, monkeypatch, failure):
    settings = Settings(backend="test", runtime_dir=str(tmp_path), demo_high_devices=(0, 1))
    saved = []

    def save(devices, quality):
        if failure == "save":
            raise OSError("Selection file is read-only")
        saved.append(devices)

    settings.save_demo_selection = save
    app = create_app(settings)
    with TestClient(app) as client:
        worker = app.state.pool.workers[0]
        start = worker.start

        async def failing_start():
            if worker.config.devices == (1,) and failure == "load":
                worker.ready = False
                worker.error = "GPU is unavailable"
            else:
                await start()

        monkeypatch.setattr(worker, "start", failing_start)
        job = wait(
            client, client.post("/v1/demo/quality", json={"quality": "high", "devices": [1]})
        )
        assert job["state"] == "error" and not saved
        assert worker.ready and worker.config.devices == (0,) and worker.config.quality == "low"
        assert worker.owner is None
        assert app.state.pool.demo_devices == {"low": (0,), "high": (0, 1)}


async def test_gpu_change_cannot_displace_session_or_race_another_change(tmp_path):
    pool = WorkerPool(Settings(backend="test", runtime_dir=str(tmp_path), demo_high_devices=(0, 1)))
    await pool.start()
    try:
        worker = await pool.acquire("low", "active-session")
        with pytest.raises(CapacityError):
            await pool.reserve_quality_switch("high", "switch", [0, 1])
        await pool.release(worker, "active-session")
        results = await asyncio.gather(
            pool.reserve_quality_switch("high", "a", [0, 1]),
            pool.reserve_quality_switch("high", "b", [1]),
            return_exceptions=True,
        )
        assert sum(isinstance(result, CapacityError) for result in results) == 1
    finally:
        await pool.close()
