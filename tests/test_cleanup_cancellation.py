import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from openspline_server.app import create_app
from openspline_server.config import Settings
from openspline_server.worker import CapacityError, WorkerPool
from PIL import Image


@pytest.mark.parametrize("stage", ["session", "reset"])
async def test_shutdown_stops_reaper_when_cleanup_replaces_cancellation(
    tmp_path, monkeypatch, stage, caplog
):
    app = create_app(Settings(backend="test", runtime_dir=str(tmp_path)))
    pool = app.state.pool
    worker = pool.workers[0]
    worker.owner, worker.ready = "expired", True
    monkeypatch.setattr(pool, "start", AsyncMock())
    monkeypatch.setattr(pool, "recover_idle", AsyncMock())
    monkeypatch.setattr(pool, "close", AsyncMock())
    entered = asyncio.Event()
    reaper = None

    async def close():
        nonlocal reaper
        reaper = asyncio.current_task()
        if stage == "session":
            entered.set()
            await asyncio.Event().wait()

    async def reset(*args):
        try:
            if stage == "reset":
                entered.set()
                await asyncio.Event().wait()
        finally:
            raise OSError("reset cleanup failed")

    monkeypatch.setattr(worker, "call", reset)
    folder = tmp_path / "sessions" / "expired"
    folder.mkdir(parents=True)
    app.state.sessions["expired"] = SimpleNamespace(
        id="expired", closed=True, worker=worker, close_lock=asyncio.Lock(), close=close
    )
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    try:
        await asyncio.wait_for(entered.wait(), 7)
    finally:
        shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
        try:
            # Do not use wait_for here: its second cancellation can stop the
            # broken reaper and be suppressed by lifespan, hiding the hang.
            done, _ = await asyncio.wait({shutdown}, timeout=2)
            assert shutdown in done, "Shutdown hung after cleanup swallowed cancellation"
            await shutdown
        finally:
            if not shutdown.done():
                shutdown.cancel()
            await asyncio.gather(shutdown, return_exceptions=True)
    assert reaper.cancelled()
    assert worker.owner is None
    assert not folder.exists()
    assert not app.state.sessions
    pool.close.assert_awaited_once()
    assert "reset cleanup failed" in caplog.text


@pytest.mark.parametrize("stage", ["send", "receive"])
@pytest.mark.parametrize("outcome", ["success", "failure"])
async def test_cancelled_reset_finishes_rpc_before_releasing_worker(
    tmp_path, monkeypatch, stage, outcome
):
    pool = WorkerPool(Settings(backend="test", runtime_dir=str(tmp_path)))
    await pool.start()
    worker = await pool.acquire("low", "old")
    entered = asyncio.Event()
    proceed = threading.Event()
    loop = asyncio.get_running_loop()
    original = worker.conn.send if stage == "send" else worker._receive

    def paused(*args):
        loop.call_soon_threadsafe(entered.set)
        if not proceed.wait(10):
            raise TimeoutError("Test did not release RPC gate")
        if outcome == "failure":
            raise OSError("IPC failed")
        return original(*args)

    target, name = (worker.conn, "send") if stage == "send" else (worker, "_receive")
    monkeypatch.setattr(target, name, paused)
    release = asyncio.create_task(pool.release(worker, "old"))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        release.cancel()
        await asyncio.sleep(0)
        # A second cancellation must not abandon the still-running thread either.
        release.cancel()
        await asyncio.sleep(0)
        assert not release.done()
        assert worker.owner == "old"
        with pytest.raises(CapacityError):
            await pool.acquire("low", "too-early")
        proceed.set()
        with pytest.raises(asyncio.CancelledError) as error:
            await asyncio.wait_for(release, 3)
        assert worker.owner is None
        monkeypatch.setattr(target, name, original)
        if outcome == "failure":
            assert isinstance(error.value.__cause__, OSError)
            assert not worker.ready
            with pytest.raises(CapacityError):
                await pool.acquire("low", "before-recovery")
            await pool.recover_idle()
            await asyncio.wait_for(worker.recovering, 10)
        else:
            assert worker.ready
            assert not worker.conn.poll(), "Reset response was left unread"
        reused = await pool.acquire("low", "new")
        portrait = tmp_path / "portrait.png"
        Image.new("RGB", (64, 64), "red").save(portrait)
        await reused.call("prepare", str(portrait))
        frames, seconds = await reused.call("infer", np.zeros(128000, dtype=np.float32))
        assert frames.shape == (24, 512, 512, 3)
        assert tuple(frames[0, 0, 0]) == (255, 0, 0)
        assert seconds >= 0
        await pool.release(reused, "new")
    finally:
        proceed.set()
        await asyncio.gather(release, return_exceptions=True)
        await pool.close()
