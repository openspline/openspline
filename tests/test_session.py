import numpy as np
from openspline_server.config import Settings
from openspline_server.session import Session
from openspline_server.worker import WorkerPool
from PIL import Image


async def test_short_utterance_flush_interrupt_and_reset(tmp_path):
    settings = Settings(backend="test", runtime_dir=str(tmp_path))
    pool = WorkerPool(settings)
    await pool.start()
    p = tmp_path / "portrait.png"
    Image.new("RGB", (64, 64), "red").save(p)
    try:
        w = await pool.acquire("low", "s")
        s = Session("s", w, p, settings)
        await s.start()
        await s.push(np.zeros(2400, dtype="<i2").tobytes(), {"sample_rate": 24000})
        target, epoch = await s.end_turn()
        assert target == 4800
        assert s.timeline.queue[0].audio.shape == (4800,)
        await s.interrupt()
        assert not s.timeline.queue and s.epoch == epoch + 1
        await s.push(np.ones(4800, dtype="<i2").tobytes(), {"sample_rate": 24000})
        target, _ = await s.end_turn()
        assert target == 9600
        await s.close()
        await pool.release(w, "s")
    finally:
        await pool.close()
