import asyncio

import numpy as np
import pytest
from openspline_server.audio import AudioConverter
from openspline_server.config import Settings, WorkerConfig
from openspline_server.worker import CapacityError, WorkerPool
from PIL import Image


@pytest.fixture
def portrait(tmp_path):
    p = tmp_path / "face.png"
    Image.new("RGB", (64, 64), "#dc6145").save(p)
    return p


def test_overlapping_devices():
    with pytest.raises(ValueError, match="overlap"):
        Settings(workers=[WorkerConfig("a", "high", (0, 1)), WorkerConfig("b", "low", (1,))])


def test_partial_samples_and_resampling():
    raw = (np.sin(np.arange(2400) * 0.07) * 16000).astype("<i2").tobytes()
    a = AudioConverter()
    expected = np.concatenate([a.push(raw), a.flush()])
    b = AudioConverter()
    parts = [b.push(raw[i : i + 17]) for i in range(0, len(raw), 17)]
    parts.append(b.flush())
    actual = np.concatenate(parts)
    assert len(actual) == 4800
    np.testing.assert_array_equal(expected, actual)


def test_incomplete_sample_rejected():
    c = AudioConverter()
    c.push(b"\x00")
    with pytest.raises(ValueError, match="incomplete"):
        c.flush()


async def test_worker_exclusive_release_and_state(tmp_path, portrait):
    pool = WorkerPool(Settings(backend="test", runtime_dir=str(tmp_path)))
    await pool.start()
    try:
        results = await asyncio.gather(
            pool.acquire("low", "a"), pool.acquire("low", "b"), return_exceptions=True
        )
        assert sum(isinstance(x, CapacityError) for x in results) == 1
        w = next(x for x in results if not isinstance(x, Exception))
        await w.call("prepare", str(portrait), 0)
        frames, elapsed = await w.call("infer", np.zeros(128000, dtype=np.float32))
        assert frames.shape == (24, 512, 512, 3)
        await pool.release(w, w.owner)
        assert (await pool.acquire("low", "next")) is w
    finally:
        await pool.close()


async def test_unconfigured_quality(tmp_path):
    pool = WorkerPool(Settings(backend="test", runtime_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="not configured"):
        await pool.acquire("high", "a")


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 44100, 48000])
def test_tiny_audio_preserves_duration(rate):
    c = AudioConverter(sample_rate=rate)
    result = np.concatenate([c.push(bytes(4)), c.flush()])
    assert len(result) == round(2 * 48000 / rate)


async def test_recovery_after_worker_exit(tmp_path, portrait):
    pool = WorkerPool(Settings(backend="test", runtime_dir=str(tmp_path)))
    await pool.start()
    try:
        worker = await pool.acquire("low", "first")
        worker.process.terminate()
        worker.process.join()
        from openspline_server.worker import WorkerFailure

        with pytest.raises(WorkerFailure):
            await worker.call("prepare", str(portrait), 0)
        await pool.release(worker, "first")
        reused = await pool.acquire("low", "second")
        assert reused.ready
    finally:
        await pool.close()


def test_mp3_chunk_boundaries(tmp_path):
    import shutil
    import subprocess

    from openspline_server.audio import EncodedDecoder

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg needed for encoded fixture")
    path = tmp_path / "tone.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=200:duration=0.2",
            "-ar",
            "24000",
            str(path),
        ],
        check=True,
    )
    raw = path.read_bytes()
    whole = EncodedDecoder("mp3")
    expected = np.concatenate([whole.push(raw), whole.flush()])
    split = EncodedDecoder("mp3")
    parts = [split.push(raw[i : i + 31]) for i in range(0, len(raw), 31)]
    parts.append(split.flush())
    np.testing.assert_array_equal(np.concatenate(parts), expected)
    assert len(expected) > 0


async def test_bounded_playback_and_interrupt_releases_producer():
    from openspline_server.media import Segment, Timeline

    timeline = Timeline(np.zeros((2, 2, 3), dtype=np.uint8))

    def segment(epoch=0):
        return Segment(
            np.zeros(960, dtype=np.int16), np.zeros((1, 2, 2, 3), dtype=np.uint8), 25, epoch
        )

    for _ in range(3):
        await timeline.put(segment())
    waiting = asyncio.create_task(timeline.put(segment()))
    await asyncio.sleep(0)
    assert not waiting.done()
    timeline.clear(1)
    await asyncio.wait_for(waiting, 1)
    assert len(timeline.queue) == 0 and timeline.submitted == 0
