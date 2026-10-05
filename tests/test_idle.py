import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from openspline_server.config import Settings
from openspline_server.media import AudioTrack, Segment, Timeline, VideoTrack
from openspline_server.session import Session
from openspline_server.socket_playback import SocketPlayback
from PIL import Image


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), 3)


class MotionWorker:
    info = {"fps": 25, "chunk_samples": 1280, "cache_samples": 6400}

    def __init__(self):
        self.calls = []
        self.hold = None

    async def call(self, op, *args):
        self.calls.append((op, args))
        if op == "infer":
            marker = len(self.calls)
            if self.hold:
                await self.hold.wait()
            return np.full((2, 8, 8, 3), marker, dtype=np.uint8), 0.01


async def make_session(tmp_path):
    path = tmp_path / "portrait.png"
    Image.new("RGB", (8, 8), "red").save(path)
    worker = MotionWorker()
    session = Session("s", worker, path, Settings(backend="test"))
    await session.start()
    return session, worker


async def test_idle_generates_bounded_motion_without_changing_speech_playback(tmp_path):
    s, worker = await make_session(tmp_path)
    try:
        await asyncio.sleep(0.1)
        assert [op for op, _ in worker.calls] == ["prepare"]
        s.viewer_ready.set()
        await until(lambda: s.timeline.idle_next is not None)
        await asyncio.sleep(0.12)
        assert len(worker.calls) == 2  # No viewer consumption: stop generating ahead.
        assert not np.any(worker.calls[-1][1][0])
        audio, video = AudioTrack(s.timeline), VideoTrack(s.timeline)
        await audio.recv()
        picture = await video.recv()
        assert np.all(picture.to_ndarray(format="rgb24") == 2)
        await until(lambda: s.metrics["idle_blocks"] == 2)
        assert s.timeline.submitted == s.timeline.sent == s.timeline.played == 0

        await s.push(np.ones(480, dtype="<i2").tobytes(), {"sample_rate": 48000})
        target, epoch = await s.end_turn()
        assert target == 480
        assert np.all(s.timeline.audio(480) == 1)
        s.playback(target, epoch)
        await asyncio.wait_for(s.drain(target, epoch), 0.1)
        await until(lambda: s.timeline.idle_next is not None)
        assert not np.any(s.timeline.audio(1920))
        assert s.timeline.submitted == s.timeline.sent == s.timeline.played == 480
        assert [op for op, _ in worker.calls].count("prepare") == 1
    finally:
        await s.close()
    calls = len(worker.calls)
    await asyncio.sleep(0.1)
    assert len(worker.calls) == calls


async def test_interrupt_discards_inflight_idle_and_keeps_motion_conditioning(tmp_path):
    s, worker = await make_session(tmp_path)
    try:
        worker.hold = asyncio.Event()
        s.viewer_ready.set()
        await until(lambda: len(worker.calls) == 2)
        before = s.timeline.frame.copy()
        await s.interrupt()
        assert np.array_equal(s.timeline.frame, before)
        worker.hold.set()
        await until(lambda: s.timeline.idle_next is not None)
        assert np.all(s.timeline.idle_next[0] == 3)  # The old epoch's result was discarded.
        assert [op for op, _ in worker.calls].count("prepare") == 1
        assert s.epoch == 1
    finally:
        await s.close()


async def test_speech_is_next_after_an_inflight_idle_block(tmp_path):
    s, worker = await make_session(tmp_path)
    try:
        worker.hold = asyncio.Event()
        s.viewer_ready.set()
        await until(lambda: len(worker.calls) == 2)
        await s.push(np.full(240, 1000, dtype="<i2").tobytes(), {"sample_rate": 24000})
        end = asyncio.create_task(s.end_turn())
        await asyncio.sleep(0)
        worker.hold.set()
        target, _ = await end
        assert target == 480
        assert np.any(worker.calls[2][1][0])  # The next inference contains speech.
        assert np.all(s.timeline.queue[0].video == 3)
    finally:
        await s.close()


async def test_socket_idle_is_paced_acknowledged_and_independent_of_audio():
    timeline = Timeline(np.zeros((8, 8, 3), dtype=np.uint8))
    timeline.put_idle(np.full((40, 8, 8, 3), 7, dtype=np.uint8), 25, 0)
    ready = asyncio.Event()
    ready.set()
    session = SimpleNamespace(timeline=timeline, viewer_ready=ready, epoch=0, closed=False)
    packets = []

    class Socket:
        async def send_json(self, event):
            packets.append(event)

        async def close(self, **kwargs):
            pass

    sink = SocketPlayback(session, Socket())
    task = asyncio.create_task(sink.send())
    try:
        await until(lambda: sink.idle_sent == 25)
        await asyncio.sleep(0.12)
        assert sink.idle_sent == 25
        assert timeline.sent == timeline.submitted == 0
        assert not any("audio" in packet for packet in packets)
        sink.acknowledge_idle(100000, -1)
        assert sink.idle_played == 0
        sink.acknowledge_idle(25, 0)
        await until(lambda: sink.idle_sent > 25)
        session.epoch = 1
        timeline.clear(1)
        sink.emit({"type": "interrupted", "epoch": 1})
        timeline.put_idle(np.full((2, 8, 8, 3), 8, dtype=np.uint8), 25, 1)
        await until(lambda: packets[-1].get("type") == "idle" and packets[-1]["epoch"] == 1)
        assert packets[-1]["sequence"] == 1
        await timeline.put(Segment(np.ones(120, np.int16), np.zeros((1, 8, 8, 3), np.uint8), 25, 1))
        await until(lambda: packets[-1].get("type") == "media")
        assert packets[-1]["samples"] == 120
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_idle_keeps_displaying_motion_while_inference_is_late():
    # A delayed GPU result must not stop idle video output or compound
    # transforms on the previous rendered frame.
    gradient = np.arange(512, dtype=np.uint8)[None, :, None]
    portrait = np.broadcast_to(gradient, (512, 512, 3)).copy()
    timeline = Timeline(portrait)
    pictures = []
    for _ in range(75):
        assert timeline.advance_idle(1920)
        pictures.append(timeline.frame.copy())
    assert np.array_equal(timeline.idle_source, portrait)
    assert not np.array_equal(pictures[0], pictures[-1])
    assert not np.array_equal(pictures[25], pictures[50])
    assert timeline.submitted == timeline.sent == timeline.played == 0


async def test_silent_model_clip_loops_without_repeated_diffusion_drift(tmp_path):
    session, worker = await make_session(tmp_path)
    try:
        session.viewer_ready.set()
        for _ in range(100):
            session.timeline.audio(1920)
            if session.metrics["idle_blocks"] == 2:
                break
            await asyncio.sleep(0.01)
        assert session.metrics["idle_blocks"] == 2
        assert session.timeline.idle_loop is not None
        for _ in range(150):
            session.timeline.audio(1920)
            await asyncio.sleep(0)
        assert [op for op, _ in worker.calls].count("infer") == 2
        assert session.timeline.idle_source is not None
        # Speech gets a fresh model result and clears the silent loop.
        await session.push(np.ones(480, dtype="<i2").tobytes(), {"sample_rate": 48000})
        await session.end_turn()
        assert session.timeline.idle_loop is None
        assert session.idle_generated_since_speech == 0
        assert [op for op, _ in worker.calls].count("infer") == 3
    finally:
        await session.close()
