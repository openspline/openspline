"""One process group, one model, one exclusive session per GPU allocation."""

from __future__ import annotations

import asyncio
import fcntl
import multiprocessing as mp
import os
import signal
import socket
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


class WorkerFailure(RuntimeError):
    pass


class CapacityError(RuntimeError):
    pass


class TestEngine:
    def __init__(self):
        self.fps, self.chunk_samples, self.cache_samples = 25, 15360, 128000
        self.image = None

    def prepare(self, image, seed=0):
        from PIL import Image

        self.image = np.asarray(Image.open(image).convert("RGB").resize((512, 512)))

    def infer(self, audio):
        return np.repeat(self.image[None], 24, axis=0)


class GPUEngine:
    def __init__(self, settings, world_size):
        from .hardware import validate_cuda

        dtype = validate_cuda(int(os.environ.get("LOCAL_RANK", "0")))
        from ._vendor.flash_head import inference as inf

        self.inf = inf
        self.pipeline = inf.get_pipeline(
            world_size,
            settings["model_dir"],
            "lite" if settings["quality"] == "low" else "pro",
            settings["audio_model_dir"],
            param_dtype=dtype,
        )
        self.params = inf.get_infer_params()
        self.fps = self.params["tgt_fps"]
        self.motion = self.params["motion_frames_num"]
        self.chunk_samples = (self.params["frame_num"] - self.motion) * 16000 // self.fps
        self.cache_samples = self.params["cached_audio_duration"] * 16000
        self.start = self.params["cached_audio_duration"] * self.fps - self.params["frame_num"]
        self.end = self.params["cached_audio_duration"] * self.fps

    def prepare(self, image, seed=0):
        import random

        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.inf.get_base_data(self.pipeline, image, seed, False)

    def infer(self, audio):
        embedding = self.inf.get_audio_embedding(self.pipeline, audio, self.start, self.end)
        frames = self.inf.run_pipeline(self.pipeline, embedding)[self.motion :]
        return frames.cpu().numpy().astype(np.uint8, copy=False)


def _rank_main(rank, world_size, settings, conn, port):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    try:
        engine = TestEngine() if settings["backend"] == "test" else GPUEngine(settings, world_size)
        dist = None
        if world_size > 1:
            import torch.distributed as dist
        if rank == 0:
            conn.send(
                (
                    "ready",
                    dict(
                        fps=engine.fps,
                        chunk_samples=engine.chunk_samples,
                        cache_samples=engine.cache_samples,
                    ),
                )
            )
        while True:
            command = conn.recv() if rank == 0 else None
            if dist:
                box = [command]
                dist.broadcast_object_list(box, src=0)
                command = box[0]
            op, args = command
            if op == "close":
                break
            if op == "prepare":
                result = engine.prepare(*args)
            elif op == "infer":
                started = time.perf_counter()
                frames = engine.infer(*args)
                result = {"frames": frames, "seconds": time.perf_counter() - started}
                if settings["backend"] == "gpu":
                    import torch

                    result["gpu_memory_bytes"] = torch.cuda.memory_reserved()
                    result["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            elif op == "reset":
                if hasattr(engine, "pipeline"):
                    for name in (
                        "cond_image_dict",
                        "cond_image_tensor_dict",
                        "ref_img_latent_dict",
                        "ref_img_latent",
                        "latent_motion_frames",
                        "original_color_reference",
                        "generator",
                    ):
                        if hasattr(engine.pipeline, name):
                            delattr(engine.pipeline, name)
                result = None
            else:
                raise ValueError("Unknown worker operation")
            if rank == 0:
                conn.send(("ok", result))
        if dist:
            dist.destroy_process_group()
    except BaseException as exc:
        if rank == 0:
            try:
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
            except (BrokenPipeError, OSError):
                pass
        raise


def _process_main(settings, conn):
    os.setsid()
    locks = []
    try:
        devices = settings["devices"]
        inherited = os.getenv("CUDA_VISIBLE_DEVICES")
        if inherited:
            visible = inherited.split(",")
            devices = [visible[d] for d in devices]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        if settings["backend"] == "gpu":
            lock_dir = Path(
                os.getenv("OPENSPLINE_GPU_LOCK_DIR", str(Path(settings["runtime_dir"]) / "locks"))
            )
            lock_dir.mkdir(parents=True, exist_ok=True)
            for device in devices:
                fd = (lock_dir / f"gpu-{device}.lock").open("w")
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locks.append(fd)
        world = len(devices)
        if world == 1 or settings["backend"] == "test":
            _rank_main(0, 1, settings, conn, 0)
        else:
            import torch.multiprocessing

            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            torch.multiprocessing.spawn(
                _rank_main, args=(world, settings, conn, port), nprocs=world, join=True
            )
    except BaseException as exc:
        try:
            conn.send(("error", str(exc)))
        except (OSError, BrokenPipeError):
            pass
    finally:
        for fd in locks:
            fd.close()
        conn.close()


class Worker:
    def __init__(self, config, settings):
        self.config, self.settings = config, settings
        self.process = self.conn = None
        self.ready = False
        self.owner = None
        self.info = {}
        self.error = None
        self.lock = asyncio.Lock()

    async def start(self):
        self.ready = False
        parent, child = mp.get_context("spawn").Pipe()
        payload = dict(
            backend=self.settings.backend,
            model_dir=self.settings.model_dir,
            audio_model_dir=self.settings.audio_model_dir,
            runtime_dir=self.settings.runtime_dir,
            quality=self.config.quality,
            devices=self.config.devices,
        )
        self.process = mp.get_context("spawn").Process(
            target=_process_main, args=(payload, child), daemon=False
        )
        self.process.start()
        child.close()
        self.conn = parent
        receive = asyncio.create_task(
            asyncio.to_thread(self._receive, self.settings.startup_timeout)
        )
        try:
            kind, value = await asyncio.shield(receive)
            if kind != "ready":
                raise WorkerFailure(value)
            self.info = value
            self.ready = True
            self.error = None
        except asyncio.CancelledError:
            await self.stop()
            await asyncio.gather(receive, return_exceptions=True)
            raise
        except Exception as exc:
            self.error = str(exc)
            await self.stop()

    def _receive(self, timeout):
        if not self.conn.poll(timeout):
            raise WorkerFailure("Worker timed out")
        try:
            return self.conn.recv()
        except EOFError as exc:
            raise WorkerFailure("Worker exited") from exc

    async def call(self, op, *args):
        async with self.lock:
            if not self.ready or not self.process.is_alive():
                self.ready = False
                raise WorkerFailure("Worker unavailable")
            try:
                await asyncio.to_thread(self.conn.send, (op, args))
                receive = asyncio.create_task(
                    asyncio.to_thread(self._receive, self.settings.inference_timeout)
                )
                try:
                    kind, result = await asyncio.shield(receive)
                except asyncio.CancelledError:
                    try:
                        await receive
                    except Exception:
                        self.ready = False
                    raise
                if kind != "ok":
                    raise WorkerFailure(result)
                if op == "infer" and isinstance(result, dict):
                    self.info.update({k: v for k, v in result.items() if k.startswith("gpu_")})
                    return result["frames"], result["seconds"]
                return result
            except Exception:
                self.ready = False
                raise

    async def stop(self):
        self.ready = False
        if self.process:
            if self.process.is_alive():
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                await asyncio.to_thread(self.process.join, 3)
                if self.process.is_alive():
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await asyncio.to_thread(self.process.join, 3)
            self.process.close()
            self.process = None
        if self.conn:
            self.conn.close()
            self.conn = None


class WorkerPool:
    def __init__(self, settings):
        self.settings = settings
        self.workers = [Worker(c, settings) for c in settings.workers]
        self.lock = asyncio.Lock()

    async def start(self):
        await asyncio.gather(*(w.start() for w in self.workers))

    async def recover_idle(self):
        # Restart in the background so one slow model load cannot stall session expiry.
        async def recover(worker):
            try:
                await worker.stop()
                await worker.start()
            finally:
                worker.recovering = None

        for worker in self.workers:
            if (
                worker.owner is None
                and not getattr(worker, "recovering", None)
                and (not worker.ready or not worker.process or not worker.process.is_alive())
            ):
                worker.ready = False
                worker.recovering = asyncio.create_task(recover(worker))

    async def acquire(self, quality, owner):
        async with self.lock:
            if not any(w.config.quality == quality for w in self.workers):
                raise ValueError("Requested quality is not configured")
            for w in self.workers:
                if w.ready and w.owner is None and w.config.quality == quality:
                    w.owner = owner
                    return w
            raise CapacityError("All workers for this quality are occupied or unavailable")

    async def reserve_quality_switch(self, quality, owner):
        """Reserve the default single-GPU worker; never displace an active session."""
        if not isinstance(quality, str) or quality not in {"low", "high"}:
            raise ValueError("quality must be low or high")
        async with self.lock:
            for worker in self.workers:
                if worker.config.quality == quality and worker.ready and worker.owner is None:
                    return None
            if len(self.workers) != 1 or len(self.workers[0].config.devices) != 1:
                if any(w.config.quality == quality for w in self.workers):
                    raise CapacityError("All workers for this quality are occupied or unavailable")
                raise ValueError("Configure a worker for this quality in a multi-worker deployment")
            worker = self.workers[0]
            if worker.owner is not None or getattr(worker, "recovering", None):
                raise CapacityError(
                    "GPU is busy. End its current session before switching quality."
                )
            worker.owner = owner
            return worker

    async def switch_quality(self, worker, quality, owner, progress):
        from .quality import prepare_quality

        previous = worker.config
        unloaded = False
        try:
            progress("checking")
            await prepare_quality(self.settings, quality)
            progress("unloading")
            unloaded = True
            await worker.stop()
            worker.config = replace(previous, quality=quality)
            progress("loading")
            await worker.start()
            if not worker.ready:
                raise WorkerFailure(worker.error or "Model could not load")
        except asyncio.CancelledError:
            await worker.stop()
            worker.config = previous
            raise
        except Exception as exc:
            if unloaded:
                progress("restoring")
                await worker.stop()
                worker.config = previous
                await worker.start()
            recovery = (
                f" {previous.quality.capitalize()} quality is still available."
                if worker.ready
                else " The worker is unavailable; check the server logs."
            )
            raise WorkerFailure(f"Could not prepare {quality} quality: {exc}.{recovery}") from exc
        finally:
            async with self.lock:
                if worker.owner == owner:
                    worker.owner = None

    async def release(self, worker, owner):
        if worker.owner != owner:
            return
        try:
            if worker.ready:
                await worker.call("reset")
        except WorkerFailure:
            pass
        if not worker.ready:
            await worker.stop()
            await worker.start()
        async with self.lock:
            if worker.owner == owner:
                worker.owner = None

    async def close(self):
        tasks = [w.recovering for w in self.workers if getattr(w, "recovering", None)]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(w.stop() for w in self.workers))
