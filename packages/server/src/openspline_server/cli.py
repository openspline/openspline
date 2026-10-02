import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .config import Settings, WorkerConfig

MODEL_REPO = "Soul-AILab/SoulX-FlashHead-1_3B"
AUDIO_REPO = "facebook/wav2vec2-base-960h"
MODEL_REVISION = "59119b6c681230c3eeee157e224ae1941746711e"
AUDIO_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"


def download(args):
    from huggingface_hub import HfApi, snapshot_download

    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    profiles = ["low", "high"] if args.quality == "all" else [args.quality]
    patterns = ["*.json"]
    if "low" in profiles:
        patterns += ["Model_Lite/*", "VAE_LTX/*"]
    if "high" in profiles:
        patterns += ["Model_Pro/*", "VAE_Wan/*"]
    manifest = {}
    for name, repo, allow in [
        ("avatar", MODEL_REPO, patterns),
        (
            "audio",
            AUDIO_REPO,
            ["config.json", "preprocessor_config.json", "model.safetensors", "pytorch_model.bin"],
        ),
    ]:
        revision = (
            HfApi()
            .model_info(
                repo,
                revision=(args.revision or MODEL_REVISION) if name == "avatar" else AUDIO_REVISION,
            )
            .sha
        )
        snapshot_download(repo, revision=revision, allow_patterns=allow, local_dir=root / name)
        manifest[name] = {"repository": repo, "revision": revision}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Models ready in {root}. Revisions recorded in manifest.json.")


def doctor(args):
    result = {
        "python": os.sys.version.split()[0],
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "modules": {
            n: importlib.util.find_spec(n) is not None
            for n in ["torch", "diffusers", "transformers", "aiortc"]
        },
    }
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        result["gpus"] = proc.stdout.strip().splitlines()
    else:
        result["gpus"] = []
    settings = Settings.from_file(args.config)
    result["models_present"] = all(
        Path(p).is_dir() for p in [settings.model_dir, settings.audio_model_dir]
    )
    print(json.dumps(result, indent=2))


async def benchmark(args):
    import numpy as np

    from .worker import Worker

    settings = Settings(
        workers=[WorkerConfig("benchmark", args.quality, tuple(args.devices))], backend=args.backend
    )
    worker = Worker(settings.workers[0], settings)
    started = time.perf_counter()
    await worker.start()
    load = time.perf_counter() - started
    if not worker.ready:
        raise RuntimeError(worker.error)
    try:
        prep = time.perf_counter()
        await worker.call("prepare", str(Path(args.portrait).resolve()), 0)
        prep = time.perf_counter() - prep
        history = np.zeros(worker.info["cache_samples"], dtype=np.float32)
        samples = worker.info["chunk_samples"]
        dur = samples / 16000
        durations = []
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline or not durations:
            # Continuous voiced fixture, rather than claiming a speech quality evaluation.
            tone = (0.1 * np.sin(2 * np.pi * 180 * np.arange(samples) / 16000)).astype(np.float32)
            history = np.concatenate([history, tone])[-len(history) :]
            frames, elapsed = await worker.call("infer", history)
            durations.append(elapsed)
            if args.paced:
                await asyncio.sleep(max(0, dur - elapsed))
        report = {
            "backend": args.backend,
            "quality": args.quality,
            "worker_count": 1,
            "active_sessions_per_worker": 1,
            "load_seconds": load,
            "prepare_seconds": prep,
            "block_audio_seconds": dur,
            "blocks": len(durations),
            "first_block_seconds": durations[0],
            "inference_p50_seconds": float(np.median(durations)),
            "inference_p95_seconds": float(np.percentile(durations, 95)),
            "generated_fps": len(frames) / float(np.mean(durations)),
            "realtime_factor": dur / float(np.mean(durations)),
            "scope": "engine only; excludes transport and browser playback",
        }
        report.update({k: v for k, v in worker.info.items() if k.startswith("gpu_")})
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        await worker.stop()


def main():
    parser = argparse.ArgumentParser(prog="openspline")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--config")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--backend", choices=["gpu", "test"], default="gpu")
    serve.add_argument("--dev", action="store_true")
    models = sub.add_parser("models")
    models.add_argument("--directory", default="models")
    models.add_argument("--quality", choices=["low", "high", "all"], default="low")
    models.add_argument("--revision")
    check = sub.add_parser("doctor")
    check.add_argument("--config")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--portrait", required=True)
    bench.add_argument("--quality", choices=["low", "high"], default="low")
    bench.add_argument("--devices", nargs="+", type=int, default=[0])
    bench.add_argument("--seconds", type=float, default=60)
    bench.add_argument("--paced", action="store_true")
    bench.add_argument("--output")
    bench.add_argument("--backend", choices=["gpu", "test"], default="gpu")
    args = parser.parse_args()
    if args.command == "models":
        download(args)
    elif args.command == "doctor":
        doctor(args)
    elif args.command == "benchmark":
        asyncio.run(benchmark(args))
    else:
        import uvicorn

        from .app import create_app

        settings = Settings.from_file(args.config, backend=args.backend)
        if args.backend == "test":
            print("TEST BACKEND: static portraits, no avatar inference.")
        uvicorn.run(create_app(settings), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
