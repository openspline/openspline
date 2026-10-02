"""Native launcher: installed paths, GPU selection, and model preparation."""

import argparse
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv, set_key

from .config import Settings, WorkerConfig


def environment(root):
    load_dotenv(root / ".env", override=False)
    # Match physical GPU indices reported by nvidia-smi, including mixed GPU hosts.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for key, relative in {
        "OPENSPLINE_MODEL_DIR": "models/avatar",
        "OPENSPLINE_AUDIO_MODEL_DIR": "models/audio",
        "OPENSPLINE_RUNTIME_DIR": "runtime",
        "HF_HOME": ".cache/huggingface",
        "TORCH_HOME": ".cache/torch",
        "XDG_CACHE_HOME": ".cache",
        "NUMBA_CACHE_DIR": ".cache/numba",
        "TMPDIR": "runtime/tmp",
    }.items():
        os.environ.setdefault(key, str(root / relative))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = os.environ["TMPDIR"]


def configure(config):
    settings = Settings.from_file(config)
    gpu = os.environ.get("OPENSPLINE_GPU")
    quality = os.environ.get("OPENSPLINE_QUALITY")
    if gpu is not None:
        if not gpu.isascii() or not gpu.isdecimal():
            raise ValueError("OPENSPLINE_GPU must be one physical GPU index, such as 0 or 1")
        if len(settings.workers) != 1 or len(settings.workers[0].devices) != 1:
            raise ValueError(
                "For multiple GPUs or workers, configure workers.yaml without OPENSPLINE_GPU"
            )
        # Select the physical GPU before any CUDA imports; the worker sees logical GPU 0.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
        worker = settings.workers[0]
        settings.workers = [WorkerConfig(worker.id, worker.quality, (0,))]
    if quality is not None:
        if quality not in {"low", "high"} or len(settings.workers) != 1:
            raise ValueError("OPENSPLINE_QUALITY must be low or high for a single worker")
        worker = settings.workers[0]
        settings.workers = [WorkerConfig(worker.id, quality, worker.devices)]
    settings.__post_init__()
    return settings


def prepare(settings):
    from .cli import download

    # The downloader writes avatar/ and audio/ under one directory.
    avatar = Path(settings.model_dir)
    audio = Path(settings.audio_model_dir)
    if avatar.name != "avatar" or audio.name != "audio" or avatar.parent != audio.parent:
        raise ValueError(
            "Native model download requires sibling model directories named avatar and audio"
        )
    qualities = {w.quality for w in settings.workers}
    download(
        SimpleNamespace(
            directory=str(avatar.parent),
            quality="all" if len(qualities) > 1 else qualities.pop(),
            revision=None,
        )
    )


def main():
    environment(Path.cwd())
    parser = argparse.ArgumentParser(description="Run openspline directly on this server")
    parser.add_argument("--config", default=os.getenv("OPENSPLINE_CONFIG", "workers.yaml"))
    parser.add_argument("--host", default=os.getenv("OPENSPLINE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=os.getenv("OPENSPLINE_PORT", "7860"))
    parser.add_argument(
        "--prepare", action="store_true", help="Download models for the configured workers"
    )
    parser.add_argument(
        "--save-selection",
        action="store_true",
        help="Remember GPU and quality overrides in .env after preparation",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    os.environ.setdefault("OPENSPLINE_PUBLIC_URL", f"http://localhost:{args.port}")
    try:
        settings = configure(args.config)
        if args.prepare:
            prepare(settings)
            if args.save_selection:
                for key in ("OPENSPLINE_GPU", "OPENSPLINE_QUALITY"):
                    if key in os.environ:
                        set_key(".env", key, os.environ[key])
            return
    except ValueError as exc:
        parser.error(str(exc))
    import uvicorn

    from .app import create_app

    print(f"Starting openspline on port {args.port}; press Ctrl-C to stop.", flush=True)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
