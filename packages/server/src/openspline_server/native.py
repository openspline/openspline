"""Native launcher: installed paths, GPU selection, and model preparation."""

import argparse
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv, set_key, unset_key

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


def configure(config, *, demo=False):
    original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    settings = Settings.from_file(config)
    gpu = os.environ.get("OPENSPLINE_GPU")
    gpus = os.environ.get("OPENSPLINE_GPUS")
    quality = os.environ.get("OPENSPLINE_QUALITY")
    if gpus is not None:
        selection = [part.strip() for part in gpus.split(",")]
        if not selection or any(not part.isascii() or not part.isdecimal() for part in selection):
            raise ValueError("OPENSPLINE_GPUS must list physical GPU indices, such as 0,1")
        physical = [int(part) for part in selection]
        if len(set(physical)) != len(physical):
            raise ValueError("OPENSPLINE_GPUS must not repeat a GPU")
        if len(settings.workers) != 1:
            raise ValueError("For multiple workers, configure workers.yaml without OPENSPLINE_GPUS")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, physical))
        group = tuple(range(len(physical)))
        worker = settings.workers[0]
        selected_quality = quality or worker.quality
        settings.workers = [
            WorkerConfig(
                worker.id, selected_quality, group if selected_quality == "high" else group[:1]
            )
        ]
        settings.demo_high_devices = group
    elif gpu is not None:
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
        settings.demo_high_devices = (0,)
    if quality is not None:
        if quality not in {"low", "high"} or len(settings.workers) != 1:
            raise ValueError("OPENSPLINE_QUALITY must be low or high for a single worker")
        worker = settings.workers[0]
        group = settings.demo_high_devices or worker.devices
        if len(worker.devices) > 1:
            settings.demo_high_devices = group
        settings.workers = [
            WorkerConfig(worker.id, quality, group if quality == "high" else group[:1])
        ]
    if demo and (gpus is not None or gpu is not None):
        # OPENSPLINE_GPUS chooses the initial allocation. Keep the launcher's
        # CUDA visibility boundary so the UI can select other allowed GPUs.
        selected = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if original_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            remap = [int(device) for device in selected]
        else:
            from .hardware import list_gpus

            os.environ["CUDA_VISIBLE_DEVICES"] = original_visible
            available = {str(d["physical_id"]): d["id"] for d in list_gpus()}
            if any(device not in available for device in selected):
                raise ValueError("Selected GPUs are outside CUDA_VISIBLE_DEVICES")
            remap = [available[device] for device in selected]
        worker = settings.workers[0]
        settings.workers = [
            WorkerConfig(worker.id, worker.quality, tuple(remap[d] for d in worker.devices))
        ]
        settings.demo_high_devices = tuple(remap[d] for d in settings.demo_high_devices)
    settings.__post_init__()
    return settings


def save_demo_selection(path, physical_devices, quality):
    """Replace just the selection, atomically preserving credentials and other settings."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".openspline-selection-", dir=path.parent)
    os.close(fd)
    try:
        copy = Path(temporary)
        copy.write_text(path.read_text() if path.exists() else "")
        if path.exists():
            copy.chmod(path.stat().st_mode & 0o777)
        unset_key(copy, "OPENSPLINE_GPU")
        set_key(copy, "OPENSPLINE_GPUS", ",".join(map(str, physical_devices)))
        set_key(copy, "OPENSPLINE_QUALITY", quality)
        os.replace(copy, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


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
    if len(settings.workers) == 1:
        # Cache every quality offered by the demo before launching it.
        qualities = {"low", "high"}
    print(f"Downloading models before startup: {', '.join(sorted(qualities))}.", flush=True)
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
        "--prepare",
        action="store_true",
        help="Download models before startup (both qualities for a single demo worker)",
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
        settings = configure(args.config, demo=True)
        if args.prepare:
            prepare(settings)
            if args.save_selection:
                if "OPENSPLINE_GPUS" in os.environ:
                    unset_key(".env", "OPENSPLINE_GPU")
                for key in ("OPENSPLINE_GPUS", "OPENSPLINE_GPU", "OPENSPLINE_QUALITY"):
                    if key == "OPENSPLINE_GPU" and "OPENSPLINE_GPUS" in os.environ:
                        continue
                    if key in os.environ:
                        set_key(".env", key, os.environ[key])
            return
    except ValueError as exc:
        parser.error(str(exc))
    import uvicorn

    from .app import create_app

    selection_path = Path.cwd() / ".env"
    settings.save_demo_selection = lambda devices, quality: save_demo_selection(
        selection_path, devices, quality
    )
    print(f"Starting openspline on port {args.port}; press Ctrl-C to stop.", flush=True)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
