"""Select a pinned NVIDIA runtime and check it before loading model weights."""

import argparse
import csv
import os
import subprocess
from pathlib import Path


def list_gpus(visible=None):
    """List allowed CUDA ordinals without initializing CUDA in the API process."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = [[column.strip() for column in row] for row in csv.reader(result.stdout.splitlines())]
    if visible is None:
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
    selectors = visible.split(",") if visible is not None else [row[0] for row in rows]
    devices = []
    for ordinal, selector in enumerate(selectors):
        matches = [
            r
            for r in rows
            if r[0] == selector.strip()
            or (selector.strip().startswith("GPU-") and r[1].startswith(selector.strip()))
        ]
        if len(matches) != 1:
            continue
        index, uuid, name, total, free = matches[0]
        devices.append(
            dict(
                id=ordinal,
                physical_id=int(index),
                uuid=uuid,
                name=name,
                memory_total_mb=int(total),
                memory_free_mb=int(free),
            )
        )
    return devices


def select_runtime(settings, inventory, visible=None):
    """Inventory is from nvidia-smi; worker IDs refer to CUDA-visible ordinals."""
    devices = settings.gpu_devices
    selectors = visible.split(",") if visible is not None else None
    selected = []
    for device in devices:
        if selectors is not None and device >= len(selectors):
            raise ValueError(f"Worker GPU {device} is outside CUDA_VISIBLE_DEVICES")
        selector = selectors[device].strip() if selectors is not None else str(device)
        matches = [
            row
            for row in inventory
            if row[0] == selector or (selector.startswith("GPU-") and row[1].startswith(selector))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Configured GPU {selector!r} was not found unambiguously by nvidia-smi"
            )
        row = matches[0]
        try:
            capability = tuple(int(part) for part in row[2].split("."))
            if len(capability) != 2:
                raise ValueError()
        except ValueError as exc:
            raise ValueError(
                "Driver cannot report compute capability; update the NVIDIA driver"
            ) from exc
        if capability < (5, 0) or capability >= (13, 0):
            raise ValueError(f"GPU compute capability {row[2]} has no supported pinned runtime")
        selected.append(capability)
    runtime = "cu128" if max(selected) >= (10, 0) else "cu126"
    if runtime == "cu128" and min(selected) < (7, 5):
        raise ValueError(
            "Blackwell and pre-Turing workers need different PyTorch builds; "
            "run them in separate openspline installations"
        )
    return runtime


def validate_cuda(device=0):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and rerun install.sh.")
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    # BF16 hardware starts with Ampere. FP32 avoids FP16 overflow on older cards.
    dtype = torch.bfloat16 if capability >= (8, 0) else torch.float32
    try:
        x = torch.ones((8, 8), device=f"cuda:{device}", dtype=dtype)
        result = x @ x
        torch.cuda.synchronize(device)
        if result[0, 0].item() != 8:
            raise RuntimeError("CUDA matrix multiplication returned an incorrect result")
    except RuntimeError as exc:
        build = "cu128" if capability >= (10, 0) else "cu126"
        raise RuntimeError(
            f"{name} (sm_{capability[0]}{capability[1]}) cannot run this PyTorch "
            f"build ({torch.__version__}, CUDA {torch.version.cuda}). "
            f"Rerun install.sh to select {build}, and check your NVIDIA driver. {exc}"
        ) from exc
    return dtype


def validate_distributed():
    try:
        from xfuser.core.distributed import initialize_model_parallel
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
        from yunchang.kernels import AttnType

        assert initialize_model_parallel and xFuserLongContextAttention and AttnType.TORCH_FLASH
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "The multi-GPU runtime is missing or incompatible. Rerun install.sh "
            "with OPENSPLINE_GPUS set to your GPU group. " + str(exc)
        ) from exc


def main():
    from .native import configure, environment

    environment(Path.cwd())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        settings = configure(os.getenv("OPENSPLINE_CONFIG", "workers.yaml"))
        if args.check:
            if len(settings.demo_high_devices or ()) > 1 or any(
                len(w.devices) > 1 for w in settings.workers
            ):
                validate_distributed()
            for device in settings.gpu_devices:
                dtype = validate_cuda(device)
                print(f"GPU {device}: CUDA verified, inference precision {dtype}")
        else:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            inventory = [
                [column.strip() for column in row] for row in csv.reader(result.stdout.splitlines())
            ]
            print(select_runtime(settings, inventory, os.getenv("CUDA_VISIBLE_DEVICES")))
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
