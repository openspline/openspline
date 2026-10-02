import sys
from types import SimpleNamespace

import pytest
from openspline_server.config import Settings, WorkerConfig
from openspline_server.hardware import select_runtime, validate_cuda


def settings(*devices):
    return Settings(workers=[WorkerConfig(f"gpu-{d}", devices=(d,)) for d in devices])


@pytest.mark.parametrize(
    "capability,expected",
    [
        ("5.2", "cu126"),
        ("6.1", "cu126"),
        ("7.0", "cu126"),
        ("7.5", "cu126"),
        ("8.0", "cu126"),
        ("8.6", "cu126"),
        ("8.9", "cu126"),
        ("9.0", "cu126"),
        ("10.0", "cu128"),
        ("12.0", "cu128"),
        ("12.1", "cu128"),
    ],
)
def test_runtime_for_architecture(capability, expected):
    assert select_runtime(settings(0), [["0", "GPU-a", capability]]) == expected


def test_only_configured_gpus_and_visibility_remapping():
    inventory = [["0", "GPU-old", "7.0"], ["1", "GPU-new", "12.0"]]
    assert select_runtime(settings(0), inventory) == "cu126"
    assert select_runtime(settings(0), inventory, "1") == "cu128"
    assert select_runtime(settings(0), inventory, "GPU-new") == "cu128"
    assert select_runtime(settings(1), inventory, "1,0") == "cu126"
    with pytest.raises(ValueError, match="separate openspline"):
        select_runtime(settings(0, 1), inventory)
    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        select_runtime(settings(1), inventory, "1")
    with pytest.raises(ValueError, match="not found"):
        select_runtime(settings(2), inventory)


def test_mixed_modern_workers_use_one_compatible_runtime():
    assert (
        select_runtime(settings(0, 1), [["0", "GPU-a", "8.0"], ["1", "GPU-b", "12.0"]]) == "cu128"
    )


@pytest.mark.parametrize("capability", ["3.5", "13.0", "[N/A]"])
def test_unsupported_or_unknown_hardware_fails_clearly(capability):
    with pytest.raises(ValueError):
        select_runtime(settings(0), [["0", "GPU-a", capability]])


@pytest.mark.parametrize(
    "capability,dtype", [((7, 5), "fp32"), ((8, 0), "bf16"), ((12, 0), "bf16")]
)
def test_runtime_probe_precision_and_actionable_failure(monkeypatch, capability, dtype):
    calls = []

    class Tensor:
        def __matmul__(self, other):
            return self

        def __getitem__(self, key):
            return SimpleNamespace(item=lambda: 8)

    def ones(shape, **kwargs):
        calls.append(kwargs)
        return Tensor()

    torch = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        ones=ones,
        __version__="2.7.1",
        version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            set_device=lambda d: calls.append(d),
            get_device_name=lambda d: "Test GPU",
            get_device_capability=lambda d: capability,
            synchronize=lambda d: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert validate_cuda(1) == dtype
    assert calls == [1, {"device": "cuda:1", "dtype": dtype}]

    def fail(*args, **kwargs):
        raise RuntimeError("no kernel image is available")

    torch.ones = fail
    with pytest.raises(RuntimeError, match="Rerun install.sh"):
        validate_cuda(1)


def test_cpu_install_has_actionable_error(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    with pytest.raises(RuntimeError, match="NVIDIA driver"):
        validate_cuda()
