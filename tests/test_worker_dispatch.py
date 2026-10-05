import sys
from types import ModuleType, SimpleNamespace

from openspline_server.worker import _process_main


def test_gpu_group_launches_one_rank_per_remapped_device(tmp_path, monkeypatch):
    """Verify process dispatch without pretending CPU workers validate GPU inference."""
    import os

    calls = []
    torch = ModuleType("torch")
    torch.multiprocessing = ModuleType("torch.multiprocessing")
    torch.multiprocessing.spawn = lambda target, **kwargs: calls.append((target, kwargs))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", torch.multiprocessing)
    monkeypatch.setattr(os, "setsid", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4,6,8")
    errors = []
    pipe = SimpleNamespace(send=errors.append, close=lambda: None)
    settings = {"backend": "gpu", "devices": (1, 3), "runtime_dir": str(tmp_path)}
    _process_main(settings, pipe)
    assert not errors
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,8"
    assert len(calls) == 1 and calls[0][1]["nprocs"] == 2
    assert calls[0][1]["args"][0] == 2
    assert (tmp_path / "locks" / "gpu-4.lock").exists()
    assert (tmp_path / "locks" / "gpu-8.lock").exists()
