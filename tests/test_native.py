import os
import tempfile
from pathlib import Path

import pytest
from openspline_server.native import configure, environment, main, prepare


@pytest.fixture(autouse=True)
def native_env(monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for key in list(os.environ):
        if key.startswith("OPENSPLINE_") or key in {
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "TORCH_HOME",
            "XDG_CACHE_HOME",
            "NUMBA_CACHE_DIR",
            "TMPDIR",
        }:
            monkeypatch.delenv(key)
    # Restore all changes made directly by the launcher, too.
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)


def workers(tmp_path, text="workers:\n  - {id: main, quality: low, devices: [0]}\n"):
    path = tmp_path / "workers.yaml"
    path.write_text(text)
    return path


def test_physical_gpu_remaps_to_single_logical_device(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OPENSPLINE_GPU", "1")
    monkeypatch.setenv("OPENSPLINE_QUALITY", "high")
    settings = configure(workers(tmp_path))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert settings.workers[0].devices == (0,)
    assert settings.workers[0].quality == "high"


def test_existing_multiple_workers_are_preserved(tmp_path):
    path = workers(
        tmp_path,
        "workers:\n  - {id: a, quality: low, devices: [0]}\n  - {id: b, quality: high, devices: [1]}\n",
    )
    settings = configure(path)
    assert [(w.id, w.devices) for w in settings.workers] == [("a", (0,)), ("b", (1,))]
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize("gpu", ["-1", "0,1", "all", "", "abc"])
def test_invalid_gpu_rejected(tmp_path, monkeypatch, gpu):
    monkeypatch.setenv("OPENSPLINE_GPU", gpu)
    with pytest.raises(ValueError, match="physical GPU"):
        configure(workers(tmp_path))


def test_single_gpu_override_cannot_collapse_multiple_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSPLINE_GPU", "1")
    with pytest.raises(ValueError, match="multiple GPUs or workers"):
        configure(
            workers(tmp_path, "workers:\n  - {id: a, devices: [0]}\n  - {id: b, devices: [1]}\n")
        )


def test_mixed_workers_download_both_profiles(tmp_path, monkeypatch):
    path = workers(
        tmp_path,
        "workers:\n  - {id: a, quality: low, devices: [0]}\n  - {id: b, quality: high, devices: [1]}\n",
    )
    environment(tmp_path)
    downloads = []
    monkeypatch.setattr("openspline_server.cli.download", downloads.append)
    prepare(configure(path))
    assert downloads[0].quality == "all"
    assert downloads[0].directory == str(tmp_path / "models")


def test_installed_env_does_not_override_explicit_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENSPLINE_GPU=0\nOPENSPLINE_PORT=7862\n")
    monkeypatch.setenv("OPENSPLINE_GPU", "1")
    environment(tmp_path)
    assert os.environ["OPENSPLINE_GPU"] == "1"
    assert os.environ["OPENSPLINE_PORT"] == "7862"
    assert Path(os.environ["TMPDIR"]).is_dir()
    assert os.environ["HF_HOME"] == str(tmp_path / ".cache/huggingface")


def test_prepare_persists_gpu_without_replacing_credentials_or_workers(tmp_path, monkeypatch):
    path = workers(tmp_path)
    original = path.read_text()
    (tmp_path / ".env").write_text("OPENAI_API_KEY=provider-secret\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENSPLINE_GPU", "1")
    monkeypatch.setattr("sys.argv", ["native", "--prepare", "--save-selection"])
    monkeypatch.setattr("openspline_server.cli.download", lambda args: None)
    main()
    assert "OPENSPLINE_GPU='1'" in (tmp_path / ".env").read_text()
    assert "OPENAI_API_KEY=provider-secret" in (tmp_path / ".env").read_text()
    assert path.read_text() == original


def test_prepare_failure_does_not_save_selection(tmp_path, monkeypatch):
    workers(tmp_path)
    (tmp_path / ".env").write_text("OPENSPLINE_GPU=0\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENSPLINE_GPU", "1")
    monkeypatch.setattr("sys.argv", ["native", "--prepare", "--save-selection"])

    def fail(args):
        raise RuntimeError("download failed")

    monkeypatch.setattr("openspline_server.cli.download", fail)
    with pytest.raises(RuntimeError, match="download failed"):
        main()
    assert (tmp_path / ".env").read_text() == "OPENSPLINE_GPU=0\n"
