import os
import tempfile
from pathlib import Path

import pytest
from openspline_server.native import configure, environment, main, prepare, save_demo_selection


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


@pytest.mark.parametrize("quality,devices", [("low", (0,)), ("high", (0, 1))])
def test_gpu_pair_remaps_and_overrides_legacy_single_gpu(tmp_path, monkeypatch, quality, devices):
    monkeypatch.setenv("OPENSPLINE_GPU", "0")
    monkeypatch.setenv("OPENSPLINE_GPUS", "2, 4")
    monkeypatch.setenv("OPENSPLINE_QUALITY", quality)
    settings = configure(workers(tmp_path))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,4"
    assert settings.workers[0].devices == devices
    assert settings.demo_high_devices == (0, 1)
    assert settings.gpu_devices == [0, 1]


def test_native_demo_keeps_other_gpus_selectable(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSPLINE_GPU", "1")
    settings = configure(workers(tmp_path), demo=True)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert settings.workers[0].devices == (1,)
    assert settings.demo_high_devices == (1,)


def test_native_demo_preserves_external_visibility_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-second,GPU-first")
    monkeypatch.setenv("OPENSPLINE_GPUS", "0,1")
    monkeypatch.setenv("OPENSPLINE_QUALITY", "high")
    monkeypatch.setattr(
        "openspline_server.hardware.list_gpus",
        lambda: [
            {"id": 0, "physical_id": 1},
            {"id": 1, "physical_id": 0},
        ],
    )
    settings = configure(workers(tmp_path), demo=True)
    assert settings.workers[0].devices == (1, 0)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-second,GPU-first"
    monkeypatch.setenv("OPENSPLINE_GPUS", "2")
    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        configure(workers(tmp_path), demo=True)


def test_demo_selection_survives_restart_without_replacing_other_settings(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("OPENSPLINE_GPU=0\nOPENAI_API_KEY=keep-existing\n")
    save_demo_selection(path, [1, 0], "high")
    assert "OPENSPLINE_GPU=" not in path.read_text()
    assert "OPENAI_API_KEY=keep-existing" in path.read_text()
    assert not list(tmp_path.glob(".openspline-selection-*"))
    environment(tmp_path)
    settings = configure(workers(tmp_path), demo=True)
    assert settings.workers[0].devices == (1, 0)
    assert settings.workers[0].quality == "high"


@pytest.mark.parametrize("selection", ["", "0,", "0,-1", "0,0", "0,00", "all", "0,x"])
def test_invalid_gpu_groups_rejected(tmp_path, monkeypatch, selection):
    monkeypatch.setenv("OPENSPLINE_GPUS", selection)
    with pytest.raises(ValueError, match="OPENSPLINE_GPUS"):
        configure(workers(tmp_path))


def test_gpu_group_cannot_replace_multiple_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSPLINE_GPUS", "0,1")
    with pytest.raises(ValueError, match="multiple workers"):
        configure(workers(tmp_path, "workers: [{id: a, devices: [0]}, {id: b, devices: [1]}]"))


def test_preparation_persists_gpu_pair_and_removes_legacy_pin(tmp_path, monkeypatch):
    workers(tmp_path)
    (tmp_path / ".env").write_text("OPENSPLINE_GPU=0\nOPENAI_API_KEY=keep\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENSPLINE_GPUS", "0,1")
    monkeypatch.setattr("sys.argv", ["native", "--prepare", "--save-selection"])
    downloads = []
    monkeypatch.setattr("openspline_server.cli.download", downloads.append)
    main()
    assert downloads[0].quality == "all"
    saved = (tmp_path / ".env").read_text()
    assert "OPENSPLINE_GPUS='0,1'" in saved and "OPENSPLINE_GPU=" not in saved
    assert "OPENAI_API_KEY=keep" in saved


def test_configured_high_group_is_retained_when_starting_low(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSPLINE_QUALITY", "low")
    settings = configure(workers(tmp_path, "workers: [{id: pair, quality: high, devices: [1, 2]}]"))
    assert settings.workers[0].devices == (1,)
    assert settings.demo_high_devices == (1, 2)


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


@pytest.mark.parametrize("quality", ["low", "high"])
def test_single_gpu_downloads_both_demo_profiles_before_launch(tmp_path, monkeypatch, quality):
    path = workers(tmp_path, f"workers: [{{id: main, quality: {quality}, devices: [0]}}]\n")
    environment(tmp_path)
    downloads = []
    monkeypatch.setattr("openspline_server.cli.download", downloads.append)
    prepare(configure(path))
    assert downloads[0].quality == "all"
    assert downloads[0].directory == str(tmp_path / "models")


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
