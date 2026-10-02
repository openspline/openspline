"""Exercise the shell installer with a local archive and fake external tools."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("no_start", [False, True])
def test_archive_install_preserves_config_and_launches_native(tmp_path, no_start):
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "release"
    for file in (
        "packages/server/pyproject.toml",
        "requirements/server.txt",
        "workers.yaml",
        ".env.example",
        "run.sh",
    ):
        target = source / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / file, target)
    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="release")
    install = tmp_path / "install with spaces"
    (install / ".tools").mkdir(parents=True)
    (install / ".venv/bin").mkdir(parents=True)
    (install / ".env").write_text("OPENAI_API_KEY=keep-existing\n")
    (install / "workers.yaml").write_text("workers: [{id: custom, quality: high, devices: [1]}]\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.jsonl"
    stub = f"#!{sys.executable}\nimport os, json, sys\nwith open(os.environ['INSTALL_TEST_LOG'], 'a') as f: f.write(json.dumps(sys.argv) + '\\n')\n"
    for tool in (install / ".tools/uv", install / ".venv/bin/python", bin_dir / "nvidia-smi"):
        tool.write_text(stub)
        tool.chmod(0o755)
    # The archive and tools are the only external inputs; no GPU or network is used.
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENSPLINE_")}
    env.update(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        OPENSPLINE_INSTALL_DIR=str(install),
        OPENSPLINE_ARCHIVE_URL=archive.as_uri(),
        OPENSPLINE_NO_START=str(int(no_start)),
        INSTALL_TEST_LOG=str(log),
    )
    result = subprocess.run(
        ["sh", str(repo / "install.sh")], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert (install / ".env").read_text() == "OPENAI_API_KEY=keep-existing\n"
    assert "id: custom" in (install / "workers.yaml").read_text()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert any("--require-hashes" in call for call in calls)
    launches = [call for call in calls if "openspline_server.native" in call]
    assert len(launches) == (1 if no_start else 2)
    assert launches[0][-2:] == ["--prepare", "--save-selection"]
    if not no_start:
        assert launches[1][-1] == "openspline_server.native"
    assert not list((install / "runtime/tmp").iterdir())
