"""Exercise the shell installer with a local archive and fake external tools."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
from io import StringIO
from pathlib import Path

import pytest
from openspline_server.install_message import print_install_message
from rich.console import Console


def test_install_message_shows_launch_and_optional_keys():
    install = Path("/tmp/install with spaces")
    plain = StringIO()
    print_install_message(install, Console(file=plain, force_terminal=False, width=120))
    output = plain.getvalue()
    assert "sh '/tmp/install with spaces/run.sh'" in output
    assert "/tmp/install with spaces/.env" in output
    assert "OPENAI_API_KEY=your_openai_key" in output
    assert "GEMINI_API_KEY=your_gemini_key" in output
    assert "Audio file mode needs no key" in output
    assert "\x1b[" not in output

    colored = StringIO()
    print_install_message(
        install, Console(file=colored, force_terminal=True, color_system="standard", width=120)
    )
    assert "\x1b[" in colored.getvalue()


@pytest.mark.parametrize("cuda", ["cu126", "cu128"])
@pytest.mark.parametrize("check_fails", [False, True])
@pytest.mark.parametrize("download_fails", [False, True])
def test_archive_install_preserves_config_without_launching_native(
    tmp_path, cuda, check_fails, download_fails
):
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "release"
    for file in (
        "packages/server/pyproject.toml",
        "requirements/server.txt",
        "requirements/cu126.txt",
        "requirements/cu128.txt",
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
    stub += f"if sys.argv[-1] == 'openspline_server.hardware': print({cuda!r})\n"
    stub += f"if sys.argv[-2:] == ['openspline_server.hardware', '--check'] and {check_fails}: sys.exit(1)\n"
    stub += f"if '--prepare' in sys.argv and {download_fails}: sys.exit(1)\n"
    for tool in (install / ".tools/uv", install / ".venv/bin/python", bin_dir / "nvidia-smi"):
        tool.write_text(stub)
        tool.chmod(0o755)
    # The archive and tools are the only external inputs; no GPU or network is used.
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENSPLINE_")}
    env.update(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        OPENSPLINE_INSTALL_DIR=str(install),
        OPENSPLINE_ARCHIVE_URL=archive.as_uri(),
        INSTALL_TEST_LOG=str(log),
    )
    result = subprocess.run(
        ["sh", str(repo / "install.sh")], env=env, capture_output=True, text=True, timeout=30
    )
    if check_fails:
        assert result.returncode != 0
        assert "--prepare" not in log.read_text()
        assert "openspline_server.install_message" not in log.read_text()
        return
    if download_fails:
        assert result.returncode != 0
        launches = [
            json.loads(line)
            for line in log.read_text().splitlines()
            if "openspline_server.native" in line
        ]
        assert len(launches) == 1 and "--prepare" in launches[0]
        assert "openspline_server.install_message" not in log.read_text()
        return
    assert result.returncode == 0, result.stderr
    assert (install / ".env").read_text() == "OPENAI_API_KEY=keep-existing\n"
    assert "id: custom" in (install / "workers.yaml").read_text()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert any("--require-hashes" in call for call in calls)
    gpu_install = next(call for call in calls if "--torch-backend" in call)
    assert gpu_install[gpu_install.index("--torch-backend") + 1] == cuda
    assert str(install / f"requirements/{cuda}.txt") in gpu_install
    assert any(call[-2:] == ["openspline_server.hardware", "--check"] for call in calls)
    launches = [call for call in calls if "openspline_server.native" in call]
    assert len(launches) == 1
    assert launches[0][-2:] == ["--prepare", "--save-selection"]
    assert any(
        call[-2:] == ["openspline_server.install_message", str(install)]
        for call in calls
    )
    assert not list((install / "runtime/tmp").iterdir())
