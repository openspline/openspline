import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from test_protocol import image


def test_node_against_service(tmp_path):
    if not Path("packages/node/dist/index.js").exists():
        pytest.skip("Build Node packages first")
    portrait = tmp_path / "portrait.png"
    portrait.write_bytes(image())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "OPENSPLINE_URL": url,
        "OPENSPLINE_PUBLIC_URL": url,
        "OPENSPLINE_API_KEY": "conformance",
        "OPENSPLINE_RUNTIME_DIR": str(tmp_path / "runtime"),
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from openspline_server.cli import main; main()",
            "serve",
            "--backend",
            "test",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                if httpx.get(url + "/readyz").status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            time.sleep(0.05)
        result = subprocess.run(
            ["node", "scripts/node-conformance.mjs", str(portrait)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        process.terminate()
        process.wait(timeout=10)
