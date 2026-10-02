import io
import json
import os
import sys
from types import ModuleType, SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub import constants, file_download
from openspline_server.cli import AUDIO_REVISION, MODEL_REVISION, download
from requests import Response


@pytest.mark.parametrize(
    "enabled,available,expected", [(True, False, False), (True, True, True), (False, False, False)]
)
def test_optional_accelerator_does_not_block_model_download(
    tmp_path, monkeypatch, enabled, available, expected
):
    # Include Hub's already-imported state: changing only os.environ cannot fix this.
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", str(int(enabled)))
    monkeypatch.setattr(constants, "HF_HUB_ENABLE_HF_TRANSFER", enabled)
    monkeypatch.setitem(sys.modules, "hf_transfer", ModuleType("hf_transfer") if available else None)
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "model_info",
        lambda self, repo, revision: SimpleNamespace(sha=revision),
    )
    def request(**kwargs):
        response = Response()
        response.status_code = 200
        response.headers["Content-Length"] = "5"
        response.raw = io.BytesIO(b"model")
        return response

    monkeypatch.setattr(file_download, "_request_wrapper", request)
    downloaded = []

    def snapshot(repo, *, revision, allow_patterns, local_dir):
        target = io.BytesIO()
        # Use the real download routine which raised the reported ValueError.
        file_download.http_get("https://example.invalid/model", target, expected_size=5)
        assert target.getvalue() == b"model"
        downloaded.append((local_dir.name, revision))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    download(SimpleNamespace(directory=tmp_path, quality="low", revision=None))
    assert downloaded == [("avatar", MODEL_REVISION), ("audio", AUDIO_REVISION)]
    assert constants.HF_HUB_ENABLE_HF_TRANSFER is expected
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == str(int(expected))
    assert json.loads((tmp_path / "manifest.json").read_text())["avatar"]["revision"] == MODEL_REVISION
