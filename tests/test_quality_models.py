import json
from types import SimpleNamespace

import pytest
from openspline_server.quality import prepare_quality


@pytest.mark.parametrize("quality", ["low", "high"])
async def test_switch_requires_complete_local_models_without_downloading(
    tmp_path, monkeypatch, quality
):
    def forbidden(*args, **kwargs):
        pytest.fail("Quality switching must not download models")

    monkeypatch.setattr("openspline_server.cli.download", forbidden)
    monkeypatch.setattr("huggingface_hub.snapshot_download", forbidden)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)
    settings = SimpleNamespace(
        backend="gpu", model_dir=str(tmp_path / "avatar"), audio_model_dir=str(tmp_path / "audio")
    )
    with pytest.raises(ValueError, match="does not download"):
        await prepare_quality(settings, quality)

    model = tmp_path / "avatar" / ("Model_Lite" if quality == "low" else "Model_Pro")
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}})
    )
    (model / "part-1.safetensors").write_bytes(b"weight")
    (model / "part-2.safetensors").write_bytes(b"weight")
    vae = tmp_path / "avatar" / ("VAE_LTX" if quality == "low" else "VAE_Wan")
    vae.mkdir()
    (vae / "config.json").write_text("{}")
    (vae / ("autoencoder.pth" if quality == "low" else "Wan2.1_VAE.pth")).write_bytes(b"weight")
    audio = tmp_path / "audio"
    audio.mkdir()
    for name in ["config.json", "preprocessor_config.json", "model.safetensors"]:
        (audio / name).write_bytes(b"weight")
    await prepare_quality(settings, quality)

    # A partially transferred sharded model must not replace the loaded quality.
    (model / "part-2.safetensors").unlink()
    with pytest.raises(ValueError, match="missing or incomplete"):
        await prepare_quality(settings, quality)
