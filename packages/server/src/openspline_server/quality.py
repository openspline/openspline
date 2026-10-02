"""Local model checks for demo quality switching; never download during playback."""

import json
from pathlib import Path


def _weights_present(directory, names):
    for name in names:
        path = directory / name
        if path.is_file() and path.stat().st_size:
            return True
        index = directory / (name + ".index.json")
        if index.is_file():
            try:
                shards = set(json.loads(index.read_text())["weight_map"].values())
                if shards and all(
                    (directory / shard).is_file() and (directory / shard).stat().st_size
                    for shard in shards
                ):
                    return True
            except (ValueError, KeyError, TypeError, OSError):
                pass
    return False


async def prepare_quality(settings, quality):
    if settings.backend == "test":
        return
    avatar, audio = Path(settings.model_dir), Path(settings.audio_model_dir)
    model = avatar / ("Model_Lite" if quality == "low" else "Model_Pro")
    ready = (model / "config.json").is_file() and _weights_present(
        model, ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"]
    )
    if quality == "low":
        vae = avatar / "VAE_LTX"
        ready = (
            ready
            and (vae / "config.json").is_file()
            and _weights_present(vae, ["autoencoder.pth", "diffusion_pytorch_model.safetensors"])
        )
    else:
        ready = ready and _weights_present(avatar / "VAE_Wan", ["Wan2.1_VAE.pth"])
    ready = ready and all(
        (audio / name).is_file() for name in ["config.json", "preprocessor_config.json"]
    )
    ready = ready and _weights_present(audio, ["model.safetensors", "pytorch_model.bin"])
    if not ready:
        raise ValueError(
            f"{quality.capitalize()}-quality model files are missing or incomplete. "
            "Stop the service and rerun the installer or run.sh --prepare before restarting. "
            "The demo does not download models."
        )
