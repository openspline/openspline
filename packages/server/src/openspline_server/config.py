import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class WorkerConfig:
    id: str = "gpu-0"
    quality: str = "low"
    devices: tuple[int, ...] = (0,)


@dataclass
class Settings:
    workers: list[WorkerConfig] = field(default_factory=lambda: [WorkerConfig()])
    model_dir: str = field(
        default_factory=lambda: os.getenv("OPENSPLINE_MODEL_DIR", "models/avatar")
    )
    audio_model_dir: str = field(
        default_factory=lambda: os.getenv("OPENSPLINE_AUDIO_MODEL_DIR", "models/audio")
    )
    runtime_dir: str = field(default_factory=lambda: os.getenv("OPENSPLINE_RUNTIME_DIR", "runtime"))
    api_key: str = field(default_factory=lambda: os.getenv("OPENSPLINE_API_KEY", ""))
    public_url: str = field(
        default_factory=lambda: os.getenv("OPENSPLINE_PUBLIC_URL", "http://localhost:7860")
    )
    backend: str = "gpu"
    session_ttl: float = 3600
    idle_timeout: float = 120
    token_ttl: float = 300
    startup_timeout: float = 600
    inference_timeout: float = 180
    ice_servers: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.backend not in {"gpu", "test"}:
            raise ValueError("Unknown backend")
        occupied, ids = set(), set()
        if not self.workers:
            raise ValueError("At least one worker is required")
        for w in self.workers:
            if w.id in ids:
                raise ValueError("Duplicate worker id")
            if w.quality not in {"low", "high"}:
                raise ValueError("quality must be low or high")
            if not w.devices or any(type(d) is not int or d < 0 for d in w.devices):
                raise ValueError("Invalid device IDs")
            if len(set(w.devices)) != len(w.devices) or occupied.intersection(w.devices):
                raise ValueError("GPU assignments overlap")
            if w.quality == "low" and len(w.devices) != 1:
                raise ValueError("Low quality uses one GPU per worker")
            ids.add(w.id)
            occupied.update(w.devices)
        self.model_dir = str(Path(self.model_dir).resolve())
        self.audio_model_dir = str(Path(self.audio_model_dir).resolve())
        self.runtime_dir = str(Path(self.runtime_dir).resolve())

    @classmethod
    def from_file(cls, path=None, **kwargs):
        data = yaml.safe_load(Path(path).read_text()) if path else {}
        data = data or {}
        workers = [
            WorkerConfig(id=w["id"], quality=w.get("quality", "low"), devices=tuple(w["devices"]))
            for w in data.pop("workers", [{"id": "gpu-0", "devices": [0]}])
        ]
        return cls(workers=workers, **(data | kwargs))
