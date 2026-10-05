import os
from collections.abc import Callable
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
    demo_high_devices: tuple[int, ...] | None = None
    # Set by the native launcher, never accepted from public configuration.
    save_demo_selection: Callable | None = field(default=None, init=False, repr=False)
    model_dir: str = field(
        default_factory=lambda: os.getenv("OPENSPLINE_MODEL_DIR", "models/avatar")
    )
    audio_model_dir: str = field(
        default_factory=lambda: os.getenv("OPENSPLINE_AUDIO_MODEL_DIR", "models/audio")
    )
    runtime_dir: str = field(default_factory=lambda: os.getenv("OPENSPLINE_RUNTIME_DIR", "runtime"))
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
        if self.demo_high_devices is not None:
            group = self.demo_high_devices
            if (
                len(self.workers) != 1
                or not isinstance(group, (tuple, list))
                or not group
                or any(type(d) is not int or d < 0 for d in group)
                or len(set(group)) != len(group)
            ):
                raise ValueError(
                    "demo_high_devices requires one worker and unique nonnegative GPU IDs"
                )
            if self.workers[0].devices[0] != group[0]:
                raise ValueError(
                    "The first demo_high_devices GPU must match the worker's first GPU"
                )
            if self.workers[0].quality == "high" and tuple(group) != self.workers[0].devices:
                raise ValueError("A high-quality worker must use all demo_high_devices")
            self.demo_high_devices = tuple(group)
        self.model_dir = str(Path(self.model_dir).resolve())
        self.audio_model_dir = str(Path(self.audio_model_dir).resolve())
        self.runtime_dir = str(Path(self.runtime_dir).resolve())

    @property
    def gpu_devices(self):
        return sorted(
            {d for w in self.workers for d in w.devices} | set(self.demo_high_devices or ())
        )

    @classmethod
    def from_file(cls, path=None, **kwargs):
        data = yaml.safe_load(Path(path).read_text()) if path else {}
        data = data or {}
        workers = [
            WorkerConfig(id=w["id"], quality=w.get("quality", "low"), devices=tuple(w["devices"]))
            for w in data.pop("workers", [{"id": "gpu-0", "devices": [0]}])
        ]
        return cls(workers=workers, **(data | kwargs))
