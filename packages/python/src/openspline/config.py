"""Validated, dependency-free client configuration."""

import math
import os
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

Quality = Literal["low", "high"]


@dataclass(frozen=True)
class OpensplineConfig:
    url: str = field(default_factory=lambda: os.getenv("OPENSPLINE_URL", "http://localhost:7860"))
    api_key: str = field(default_factory=lambda: os.getenv("OPENSPLINE_API_KEY", ""), repr=False)
    quality: Quality = "low"
    timeout: float = 180
    viewer_timeout: float = 60

    def __post_init__(self):
        if not isinstance(self.url, str):
            raise ValueError("url must be an HTTP or HTTPS URL")
        try:
            parsed = urlsplit(self.url)
            valid = parsed.scheme in {"http", "https"} and parsed.hostname and parsed.port != 0
        except ValueError:
            valid = False
        if (
            not valid
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or any(c.isspace() for c in self.url)
        ):
            raise ValueError(
                "url must be an HTTP or HTTPS URL without credentials, query, or fragment"
            )
        if not isinstance(self.api_key, str):
            raise ValueError("api_key must be a string")
        if self.quality not in ("low", "high"):
            raise ValueError("quality must be low or high")
        for name in ("timeout", "viewer_timeout"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number of seconds")
        object.__setattr__(self, "url", self.url.rstrip("/"))
