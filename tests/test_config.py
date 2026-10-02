import asyncio
import json
from pathlib import Path

import pytest
from openspline import Openspline, OpensplineConfig


def test_config_precedence_and_session_override(monkeypatch):
    monkeypatch.setenv("OPENSPLINE_URL", "http://environment:7860")
    monkeypatch.setenv("OPENSPLINE_API_KEY", "environment-key")
    config = OpensplineConfig(quality="high", timeout=300)
    client = Openspline(config=config, url="https://service.example/", api_key="", timeout=12)
    assert client.url == "https://service.example"
    assert client.api_key == ""
    assert client.timeout == 12
    assert client.avatar("unused.png").quality == "high"
    assert client.avatar("unused.png", quality="low").quality == "low"
    assert config.timeout == 300
    assert "environment-key" not in repr(config)


def test_dict_config_and_legacy_constructor(monkeypatch):
    monkeypatch.setenv("OPENSPLINE_URL", "invalid environment default")
    client = Openspline("http://localhost:7860/", "key", 30, config={"quality": "high"})
    assert client.config == OpensplineConfig(
        url="http://localhost:7860", api_key="key", timeout=30, quality="high"
    )
    with pytest.raises(TypeError):
        Openspline(config={"quailty": "high"})


@pytest.mark.parametrize(
    "settings",
    [
        {"url": "ftp://example.com"},
        {"url": "http://user:key@example.com"},
        {"url": "https://example.com?key=secret"},
        {"url": "http://localhost:99999"},
        {"url": ""},
        {"quality": "pro"},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"viewer_timeout": True},
        {"viewer_timeout": -1},
        {"api_key": 123},
    ],
)
def test_invalid_configuration_fails_at_initialization(settings):
    with pytest.raises(ValueError):
        Openspline(**settings)


async def test_viewer_timeout_uses_client_config():
    avatar = Openspline(viewer_timeout=0.001).avatar("unused.png")
    with pytest.raises(asyncio.TimeoutError):
        await avatar.wait_for_viewer()
    avatar.viewer_ready.set()
    await avatar.wait_for_viewer(timeout=0.1)


def test_schema_validates_client_and_worker_examples():
    import yaml
    from jsonschema import Draft202012Validator

    schema = json.loads(Path("config.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    for kind, data in [
        ("python", {"quality": "high", "timeout": 30}),
        ("node", {"quality": "low", "viewerTimeout": 1000}),
        ("server", yaml.safe_load(Path("workers.yaml").read_text())),
    ]:
        validator = Draft202012Validator({**schema, "$ref": f"#/$defs/{kind}"})
        validator.validate(data)
        assert not validator.is_valid({"unknown_field": True})
    assert not Draft202012Validator(schema).is_valid({"quality": "pro"})
