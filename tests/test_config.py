"""Configuration loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from duckbrain.config import Settings


def test_defaults_have_no_model_downloads() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    # Stage 0: paths are configured but nothing is fetched.
    assert settings.llm_model_path == Path("models/gemma4-e2b")
    assert settings.brain_host == "127.0.0.1"
    assert settings.brain_port == 8765
    assert settings.input_device is None


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUCK_BRAIN_PORT", "9000")
    monkeypatch.setenv("DUCK_DATA_DIR", "/tmp/duckdata")
    monkeypatch.setenv("DUCK_INPUT_DEVICE", "3")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.brain_port == 9000
    assert settings.data_dir == Path("/tmp/duckdata")
    assert settings.input_device == 3
