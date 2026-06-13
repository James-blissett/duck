"""Runtime configuration for the duck brain.

Settings are environment-driven (``.env`` file, ``DUCK_`` prefix) via
pydantic-settings. Importing this module has no side effects; call
:func:`get_settings` to load configuration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Brain configuration. All fields overridable via ``DUCK_*`` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="DUCK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model paths (no models are downloaded in Stage 0) ---
    llm_model_path: Path = Field(default=Path("models/gemma4-e2b"))
    stt_model_path: Path = Field(default=Path("models/parakeet"))
    tts_model_path: Path = Field(default=Path("models/kokoro"))

    # --- LLM backend (OpenAI-compatible local endpoint: Ollama by default;
    # also works with litert-lm serve or llama.cpp server — swap base URL/model) ---
    llm_base_url: str = Field(default="http://127.0.0.1:11434")
    llm_model: str = Field(default="gemma3:4b")

    # When the backend is Ollama, the brain can start the server and download the
    # model on first run so the operator does nothing. Disable to point at an
    # already-running endpoint (litert-lm serve, llama.cpp) you manage yourself.
    llm_auto_bootstrap: bool = Field(default=True)
    # Optional explicit path to the ``ollama`` binary; auto-discovered if unset.
    ollama_bin: Path | None = Field(default=None)

    # --- Persona ---
    duck_name: str = Field(default="Waddles")
    handoff_phrase: str = Field(default="I'll let the staff know you'd like a hand.")

    # --- Audio device IDs (sounddevice indices; None = system default) ---
    input_device: int | None = Field(default=None)
    output_device: int | None = Field(default=None)

    # --- Brain bus (WebSocket) ---
    brain_host: str = Field(default="127.0.0.1")
    brain_port: int = Field(default=8765)

    # --- Local data directory (resident profiles, conversation logs) ---
    data_dir: Path = Field(default=Path("data"))

    @property
    def db_path(self) -> Path:
        """Path to the per-resident memory SQLite database."""
        return self.data_dir / "memory.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached, environment-loaded settings."""
    return Settings()
