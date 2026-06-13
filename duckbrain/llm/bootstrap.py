"""Zero-touch Ollama bootstrap: make the model server and model ready.

When the backend is Ollama (the default), the brain starts the local Ollama
server if it is not already running and downloads the configured model on first
run, so the operator does not have to install or pull anything by hand.

This is a convenience layer around an Ollama endpoint. It is gated by
``DUCK_LLM_AUTO_BOOTSTRAP``: point ``DUCK_LLM_BASE_URL`` at a server you manage
yourself (``litert-lm serve``, ``llama.cpp``) and set the flag to ``false`` to
skip all of this. Importing this module has no side effects.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

# Where the headless Ollama runtime is vendored when there is no system install.
VENDORED_OLLAMA = Path.home() / ".local" / "share" / "duck" / "ollama" / "ollama"


@dataclass(frozen=True)
class PullProgress:
    """Progress of a model download (one streamed update from Ollama)."""

    status: str
    completed: int | None = None
    total: int | None = None

    @property
    def percent(self) -> float | None:
        if self.total is None or self.completed is None or self.total == 0:
            return None
        return 100.0 * self.completed / self.total


ProgressCallback = Callable[[PullProgress], None]


class BootstrapError(RuntimeError):
    """The model backend could not be made ready."""


def find_ollama(explicit: Path | None = None) -> Path | None:
    """Locate the ``ollama`` binary: explicit override, then PATH, then vendored."""
    if explicit is not None and explicit.exists():
        return explicit
    on_path = shutil.which("ollama")
    if on_path is not None:
        return Path(on_path)
    if VENDORED_OLLAMA.exists():
        return VENDORED_OLLAMA
    return None


def _host_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    return host, port


def _is_local(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


async def _server_reachable(client: httpx.AsyncClient) -> bool:
    try:
        response = await client.get("/api/version", timeout=2.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


async def _model_present(client: httpx.AsyncClient, model: str) -> bool:
    """Whether ``model`` is already downloaded (matching Ollama's tag rules)."""
    response = await client.get("/api/tags", timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    names = {entry.get("name") for entry in payload.get("models", [])}
    candidates = {model}
    if ":" in model:
        candidates.add(model)
    else:
        candidates.add(f"{model}:latest")
    return bool(candidates & names)


async def _pull_model(
    client: httpx.AsyncClient,
    model: str,
    on_progress: ProgressCallback,
) -> None:
    """Stream ``/api/pull`` for ``model``, reporting progress; raise on failure."""
    async with client.stream(
        "POST", "/api/pull", json={"model": model, "stream": True}, timeout=None
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if "error" in event:
                raise BootstrapError(f"Failed to pull {model}: {event['error']}")
            on_progress(
                PullProgress(
                    status=event.get("status", ""),
                    completed=event.get("completed"),
                    total=event.get("total"),
                )
            )


def _start_server(ollama_bin: Path, host: str, port: int) -> subprocess.Popen[bytes]:
    """Spawn ``ollama serve`` bound to ``host:port``, tied to this process.

    The server is deliberately *not* detached: it shares this process's session
    so closing the terminal (SIGHUP) stops it, and the caller also stops it
    explicitly via :func:`stop_server` on exit. This avoids leaving a multi-GB
    model server running in the background after the CLI quits.
    """
    env = dict(os.environ)
    env["OLLAMA_HOST"] = f"{host}:{port}"
    return subprocess.Popen(
        [str(ollama_bin), "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_server(proc: subprocess.Popen[bytes] | None) -> None:
    """Stop a server started by :func:`ensure_ready` (no-op if ``None``/dead)."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


async def _wait_until_reachable(client: httpx.AsyncClient, timeout: float = 60.0) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if await _server_reachable(client):
            return True
        await asyncio.sleep(0.5)
    return False


async def ensure_ready(
    base_url: str,
    model: str,
    *,
    ollama_bin: Path | None = None,
    auto_start: bool = True,
    on_progress: ProgressCallback | None = None,
    on_status: Callable[[str], None] | None = None,
) -> subprocess.Popen[bytes] | None:
    """Ensure the Ollama server is up and ``model`` is downloaded.

    Starts a local server if needed and pulls the model on first run. Returns the
    spawned server process so the caller can stop it on exit (via
    :func:`stop_server`), or ``None`` if a server was already running (in which
    case we leave it alone — we only stop what we started). Raises
    :class:`BootstrapError` if the backend cannot be made ready (e.g. no Ollama
    binary and nothing already listening).
    """
    notify = on_status or (lambda _msg: None)
    host, port = _host_port(base_url)
    started: subprocess.Popen[bytes] | None = None

    async with httpx.AsyncClient(base_url=base_url.rstrip("/")) as client:
        if not await _server_reachable(client):
            if not (auto_start and _is_local(host)):
                raise BootstrapError(
                    f"No model server reachable at {base_url} and it is not local, "
                    "so it cannot be started automatically."
                )
            binary = find_ollama(ollama_bin)
            if binary is None:
                raise BootstrapError(
                    "Ollama is not installed and no server is running at "
                    f"{base_url}. Install Ollama (https://ollama.com) or point "
                    "DUCK_LLM_BASE_URL at a running endpoint."
                )
            notify("Starting the local model server…")
            started = _start_server(binary, host, port)
            if not await _wait_until_reachable(client):
                stop_server(started)
                raise BootstrapError(
                    f"Started Ollama but it did not become reachable at {base_url}."
                )

        if not await _model_present(client, model):
            notify(f"Downloading model {model} (first run only)…")
            await _pull_model(client, model, on_progress or (lambda _progress: None))
            notify(f"Model {model} is ready.")

    return started
