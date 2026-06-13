"""Ollama bootstrap tests.

The HTTP interactions (server reachability, tag listing, pull streaming) are
exercised via ``httpx.MockTransport`` so no Ollama install is needed. Binary
discovery is tested against the filesystem with monkeypatching.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from duckbrain.llm import bootstrap
from duckbrain.llm.bootstrap import (
    BootstrapError,
    PullProgress,
    _model_present,
    _pull_model,
    ensure_ready,
    find_ollama,
    stop_server,
)


def _client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler, base_url="http://127.0.0.1:11434")


def test_stop_server_noop_on_none() -> None:
    stop_server(None)  # must not raise


def test_stop_server_terminates_process() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    stop_server(proc)
    assert proc.poll() is not None  # process has exited
    # Idempotent: stopping an already-dead process is a no-op.
    stop_server(proc)


def test_pull_progress_percent() -> None:
    assert PullProgress("pulling", completed=50, total=200).percent == 25.0
    assert PullProgress("manifest").percent is None
    assert PullProgress("x", completed=1, total=0).percent is None


def test_find_ollama_prefers_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "ollama"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert find_ollama(binary) == binary


def test_find_ollama_falls_back_to_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ollama")
    assert find_ollama(None) == Path("/usr/bin/ollama")


def test_find_ollama_none_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(bootstrap, "VENDORED_OLLAMA", Path("/nonexistent/ollama"))
    assert find_ollama(None) is None


async def test_model_present_matches_tags() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "gemma3:4b"}]})

    async with _client(httpx.MockTransport(handler)) as client:
        assert await _model_present(client, "gemma3:4b") is True
        assert await _model_present(client, "gemma3:1b") is False


async def test_model_present_handles_implicit_latest() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "gemma3:latest"}]})

    async with _client(httpx.MockTransport(handler)) as client:
        assert await _model_present(client, "gemma3") is True


async def test_pull_model_streams_progress_and_completes() -> None:
    lines = [
        {"status": "pulling manifest"},
        {"status": "pulling abc", "completed": 500, "total": 1000},
        {"status": "pulling abc", "completed": 1000, "total": 1000},
        {"status": "success"},
    ]
    body = "\n".join(json.dumps(line) for line in lines).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["model"] == "gemma3:4b"
        return httpx.Response(200, content=body)

    seen: list[PullProgress] = []
    async with _client(httpx.MockTransport(handler)) as client:
        await _pull_model(client, "gemma3:4b", seen.append)

    assert [p.status for p in seen] == [
        "pulling manifest",
        "pulling abc",
        "pulling abc",
        "success",
    ]
    assert seen[1].percent == 50.0


async def test_pull_model_raises_on_error_event() -> None:
    body = json.dumps({"error": "model not found"}).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    async with _client(httpx.MockTransport(handler)) as client:
        with pytest.raises(BootstrapError, match="model not found"):
            await _pull_model(client, "ghost", lambda _p: None)


async def test_ensure_ready_noop_when_server_up_and_model_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma3:4b"}]})
        raise AssertionError(f"unexpected request to {request.url.path}")

    transport = httpx.MockTransport(handler)
    real_cls = httpx.AsyncClient  # capture before patching to avoid recursion
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_cls(transport=transport, base_url="http://127.0.0.1:11434"),
    )

    statuses: list[str] = []
    # Server reachable + model present => no server start, no pull.
    await ensure_ready("http://127.0.0.1:11434", "gemma3:4b", on_status=statuses.append)
    assert statuses == []


async def test_ensure_ready_pulls_when_model_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pulled: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if path == "/api/pull":
            pulled.append(json.loads(request.content)["model"])
            return httpx.Response(200, content=json.dumps({"status": "success"}).encode())
        raise AssertionError(f"unexpected request to {path}")

    transport = httpx.MockTransport(handler)
    real_cls = httpx.AsyncClient  # capture before patching to avoid recursion
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_cls(transport=transport, base_url="http://127.0.0.1:11434"),
    )

    statuses: list[str] = []
    await ensure_ready("http://127.0.0.1:11434", "gemma3:4b", on_status=statuses.append)
    assert pulled == ["gemma3:4b"]
    assert any("Downloading" in s for s in statuses)
