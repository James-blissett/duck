"""Streaming LLM client, backend-agnostic.

The conversation engine talks to a locally served Gemma 4 E2B model. The model
may be served by ``litert-lm serve``, ``llama.cpp`` server, or Ollama — all of
which expose an OpenAI-compatible ``/v1/chat/completions`` endpoint with
Server-Sent Events (SSE) streaming. We hide that behind the :class:`LLMClient`
protocol so the backend is swappable: anything implementing
``generate_stream(messages) -> AsyncIterator[str]`` works, including a fake for
tests.

Reference for LiteRT-LM usage: github.com/fikrikarim/parlor.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Literal, Protocol, TypedDict, runtime_checkable

import httpx


class ChatMessage(TypedDict):
    """One OpenAI-style chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


@runtime_checkable
class LLMClient(Protocol):
    """A swappable streaming chat backend.

    Implementations yield reply text incrementally (token-by-token or chunked)
    so callers can render or speak the reply as it is produced.
    """

    def generate_stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Stream the assistant reply for ``messages`` as text chunks."""
        ...


class OpenAICompatibleClient:
    """:class:`LLMClient` over an OpenAI-compatible ``/v1/chat/completions`` API.

    Works against ``litert-lm serve``, ``llama.cpp`` server, and Ollama. Pass a
    custom :class:`httpx.AsyncClient` (e.g. with a mock transport) to test the
    SSE parsing without a live server.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.7,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client if this instance created it."""
        if self._owns_client:
            await self._client.aclose()

    async def generate_stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "messages": list(messages),
            "temperature": self._temperature,
            "stream": True,
        }
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                chunk = _parse_sse_line(line)
                if chunk is not None:
                    yield chunk


def _parse_sse_line(line: str) -> str | None:
    """Extract a content delta from one SSE line, or ``None`` if there is none.

    Lines look like ``data: {json}``; the terminal line is ``data: [DONE]``.
    Comment/blank lines and chunks without a content delta yield ``None``.
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == "" or data == "[DONE]":
        return None
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    choices = parsed.get("choices")
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) and content != "" else None
