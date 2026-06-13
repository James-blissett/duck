"""Streaming LLM client tests (Spec 1 acceptance 1: token-by-token output).

The SSE parsing is exercised end-to-end via an ``httpx.MockTransport`` so no
live model server is needed.
"""

from __future__ import annotations

import json

import httpx
import pytest

from duckbrain.llm.client import ChatMessage, OpenAICompatibleClient, _parse_sse_line


def _sse(*deltas: str) -> bytes:
    lines: list[str] = []
    for delta in deltas:
        body = {"choices": [{"delta": {"content": delta}}]}
        lines.append(f"data: {json.dumps(body)}\n")
    lines.append("data: [DONE]\n")
    return "\n".join(lines).encode("utf-8")


async def _collect(client: OpenAICompatibleClient, messages: list[ChatMessage]) -> list[str]:
    return [chunk async for chunk in client.generate_stream(messages)]


async def test_streams_tokens_in_order() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=_sse("Hello", ", ", "Margaret", "."))

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = OpenAICompatibleClient("http://test", "gemma-4-e2b", http_client=http_client)

    messages: list[ChatMessage] = [
        {"role": "system", "content": "You are a duck."},
        {"role": "user", "content": "Hi"},
    ]
    chunks = await _collect(client, messages)

    assert chunks == ["Hello", ", ", "Margaret", "."]
    assert "".join(chunks) == "Hello, Margaret."
    # Request shape: OpenAI-compatible streaming chat completion.
    assert captured["url"] == "http://test/v1/chat/completions"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["stream"] is True
    assert body["model"] == "gemma-4-e2b"
    assert body["messages"] == messages

    await http_client.aclose()


async def test_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = OpenAICompatibleClient("http://test", "m", http_client=http_client)

    with pytest.raises(httpx.HTTPStatusError):
        await _collect(client, [{"role": "user", "content": "hi"}])

    await http_client.aclose()


def test_parse_sse_line_variants() -> None:
    assert _parse_sse_line('data: {"choices":[{"delta":{"content":"hi"}}]}') == "hi"
    assert _parse_sse_line("data: [DONE]") is None
    assert _parse_sse_line("") is None
    assert _parse_sse_line(": keep-alive comment") is None
    # Final chunk with no content delta (e.g. finish_reason only).
    assert _parse_sse_line('data: {"choices":[{"delta":{}}]}') is None
    assert _parse_sse_line("data: not-json") is None
