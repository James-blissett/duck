"""LLM: model client + persona (Gemma 4 E2B via LiteRT-LM/llama.cpp/Ollama).

The conversation engine streams replies from a locally served model behind the
:class:`LLMClient` protocol, and builds its system prompt (persona + remembered
profile) with :func:`build_system_prompt`.
"""

from duckbrain.llm.bootstrap import (
    BootstrapError,
    PullProgress,
    ensure_ready,
    find_ollama,
    stop_server,
)
from duckbrain.llm.client import (
    ChatMessage,
    LLMClient,
    OpenAICompatibleClient,
)
from duckbrain.llm.persona import build_system_prompt

__all__ = [
    "BootstrapError",
    "ChatMessage",
    "LLMClient",
    "OpenAICompatibleClient",
    "PullProgress",
    "build_system_prompt",
    "ensure_ready",
    "find_ollama",
    "stop_server",
]
