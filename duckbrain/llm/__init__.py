"""LLM: model client + persona (Gemma 4 E2B via LiteRT-LM/llama.cpp/Ollama).

The conversation engine streams replies from a locally served model behind the
:class:`LLMClient` protocol, and builds its system prompt (persona + remembered
profile) with :func:`build_system_prompt`.
"""

from duckbrain.llm.client import (
    ChatMessage,
    LLMClient,
    OpenAICompatibleClient,
)
from duckbrain.llm.persona import build_system_prompt

__all__ = [
    "ChatMessage",
    "LLMClient",
    "OpenAICompatibleClient",
    "build_system_prompt",
]
