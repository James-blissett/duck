# Companion Duck — Brain

On-device conversational companion robot for aged care, built on Open Duck Mini
v2. This repo is the **brain**: VAD → STT → conversation engine → safety → TTS,
plus longitudinal per-resident memory. All inference runs on-device; no resident
voice data leaves the facility network.

See [CLAUDE.md](CLAUDE.md) for architecture, the brain⇄body message contract, the
hard safety requirements, and coding standards.

## Quickstart

```bash
make setup      # uv sync (fetches Python 3.11 + deps; no models downloaded)
make test       # pytest
make lint       # ruff check + ruff format --check + mypy --strict
make eval       # persona red-team evals against a live model (flaky; skips if no server)
make run-chat   # text chat: pick/create a resident, chat, /end
make run-voice  # Stage 2 entry point (stub)
```

Copy `.env.example` to `.env` to override configuration (`DUCK_` prefix). The
chat loop talks to a locally served, OpenAI-compatible model endpoint
(`litert-lm serve`, `llama.cpp` server, or Ollama) at `DUCK_LLM_BASE_URL`.

## Status

**Spec 1 (Text brain) — complete.** Typed streaming chat against a local
Gemma 4 E2B endpoint, with the companion-duck persona and per-resident SQLite
memory (profiles, facts, sessions, summaries, transcripts) injected into the
system prompt. `make test` green; persona red-team suite runs via `make eval`.
Next: Stage 2 (voice loop — VAD/STT/TTS).
