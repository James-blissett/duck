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

### Model — zero setup

`make run-chat` is self-contained: by default it talks to Ollama, and on first
run it **starts the local model server and downloads the model (Gemma 3)
automatically** — you don't install or pull anything by hand. The first run
downloads a few GB; later runs only re-download nothing (the model is cached).

The model server is started as a child of the chat process and **stopped when
you exit**, so it never lingers in the background eating memory. (If a server is
already running when you start, the duck reuses it and leaves it running.) To use
a server you manage yourself (`litert-lm serve`, `llama.cpp`), set
`DUCK_LLM_AUTO_BOOTSTRAP=false` and point `DUCK_LLM_BASE_URL` at it.

## Status

**Spec 1 (Text brain) — complete.** Typed streaming chat against a local
Gemma 4 E2B endpoint, with the companion-duck persona and per-resident SQLite
memory (profiles, facts, sessions, summaries, transcripts) injected into the
system prompt. `make test` green; persona red-team suite runs via `make eval`.
Next: Stage 2 (voice loop — VAD/STT/TTS).
