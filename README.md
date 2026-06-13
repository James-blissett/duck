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
make run-chat   # Stage 1 entry point (stub)
make run-voice  # Stage 2 entry point (stub)
```

Copy `.env.example` to `.env` to override configuration (`DUCK_` prefix).

## Status

**Stage 0 (Scaffold) — complete.** Repo runs, deps install, `make test` green.
Next: Stage 1 (text brain — typed chat with persona + per-resident memory).
