.PHONY: setup test eval lint format run-chat run-voice

# Stage 0 entry points. Models are NOT downloaded in this stage.

setup:
	uv sync

test:
	uv run pytest

# Persona red-team evals against a live model (flaky; skips if no server).
eval:
	uv run pytest -m eval

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff format .
	uv run ruff check --fix .

run-chat:
	uv run python -m apps.chat_cli

run-voice:
	uv run python -m apps.voice_loop
