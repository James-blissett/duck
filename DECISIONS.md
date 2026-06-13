Where I will write the reasoning and logic behind project decisions, for later reference

## Spec 1 — Text conversation engine

- **LLM transport = OpenAI-compatible `/v1/chat/completions` + SSE.** The spec
  wants the backend swappable across `litert-lm serve`, llama.cpp server, and
  Ollama. The one wire format all three share is the OpenAI chat-completions
  endpoint with `stream: true` SSE, so `OpenAICompatibleClient` targets that.
  The parlor reference drives LiteRT-LM via its in-process Python engine, but an
  HTTP endpoint is the only common denominator, hence the choice. Anything
  implementing the `LLMClient` protocol (`generate_stream(messages) ->
  AsyncIterator[str]`) can be dropped in, including the test fakes.
- **`httpx`** added as the async HTTP client (streaming + `MockTransport` makes
  SSE parsing unit-testable without a live server).
- **Persona safety rules are prompt-level only in Spec 1.** The PRD requires a
  *hard* filter (medical/dietary refusal, distress handoff, identity honesty);
  that hard safety layer (`duckbrain/safety/`) is a later spec. Here the rules
  live in the system prompt, and the red-team eval suite checks the live model
  honours them.
- **Red-team tests are an eval suite, not unit tests.** Marked `eval`,
  deselected by default (`addopts = -m 'not eval'`), run via `make eval`. They
  skip (not fail) when no model server is reachable, so they never break CI but
  can be run on demand.
- **Memory data models live in `memory/store.py`** (frozen dataclasses +
  `ConsentStatus`/`Role` enums), re-exported from `duckbrain.memory`. `persona`
  imports `ResidentProfile`/`SessionSummary` from there; no extra models module.
- **`delete_resident_completely`** deletes children (transcript turns, summaries)
  by the resident's session ids first, then sessions/facts/residents — covered by
  a test asserting zero rows across every table (right-to-erasure).

## Zero-touch model bootstrap (follow-up to Spec 1)

- **Default backend is Ollama**, serving **Gemma 3** (`gemma3:4b`), at
  `http://127.0.0.1:11434`. Chosen because it runs Gemma 3 on Apple Silicon with
  Metal out of the box and exposes the OpenAI-compatible endpoint our client
  already speaks. (Production target on the Jetson remains LiteRT-LM; swap via
  `DUCK_LLM_BASE_URL`/`DUCK_LLM_MODEL` + `DUCK_LLM_AUTO_BOOTSTRAP=false`.)
- **`duckbrain/llm/bootstrap.py` makes startup zero-touch**: finds the `ollama`
  binary (explicit path → PATH → vendored), starts `ollama serve` if nothing is
  listening, and pulls the model via `/api/pull` with streamed progress if it is
  not present. Gated by `DUCK_LLM_AUTO_BOOTSTRAP` so it never interferes with a
  self-managed endpoint.
- **Ollama vendored headlessly** at `~/.local/share/duck/ollama/` from the
  official `ollama-darwin.tgz` (no Homebrew, no GUI app, no admin) so the
  operator installs nothing.
- **The spawned server is not a daemon.** `_start_server` runs `ollama serve` as
  a child of the CLI (no `start_new_session`), and `ensure_ready` returns the
  process so the CLI stops it in a `finally` on exit (`stop_server`). This trades
  a slower first message each run (model reloads into the GPU) for the guarantee
  that no multi-GB model server lingers in the background. We only stop a server
  we started — if one was already running, `ensure_ready` returns `None` and
  leaves it untouched.
- The model download is the only first-run cost (~3.3 GB for `gemma3:4b`); the
  binary discovery and pull-progress parsing are unit-tested with
  `httpx.MockTransport` (no network).

