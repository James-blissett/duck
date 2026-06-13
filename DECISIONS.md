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

