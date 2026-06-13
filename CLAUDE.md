# Companion Duck — Brain

**Current stage: Spec 1 (Text brain) — complete.** Update this line as you progress.

## Project purpose

Loneliness and social isolation are endemic in residential aged care, and staff
rarely have time for sustained one-on-one conversation — loneliness is a recorded
quality indicator in Australian aged care assessment frameworks. The Companion
Duck is a small, friendly biped robot (built on Open Duck Mini v2) that holds
natural voice conversations with residents and **remembers them** — their family,
their stories, their preferences — so each conversation builds on the last. The
core differentiator is longitudinal conversational memory per resident: a duck
that remembers your granddaughter's name beats a smarter duck that doesn't.

All inference runs on-device; no resident voice data leaves the facility network.
The primary users are aged-care residents; the secondary users are lifestyle/
activities staff, who deploy, supervise, and configure the duck. This repo is the
**brain** (off-duck, on a Jetson Orin Nano): VAD → STT → conversation engine →
safety layer → TTS, plus per-resident memory in SQLite. The **body** (duck /
Raspberry Pi) runs the stock Open Duck Mini runtime plus a thin expression
client. Brain and body talk over a local WebSocket bus (see the message contract
below).

## Architecture

```
┌─────────────────────── BRAIN (Jetson Orin Nano) ───────────────────────┐
│  Mic in ─► VAD (Silero) ─► STT (Parakeet/onnx_asr) ─┐                   │
│                                                     ▼                   │
│   Memory (SQLite) ◄── Session summariser ◄── Conversation engine        │
│      │                  (async, post-conv)     (Gemma 4 E2B)            │
│      └── profile + summaries ─► system prompt ─┘     │                  │
│                                                      ▼                  │
│                              Safety layer ─► TTS (Kokoro, streamed)     │
│                                   │                  │                  │
│                                   ▼                  ▼                  │
│                          Expression events     Speaker out             │
└──────────────────────────────────│─────────────────────────────────────┘
                                    │  JSON over WebSocket (local Wi-Fi)
┌───────────────────────────────────▼──── BODY (Duck / Raspberry Pi) ─────┐
│  Open_Duck_Mini_Runtime (stock): walking policy, servos, IMU            │
│  + thin expression client: head/antenna motion, LED eyes, attention pose│
└──────────────────────────────────────────────────────────────────────────┘
```

## Message contract (brain ⇄ body)

Defined in `duckbrain/bus/schemas.py`. **This is the stable wire contract — keep
it stable.** Every message carries a `ts` (unix seconds) and a `type` tag.

| Message | Fields |
| --- | --- |
| `UtteranceHeard` | `session_id: str`, `text: str`, `confidence: float`, `ts: float` |
| `ReplyChunk` | `session_id: str`, `text: str`, `is_final: bool`, `ts: float` |
| `ExpressionEvent` | `kind: idle\|listening\|thinking\|speaking\|happy\|concerned\|alert_staff`, `intensity: float`, `ts: float` |
| `SessionControl` | `action: start\|end\|pause`, `resident_id: str\|None`, `ts: float` |

Transport: `duckbrain/bus/broker.py` (topic pub/sub over WebSocket) and
`duckbrain/bus/client.py`. The broker routes by topic and is payload-agnostic, so
schemas can evolve without broker changes. Clients publish/receive typed
`BusMessage` values.

## Coding standards

- Python 3.11, **fully typed**; `mypy` runs in `--strict` mode and must pass.
- Formatted and linted with `ruff` (`make lint` = `ruff check` + `ruff format
  --check` + `mypy`).
- Write `pytest` tests for every acceptance criterion **before** implementing.
- Modules are importable and side-effect-free; executable entry points live in
  `apps/`.
- Config is environment-driven via pydantic-settings (`DUCK_` prefix, `.env`);
  see `duckbrain/config.py`. Config handles: model paths, audio device IDs, brain
  host/port, data directory.
- uv-managed (`make setup` = `uv sync`). No features beyond the current spec.

## Hard safety requirements (verbatim from the PRD)

- The duck never gives medical, medication, or dietary advice. Hard-filtered, not
  just prompted.
- Distress detection: if a resident expresses distress, pain, or asks for help,
  the duck gives a fixed, scripted response ("I'll let the staff know you'd like a
  hand") and flags staff. It never improvises in these moments.
- Honest identity: the duck never claims to be human, a family member, or a
  deceased person, even if asked or confused for one. Critical for residents with
  dementia.
- All conversations logged locally; staff/family can review and delete a
  resident's entire profile on request.
- Consent workflow: resident or their decision-maker consents before a profile is
  created. A "guest mode" with no memory exists for everyone else.

## Layout

```
duckbrain/   audio/ llm/ memory/ safety/ bus/ config.py   (importable library)
apps/        chat_cli.py voice_loop.py staff_panel/        (entry points)
tests/                                                     (pytest)
Makefile     setup / test / lint / format / run-chat / run-voice
```
