"""Brain <-> body message contract.

These pydantic models are the stable wire contract between the brain (Jetson)
and the body (duck / Raspberry Pi). Keep them stable: changing a field is a
breaking change for both sides. Every model carries a ``ts`` unix timestamp.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field


def _now() -> float:
    """Current unix time in seconds (wall clock)."""
    return time.time()


class ExpressionKind(StrEnum):
    """Body-facing expression states driven by conversation state."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    HAPPY = "happy"
    CONCERNED = "concerned"
    ALERT_STAFF = "alert_staff"


class SessionAction(StrEnum):
    """Lifecycle actions for a conversation session."""

    START = "start"
    END = "end"
    PAUSE = "pause"


class UtteranceHeard(BaseModel):
    """A transcribed resident utterance, emitted by the STT stage."""

    type: Literal["UtteranceHeard"] = "UtteranceHeard"
    session_id: str
    text: str
    confidence: float
    ts: float = Field(default_factory=_now)


class ReplyChunk(BaseModel):
    """A streamed chunk of the duck's spoken reply."""

    type: Literal["ReplyChunk"] = "ReplyChunk"
    session_id: str
    text: str
    is_final: bool
    ts: float = Field(default_factory=_now)


class ExpressionEvent(BaseModel):
    """A body expression cue (head/antenna/LED), driven by conversation state."""

    type: Literal["ExpressionEvent"] = "ExpressionEvent"
    kind: ExpressionKind
    intensity: float
    ts: float = Field(default_factory=_now)


class SessionControl(BaseModel):
    """Session lifecycle control, including which resident profile is active."""

    type: Literal["SessionControl"] = "SessionControl"
    action: SessionAction
    resident_id: str | None = None
    ts: float = Field(default_factory=_now)


# Discriminated union of everything that may travel over the bus.
BusMessage: TypeAlias = Annotated[
    UtteranceHeard | ReplyChunk | ExpressionEvent | SessionControl,
    Field(discriminator="type"),
]
