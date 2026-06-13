"""Schema contract tests for the brain<->body messages."""

from __future__ import annotations

from pydantic import TypeAdapter

from duckbrain.bus.schemas import (
    BusMessage,
    ExpressionEvent,
    ExpressionKind,
    ReplyChunk,
    SessionAction,
    SessionControl,
    UtteranceHeard,
)

_ADAPTER: TypeAdapter[BusMessage] = TypeAdapter(BusMessage)


def test_messages_carry_a_default_timestamp() -> None:
    event = ExpressionEvent(kind=ExpressionKind.HAPPY, intensity=0.8)
    assert event.ts > 0


def test_utterance_heard_roundtrips() -> None:
    msg = UtteranceHeard(session_id="s1", text="hello", confidence=0.92)
    restored = UtteranceHeard.model_validate_json(msg.model_dump_json())
    assert restored == msg


def test_reply_chunk_fields() -> None:
    chunk = ReplyChunk(session_id="s1", text="hi there", is_final=False)
    assert chunk.is_final is False
    assert chunk.type == "ReplyChunk"


def test_session_control_allows_null_resident() -> None:
    ctrl = SessionControl(action=SessionAction.START, resident_id=None)
    assert ctrl.resident_id is None
    assert ctrl.action == "start"


def test_discriminated_union_parses_by_type_tag() -> None:
    raw = ExpressionEvent(kind=ExpressionKind.ALERT_STAFF, intensity=1.0).model_dump_json()
    parsed = _ADAPTER.validate_json(raw)
    assert isinstance(parsed, ExpressionEvent)
    assert parsed.kind is ExpressionKind.ALERT_STAFF
