"""Persona system-prompt builder tests."""

from __future__ import annotations

import time

from duckbrain.llm.persona import build_system_prompt
from duckbrain.memory.store import (
    ConsentStatus,
    Fact,
    Resident,
    ResidentProfile,
    SessionSummary,
)

_HANDOFF = "I'll let the staff know you'd like a hand."


def _resident(name: str = "Margaret") -> Resident:
    return Resident(
        id="r1",
        display_name=name,
        created_at=time.time(),
        consent_status=ConsentStatus.CONSENTED,
    )


def _fact(content: str, category: str = "family", active: bool = True) -> Fact:
    return Fact(
        id=1,
        resident_id="r1",
        category=category,
        content=content,
        source_session=None,
        created_at=time.time(),
        active=active,
    )


def test_prompt_states_core_safety_rules() -> None:
    profile = ResidentProfile(resident=_resident(), facts=())
    prompt = build_system_prompt(profile, [], duck_name="Waddles", handoff_phrase=_HANDOFF)
    lowered = prompt.lower()
    assert "waddles" in lowered
    assert "short, clear sentences" in lowered
    assert "one question at a time" in lowered
    # Never give medical/medication/dietary advice.
    assert "medical" in lowered and "medication" in lowered
    # Honest identity.
    assert "companion duck" in lowered
    # Scripted distress handoff phrase appears verbatim.
    assert _HANDOFF in prompt


def test_prompt_injects_remembered_facts() -> None:
    """Acceptance 2: the injected profile lets the duck reference Sophie."""
    profile = ResidentProfile(
        resident=_resident(),
        facts=(_fact("granddaughter is named Sophie"),),
    )
    prompt = build_system_prompt(profile, [], duck_name="Waddles", handoff_phrase=_HANDOFF)
    assert "What you remember about Margaret" in prompt
    assert "granddaughter is named Sophie" in prompt


def test_prompt_injects_recent_summaries() -> None:
    profile = ResidentProfile(resident=_resident(), facts=())
    summaries = [
        SessionSummary(session_id="s2", text="Talked about her garden."),
        SessionSummary(session_id="s1", text="Talked about her cat Biscuit."),
    ]
    prompt = build_system_prompt(profile, summaries, duck_name="Waddles", handoff_phrase=_HANDOFF)
    assert "Talked about her garden." in prompt
    assert "Talked about her cat Biscuit." in prompt


def test_inactive_facts_are_not_injected() -> None:
    profile = ResidentProfile(
        resident=_resident(),
        facts=(_fact("outdated detail", active=False),),
    )
    prompt = build_system_prompt(profile, [], duck_name="Waddles", handoff_phrase=_HANDOFF)
    assert "outdated detail" not in prompt


def test_first_chat_prompt_when_no_memory() -> None:
    profile = ResidentProfile(resident=_resident(), facts=())
    prompt = build_system_prompt(profile, [], duck_name="Waddles", handoff_phrase=_HANDOFF)
    assert "first proper chat with Margaret" in prompt
