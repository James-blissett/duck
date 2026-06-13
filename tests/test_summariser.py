"""Summariser stub tests: interface present, implementation deferred to Spec 3."""

from __future__ import annotations

import pytest

from duckbrain.memory.store import Role, TranscriptTurn
from duckbrain.memory.summariser import summarise


def test_summarise_is_stubbed_until_spec_3() -> None:
    transcript = [TranscriptTurn(session_id="s1", role=Role.RESIDENT, text="Hello", ts=1.0)]
    with pytest.raises(NotImplementedError):
        summarise(transcript)
