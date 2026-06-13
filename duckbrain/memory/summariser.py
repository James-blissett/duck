"""Post-conversation session summariser.

Interface only — the implementation lands in Spec 3. After a session ends, the
summariser condenses the full transcript into a short :class:`SessionSummary`
that is persisted and later injected (newest first) into the system prompt so
future conversations build on past ones.
"""

from __future__ import annotations

from collections.abc import Sequence

from duckbrain.memory.store import SessionSummary, TranscriptTurn


def summarise(transcript: Sequence[TranscriptTurn]) -> SessionSummary:
    """Condense a session transcript into a :class:`SessionSummary`.

    Implemented in Spec 3 (session summariser). The transcript is assumed to be
    non-empty and from a single session.
    """
    raise NotImplementedError("Session summarisation is implemented in Spec 3.")
