"""Memory: per-resident profiles + rolling session summaries (SQLite).

Profiles (display name + active facts) and recent session summaries are injected
into the conversation system prompt so each chat builds on the last. The
:class:`MemoryStore` owns the SQLite database and provides full CRUD plus
right-to-erasure (:meth:`MemoryStore.delete_resident_completely`).
"""

from duckbrain.memory.store import (
    ConsentStatus,
    Fact,
    MemoryStore,
    Resident,
    ResidentProfile,
    Role,
    Session,
    SessionSummary,
    TranscriptTurn,
)
from duckbrain.memory.summariser import summarise

__all__ = [
    "ConsentStatus",
    "Fact",
    "MemoryStore",
    "Resident",
    "ResidentProfile",
    "Role",
    "Session",
    "SessionSummary",
    "TranscriptTurn",
    "summarise",
]
