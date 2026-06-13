"""Per-resident memory: profiles, facts, sessions, summaries, transcripts.

Backed by the standard-library :mod:`sqlite3`. A :class:`MemoryStore` owns one
connection and creates its schema on open. Importing this module has no side
effects; construct a store (which opens/creates the database file) to use it.

The longitudinal memory is the project's core differentiator: a resident's
profile (display name + active facts) and the most recent session summaries are
injected into the conversation system prompt so each chat builds on the last.

Right to erasure (PRD): :meth:`MemoryStore.delete_resident_completely` removes
every trace of a resident across all tables.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType


class ConsentStatus(StrEnum):
    """Consent state for a resident profile (PRD consent workflow)."""

    PENDING = "pending"
    CONSENTED = "consented"
    REVOKED = "revoked"


class Role(StrEnum):
    """Speaker role for a stored transcript turn."""

    RESIDENT = "resident"
    DUCK = "duck"


@dataclass(frozen=True)
class Resident:
    """A consented resident with a remembered profile."""

    id: str
    display_name: str
    created_at: float
    consent_status: ConsentStatus


@dataclass(frozen=True)
class Fact:
    """A single remembered fact about a resident (e.g. family, preference)."""

    id: int
    resident_id: str
    category: str
    content: str
    source_session: str | None
    created_at: float
    active: bool


@dataclass(frozen=True)
class Session:
    """One conversation session with a resident."""

    id: str
    resident_id: str
    started_at: float
    ended_at: float | None


@dataclass(frozen=True)
class SessionSummary:
    """A short summary of one session, produced post-conversation (Spec 3)."""

    session_id: str
    text: str


@dataclass(frozen=True)
class TranscriptTurn:
    """A single stored turn of a conversation transcript."""

    session_id: str
    role: Role
    text: str
    ts: float


@dataclass(frozen=True)
class ResidentProfile:
    """A resident plus their active facts — the unit injected into the prompt."""

    resident: Resident
    facts: tuple[Fact, ...]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS residents (
    id             TEXT PRIMARY KEY,
    display_name   TEXT NOT NULL,
    created_at     REAL NOT NULL,
    consent_status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    resident_id    TEXT NOT NULL REFERENCES residents(id),
    category       TEXT NOT NULL,
    content        TEXT NOT NULL,
    source_session TEXT,
    created_at     REAL NOT NULL,
    active         INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    resident_id TEXT NOT NULL REFERENCES residents(id),
    started_at  REAL NOT NULL,
    ended_at    REAL
);

CREATE TABLE IF NOT EXISTS summaries (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    text       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role       TEXT NOT NULL,
    text       TEXT NOT NULL,
    ts         REAL NOT NULL
);
"""


class MemoryStore:
    """SQLite-backed store for resident profiles and conversation history.

    Use as a context manager or call :meth:`close` when done::

        with MemoryStore(path) as store:
            resident = store.create_resident("Margaret")
    """

    def __init__(self, path: Path | str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # --- Residents ---------------------------------------------------------

    def create_resident(
        self,
        display_name: str,
        consent_status: ConsentStatus = ConsentStatus.CONSENTED,
    ) -> Resident:
        resident = Resident(
            id=uuid.uuid4().hex,
            display_name=display_name,
            created_at=time.time(),
            consent_status=consent_status,
        )
        self._conn.execute(
            "INSERT INTO residents (id, display_name, created_at, consent_status)"
            " VALUES (?, ?, ?, ?)",
            (
                resident.id,
                resident.display_name,
                resident.created_at,
                resident.consent_status.value,
            ),
        )
        self._conn.commit()
        return resident

    def get_resident(self, resident_id: str) -> Resident | None:
        row = self._conn.execute("SELECT * FROM residents WHERE id = ?", (resident_id,)).fetchone()
        return _row_to_resident(row) if row is not None else None

    def find_resident_by_name(self, display_name: str) -> Resident | None:
        row = self._conn.execute(
            "SELECT * FROM residents WHERE display_name = ? ORDER BY created_at LIMIT 1",
            (display_name,),
        ).fetchone()
        return _row_to_resident(row) if row is not None else None

    def list_residents(self) -> list[Resident]:
        rows = self._conn.execute("SELECT * FROM residents ORDER BY created_at").fetchall()
        return [_row_to_resident(row) for row in rows]

    def set_consent_status(self, resident_id: str, status: ConsentStatus) -> None:
        self._conn.execute(
            "UPDATE residents SET consent_status = ? WHERE id = ?",
            (status.value, resident_id),
        )
        self._conn.commit()

    # --- Facts -------------------------------------------------------------

    def add_fact(
        self,
        resident_id: str,
        category: str,
        content: str,
        source_session: str | None = None,
    ) -> Fact:
        created_at = time.time()
        cur = self._conn.execute(
            "INSERT INTO facts (resident_id, category, content, source_session,"
            " created_at, active) VALUES (?, ?, ?, ?, ?, 1)",
            (resident_id, category, content, source_session, created_at),
        )
        self._conn.commit()
        fact_id = cur.lastrowid
        assert fact_id is not None
        return Fact(
            id=fact_id,
            resident_id=resident_id,
            category=category,
            content=content,
            source_session=source_session,
            created_at=created_at,
            active=True,
        )

    def get_active_facts(self, resident_id: str) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE resident_id = ? AND active = 1 ORDER BY created_at",
            (resident_id,),
        ).fetchall()
        return [_row_to_fact(row) for row in rows]

    def deactivate_fact(self, fact_id: int) -> None:
        self._conn.execute("UPDATE facts SET active = 0 WHERE id = ?", (fact_id,))
        self._conn.commit()

    def get_profile(self, resident_id: str) -> ResidentProfile | None:
        resident = self.get_resident(resident_id)
        if resident is None:
            return None
        return ResidentProfile(
            resident=resident,
            facts=tuple(self.get_active_facts(resident_id)),
        )

    # --- Sessions ----------------------------------------------------------

    def start_session(self, resident_id: str) -> Session:
        session = Session(
            id=uuid.uuid4().hex,
            resident_id=resident_id,
            started_at=time.time(),
            ended_at=None,
        )
        self._conn.execute(
            "INSERT INTO sessions (id, resident_id, started_at, ended_at) VALUES (?, ?, ?, NULL)",
            (session.id, session.resident_id, session.started_at),
        )
        self._conn.commit()
        return session

    def end_session(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return _row_to_session(row) if row is not None else None

    # --- Summaries ---------------------------------------------------------

    def add_summary(self, session_id: str, text: str) -> SessionSummary:
        self._conn.execute(
            "INSERT INTO summaries (session_id, text) VALUES (?, ?)",
            (session_id, text),
        )
        self._conn.commit()
        return SessionSummary(session_id=session_id, text=text)

    def get_recent_summaries(self, resident_id: str, limit: int = 3) -> list[SessionSummary]:
        """Most recent session summaries for a resident, newest first."""
        rows = self._conn.execute(
            "SELECT s.session_id AS session_id, s.text AS text"
            " FROM summaries s JOIN sessions sess ON s.session_id = sess.id"
            " WHERE sess.resident_id = ?"
            " ORDER BY sess.started_at DESC LIMIT ?",
            (resident_id, limit),
        ).fetchall()
        return [SessionSummary(session_id=row["session_id"], text=row["text"]) for row in rows]

    # --- Transcript turns --------------------------------------------------

    def add_turn(
        self,
        session_id: str,
        role: Role,
        text: str,
        ts: float | None = None,
    ) -> TranscriptTurn:
        turn = TranscriptTurn(
            session_id=session_id,
            role=role,
            text=text,
            ts=time.time() if ts is None else ts,
        )
        self._conn.execute(
            "INSERT INTO transcript_turns (session_id, role, text, ts) VALUES (?, ?, ?, ?)",
            (turn.session_id, turn.role.value, turn.text, turn.ts),
        )
        self._conn.commit()
        return turn

    def get_transcript(self, session_id: str) -> list[TranscriptTurn]:
        rows = self._conn.execute(
            "SELECT * FROM transcript_turns WHERE session_id = ? ORDER BY ts, id",
            (session_id,),
        ).fetchall()
        return [_row_to_turn(row) for row in rows]

    # --- Right to erasure --------------------------------------------------

    def delete_resident_completely(self, resident_id: str) -> None:
        """Remove every trace of a resident across all tables (PRD erasure)."""
        cur = self._conn.cursor()
        session_ids = [
            row["id"]
            for row in cur.execute(
                "SELECT id FROM sessions WHERE resident_id = ?", (resident_id,)
            ).fetchall()
        ]
        for session_id in session_ids:
            cur.execute("DELETE FROM transcript_turns WHERE session_id = ?", (session_id,))
            cur.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
        cur.execute("DELETE FROM sessions WHERE resident_id = ?", (resident_id,))
        cur.execute("DELETE FROM facts WHERE resident_id = ?", (resident_id,))
        cur.execute("DELETE FROM residents WHERE id = ?", (resident_id,))
        self._conn.commit()

    def count_rows_for_resident(self, resident_id: str) -> int:
        """Total rows referencing a resident across all tables (test helper)."""
        cur = self._conn.cursor()
        total = cur.execute(
            "SELECT COUNT(*) FROM residents WHERE id = ?", (resident_id,)
        ).fetchone()[0]
        total += cur.execute(
            "SELECT COUNT(*) FROM facts WHERE resident_id = ?", (resident_id,)
        ).fetchone()[0]
        total += cur.execute(
            "SELECT COUNT(*) FROM sessions WHERE resident_id = ?", (resident_id,)
        ).fetchone()[0]
        total += cur.execute(
            "SELECT COUNT(*) FROM summaries WHERE session_id IN"
            " (SELECT id FROM sessions WHERE resident_id = ?)",
            (resident_id,),
        ).fetchone()[0]
        total += cur.execute(
            "SELECT COUNT(*) FROM transcript_turns WHERE session_id IN"
            " (SELECT id FROM sessions WHERE resident_id = ?)",
            (resident_id,),
        ).fetchone()[0]
        return int(total)


def _row_to_resident(row: sqlite3.Row) -> Resident:
    return Resident(
        id=row["id"],
        display_name=row["display_name"],
        created_at=row["created_at"],
        consent_status=ConsentStatus(row["consent_status"]),
    )


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        id=row["id"],
        resident_id=row["resident_id"],
        category=row["category"],
        content=row["content"],
        source_session=row["source_session"],
        created_at=row["created_at"],
        active=bool(row["active"]),
    )


def _row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        resident_id=row["resident_id"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
    )


def _row_to_turn(row: sqlite3.Row) -> TranscriptTurn:
    return TranscriptTurn(
        session_id=row["session_id"],
        role=Role(row["role"]),
        text=row["text"],
        ts=row["ts"],
    )
