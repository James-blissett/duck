"""Memory store tests (Spec 1 acceptance criteria 2 and 3)."""

from __future__ import annotations

from pathlib import Path

from duckbrain.memory.store import ConsentStatus, MemoryStore, Role


def test_create_and_get_resident(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.db") as store:
        margaret = store.create_resident("Margaret")
        assert margaret.display_name == "Margaret"
        assert margaret.consent_status is ConsentStatus.CONSENTED

        fetched = store.get_resident(margaret.id)
        assert fetched == margaret
        assert store.find_resident_by_name("Margaret") == margaret
        assert [r.id for r in store.list_residents()] == [margaret.id]


def test_facts_active_flag(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.db") as store:
        margaret = store.create_resident("Margaret")
        fact = store.add_fact(margaret.id, "family", "Has a dog named Rex")
        assert [f.content for f in store.get_active_facts(margaret.id)] == ["Has a dog named Rex"]
        store.deactivate_fact(fact.id)
        assert store.get_active_facts(margaret.id) == []


def test_profile_persists_across_reopen(tmp_path: Path) -> None:
    """Acceptance 2: facts survive an app restart (new MemoryStore, same file).

    Mirrors: create Margaret + facts, /end, restart, insert "granddaughter is
    named Sophie", and the new session can build a profile that includes Sophie.
    """
    db = tmp_path / "memory.db"
    with MemoryStore(db) as store:
        margaret = store.create_resident("Margaret")
        margaret_id = margaret.id
        store.add_fact(margaret_id, "pet", "Has a cat called Biscuit")
        session = store.start_session(margaret_id)
        store.add_turn(session.id, Role.RESIDENT, "Hello there")
        store.end_session(session.id)

    # "Restart": brand new store object against the same file.
    with MemoryStore(db) as store:
        store.add_fact(margaret_id, "family", "granddaughter is named Sophie")
        profile = store.get_profile(margaret_id)
        assert profile is not None
        assert profile.resident.display_name == "Margaret"
        contents = {f.content for f in profile.facts}
        assert "granddaughter is named Sophie" in contents
        assert "Has a cat called Biscuit" in contents


def test_recent_summaries_newest_first_and_limited(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.db") as store:
        margaret = store.create_resident("Margaret")
        for i in range(5):
            session = store.start_session(margaret.id)
            store.end_session(session.id)
            store.add_summary(session.id, f"summary {i}")

        recent = store.get_recent_summaries(margaret.id, limit=3)
        assert [s.text for s in recent] == ["summary 4", "summary 3", "summary 2"]


def test_transcript_round_trip(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.db") as store:
        margaret = store.create_resident("Margaret")
        session = store.start_session(margaret.id)
        store.add_turn(session.id, Role.RESIDENT, "Hello", ts=1.0)
        store.add_turn(session.id, Role.DUCK, "Hello, Margaret", ts=2.0)
        turns = store.get_transcript(session.id)
        assert [(t.role, t.text) for t in turns] == [
            (Role.RESIDENT, "Hello"),
            (Role.DUCK, "Hello, Margaret"),
        ]


def test_delete_resident_completely_leaves_zero_rows(tmp_path: Path) -> None:
    """Acceptance 3: erasure removes every trace across all tables."""
    with MemoryStore(tmp_path / "m.db") as store:
        margaret = store.create_resident("Margaret")
        other = store.create_resident("Arthur")

        store.add_fact(margaret.id, "family", "granddaughter is named Sophie")
        session = store.start_session(margaret.id)
        store.add_turn(session.id, Role.RESIDENT, "Hello")
        store.add_turn(session.id, Role.DUCK, "Hello, Margaret")
        store.add_summary(session.id, "We talked about Sophie.")

        # Sanity: there is data to erase.
        assert store.count_rows_for_resident(margaret.id) > 0

        store.delete_resident_completely(margaret.id)

        assert store.count_rows_for_resident(margaret.id) == 0
        assert store.get_resident(margaret.id) is None
        assert store.get_active_facts(margaret.id) == []
        assert store.get_transcript(session.id) == []
        assert store.get_recent_summaries(margaret.id) == []
        # Other residents are untouched.
        assert store.get_resident(other.id) == other
