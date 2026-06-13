"""Spec 1 entry point: typed text chat with persona + per-resident memory.

Pick or create a resident, then chat. The resident's profile (display name +
active facts) and last three session summaries are injected into the system
prompt at session start, so the duck remembers them across conversations.
Replies stream token-by-token. ``/end`` closes the session (storing the full
transcript) and exits.

This is an executable entry point; the importable library lives in
``duckbrain``. Run with ``make run-chat``.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

import httpx

from duckbrain.config import Settings, get_settings
from duckbrain.llm.bootstrap import BootstrapError, PullProgress, ensure_ready, stop_server
from duckbrain.llm.client import ChatMessage, LLMClient, OpenAICompatibleClient
from duckbrain.llm.persona import build_system_prompt
from duckbrain.memory.store import MemoryStore, Resident, Role

_END_COMMAND = "/end"


def _render_pull_progress(progress: PullProgress) -> None:
    """Render model-download progress on a single updating line."""
    percent = progress.percent
    if percent is None:
        sys.stdout.write(f"\r  {progress.status}".ljust(60))
    else:
        sys.stdout.write(f"\r  {progress.status}: {percent:5.1f}%".ljust(60))
    sys.stdout.flush()


async def _bootstrap_model(
    settings: Settings,
) -> tuple[bool, subprocess.Popen[bytes] | None]:
    """Make the model server + model ready.

    Returns ``(ok, server)`` where ``server`` is the process this run started (to
    be stopped on exit) or ``None`` if a server was already running.
    """
    if not settings.llm_auto_bootstrap:
        return True, None
    try:
        server = await ensure_ready(
            settings.llm_base_url,
            settings.llm_model,
            ollama_bin=settings.ollama_bin,
            on_status=print,
            on_progress=_render_pull_progress,
        )
    except BootstrapError as exc:
        print(f"\n{exc}")
        return False, None
    else:
        print()  # finish any progress line
        return True, server


def _prompt(text: str) -> str:
    """Read a line from stdin with a prompt, treating EOF as ``/end``."""
    try:
        return input(text)
    except EOFError:
        return _END_COMMAND


def _select_resident(store: MemoryStore) -> Resident:
    """Pick an existing resident or create a new one."""
    residents = store.list_residents()
    if residents:
        print("Residents:")
        for index, resident in enumerate(residents, start=1):
            print(f"  {index}. {resident.display_name}")
        print("  n. new resident")
        choice = _prompt("Choose a number (or 'n' for new): ").strip().lower()
        if choice not in {"n", "new", ""}:
            try:
                picked = residents[int(choice) - 1]
            except (ValueError, IndexError):
                print("Didn't understand that — creating a new resident instead.")
            else:
                return picked

    name = _prompt("New resident's name: ").strip()
    while not name:
        name = _prompt("Please enter a name: ").strip()
    resident = store.create_resident(name)
    print(f"Created profile for {resident.display_name}.")
    return resident


async def _run_session(
    store: MemoryStore,
    client: LLMClient,
    settings: Settings,
    resident: Resident,
) -> None:
    """Run one interactive chat session for ``resident``."""
    session = store.start_session(resident.id)
    profile = store.get_profile(resident.id)
    assert profile is not None  # resident was just created/selected
    summaries = store.get_recent_summaries(resident.id, limit=3)
    system_prompt = build_system_prompt(
        profile,
        summaries,
        duck_name=settings.duck_name,
        handoff_phrase=settings.handoff_phrase,
    )

    messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]
    print(f"\nChatting with {settings.duck_name}. Type '{_END_COMMAND}' to finish.\n")

    while True:
        user_text = _prompt(f"{resident.display_name}: ").strip()
        if user_text == _END_COMMAND:
            break
        if not user_text:
            continue

        store.add_turn(session.id, Role.RESIDENT, user_text)
        messages.append({"role": "user", "content": user_text})

        print(f"{settings.duck_name}: ", end="", flush=True)
        reply_parts: list[str] = []
        try:
            async for chunk in client.generate_stream(messages):
                print(chunk, end="", flush=True)
                reply_parts.append(chunk)
        except httpx.HTTPError as exc:
            print(
                f"\n[Could not reach the model at {settings.llm_base_url}: {exc}."
                " Is the model server running?]"
            )
            # Drop the unanswered resident turn from the LLM context.
            messages.pop()
            continue
        print()

        reply = "".join(reply_parts)
        store.add_turn(session.id, Role.DUCK, reply)
        messages.append({"role": "assistant", "content": reply})

    store.end_session(session.id)
    print(f"\nSession saved. Goodbye from {settings.duck_name}.")


async def _main_async() -> None:
    settings = get_settings()
    ok, server = await _bootstrap_model(settings)
    if not ok:
        return
    client = OpenAICompatibleClient(base_url=settings.llm_base_url, model=settings.llm_model)
    try:
        with MemoryStore(settings.db_path) as store:
            resident = _select_resident(store)
            await _run_session(store, client, settings, resident)
    finally:
        await client.aclose()
        # Stop the model server we started so it never lingers in the background.
        if server is not None:
            stop_server(server)
            print("Stopped the local model server.")


def main() -> None:
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
