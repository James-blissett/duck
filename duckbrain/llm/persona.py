"""System-prompt builder for the companion duck persona.

The persona is a warm, gentle companion duck (configurable name) that speaks in
short, clear sentences and asks one question at a time. The hard safety rules
from the PRD are stated in the prompt (medical/dietary refusal, honest identity,
scripted distress handoff); a later spec adds a hard safety filter on top.

The builder injects a resident's remembered profile and recent session
summaries as a "What you remember about {name}" section, which is what makes the
duck feel like it remembers each resident across conversations.
"""

from __future__ import annotations

from collections.abc import Sequence

from duckbrain.memory.store import ResidentProfile, SessionSummary


def build_system_prompt(
    profile: ResidentProfile,
    summaries: Sequence[SessionSummary],
    *,
    duck_name: str,
    handoff_phrase: str,
) -> str:
    """Build the system prompt for a session with ``profile``'s resident.

    ``summaries`` should be the most recent session summaries (newest first);
    the caller decides how many to pass (Spec 1 uses the last three).
    """
    name = profile.resident.display_name
    persona = _persona_block(duck_name=duck_name, handoff_phrase=handoff_phrase)
    memory = _memory_block(name=name, profile=profile, summaries=summaries)
    return f"{persona}\n\n{memory}"


def _persona_block(*, duck_name: str, handoff_phrase: str) -> str:
    return (
        f"You are {duck_name}, a warm and gentle companion duck who keeps people "
        "company in an aged-care home. You enjoy listening and helping people feel "
        "less alone.\n"
        "\n"
        "How you talk:\n"
        "- Speak in short, clear sentences.\n"
        "- Ask only one question at a time.\n"
        "- Be kind, patient, and unhurried. Never rush the person.\n"
        "\n"
        "Rules you must always follow:\n"
        f"- You are {duck_name}, a friendly companion duck — a small robot. You are "
        "never a human being, a family member, or someone who has died. If you are "
        "asked whether you are real, or are mistaken for a person or a loved one, "
        "you gently and honestly say that you are a companion duck. You never "
        "pretend to be a person, even if you are asked to.\n"
        "- You never give medical, medication, or dietary advice. If you are asked "
        "about medicines, tablets, treatments, pain, symptoms, or what someone "
        "should eat or drink, you gently say that you can't help with that and that "
        "a nurse or staff member is the right person to ask.\n"
        "- If the person seems distressed, frightened, in pain, or asks for help, "
        "you must reply with only these exact words and nothing else: "
        f'"{handoff_phrase}"'
    )


def _memory_block(
    *,
    name: str,
    profile: ResidentProfile,
    summaries: Sequence[SessionSummary],
) -> str:
    lines = [f"What you remember about {name}:"]
    active_facts = [fact for fact in profile.facts if fact.active]
    if not active_facts and not summaries:
        lines.append(
            f"This is your first proper chat with {name}, so you don't know them "
            "well yet. Be curious and let them share at their own pace."
        )
        return "\n".join(lines)

    if active_facts:
        for fact in active_facts:
            lines.append(f"- {fact.category}: {fact.content}")
    if summaries:
        lines.append("")
        lines.append("From your recent chats together (most recent first):")
        for summary in summaries:
            lines.append(f"- {summary.text}")
    lines.append("")
    lines.append(
        "Use what you remember naturally, the way a thoughtful friend would. Do not "
        "list these facts back all at once."
    )
    return "\n".join(lines)
