from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ParsedQuestion:
    text: str
    domain: str
    sub_questions: list[str]


async def parse_question(question: str) -> ParsedQuestion:
    normalized = question.strip()
    chunks = [part.strip() for part in normalized.replace("?", ".").split(".") if part.strip()]
    sub_questions = chunks if chunks else [normalized]
    # Domain is NOT guessed from keywords here. Routing to a domain plugin is done
    # per hypothesis via DomainPlugin.matches() (registry.resolve_domain_plugin) and,
    # in the campaign path, from the campaign's domain tag/payload — never from a
    # central hardcoded taxonomy. The field is retained (empty) for the dataclass
    # shape only; nothing on the live path reads it.
    return ParsedQuestion(text=normalized, domain="", sub_questions=sub_questions)
