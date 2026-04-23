from __future__ import annotations

from dataclasses import dataclass

from .question_domain import infer_session_domain


@dataclass(slots=True)
class ParsedQuestion:
    text: str
    domain: str
    sub_questions: list[str]


async def parse_question(question: str) -> ParsedQuestion:
    normalized = question.strip()
    chunks = [part.strip() for part in normalized.replace("?", ".").split(".") if part.strip()]
    sub_questions = chunks if chunks else [normalized]
    domain = infer_session_domain(normalized)
    return ParsedQuestion(text=normalized, domain=domain, sub_questions=sub_questions)
