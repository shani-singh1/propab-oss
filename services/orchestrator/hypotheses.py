from __future__ import annotations

from dataclasses import dataclass
import json

from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.types import EventType
from services.orchestrator.intake import ParsedQuestion
from services.orchestrator.schemas import Prior


@dataclass(slots=True)
class RankedHypothesis:
    id: str
    text: str
    test_methodology: str
    scores: dict[str, float]
    rank: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "test_methodology": self.test_methodology,
            "scores": self.scores,
            "rank": self.rank,
        }


def _build_hypothesis_prompt(parsed: ParsedQuestion, prior: Prior, max_hypotheses: int) -> str:
    return f"""
You are a research hypothesis generator.

Research question: {parsed.text}

Prior established facts:
{json.dumps(prior.established_facts)}

Prior open gaps:
{json.dumps(prior.open_gaps)}

Prior dead ends:
{json.dumps(prior.dead_ends)}

Generate exactly {max_hypotheses} hypotheses.
Return JSON array only with fields: id, text, test_methodology, gap_reference, expected_result.
"""


async def generate_ranked_hypotheses(
    parsed: ParsedQuestion,
    prior: Prior,
    max_hypotheses: int,
    llm: LLMClient,
    session_id: str,
    emitter: EventEmitter,
) -> list[RankedHypothesis]:
    prompt = _build_hypothesis_prompt(parsed, prior, max_hypotheses)
    raw = await llm.call(prompt=prompt, purpose="hypothesis_generation", session_id=session_id)
    try:
        generated = json.loads(raw)
    except json.JSONDecodeError:
        generated = []

    if isinstance(generated, list):
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.HYPO_GENERATED,
            step="hypothesis.generate",
            payload={"hypotheses": generated},
        )
    else:
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.HYPO_GENERATED,
            step="hypothesis.generate",
            payload={"hypotheses": [], "note": "Model returned non-array JSON; falling back to templates."},
        )

    hypotheses: list[RankedHypothesis] = []
    for idx in range(max_hypotheses):
        rank = idx + 1
        composite = round(max(0.15, 1.0 - idx * 0.12), 3)
        gen_list = generated if isinstance(generated, list) else []
        entry = gen_list[idx] if idx < len(gen_list) and isinstance(gen_list[idx], dict) else {}
        hypotheses.append(
            RankedHypothesis(
                id=str(entry.get("id", f"h{rank}")),
                text=str(entry.get("text", f"Hypothesis {rank}: A measurable intervention improves outcomes for '{parsed.text}'.")),
                test_methodology=str(
                    entry.get(
                        "test_methodology",
                        "Run a controlled experiment and compare baseline vs intervention metrics.",
                    )
                ),
                scores={
                    "novelty": round(max(0.2, composite - 0.1), 3),
                    "testability": round(max(0.3, composite), 3),
                    "impact": round(max(0.25, composite - 0.05), 3),
                    "scope_fit": round(max(0.2, composite - 0.08), 3),
                    "composite": composite,
                },
                rank=rank,
            )
        )
    return hypotheses
