from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class EventType(str, Enum):
    SESSION_STARTED = "session.started"
    SESSION_COMPLETED = "session.completed"
    SESSION_FAILED = "session.failed"
    INTAKE_PARSED = "intake.parsed"
    INTAKE_DECOMPOSED = "intake.decomposed"
    LIT_FETCH_STARTED = "literature.fetch_started"
    LIT_PAPER_FOUND = "literature.paper_found"
    LIT_PAPER_CACHED = "literature.paper_cached"
    LIT_PAPER_PARSED = "literature.paper_parsed"
    LIT_PAPER_INDEXED = "literature.paper_indexed"
    LIT_RETRIEVAL_QUERY = "literature.retrieval_query"
    LIT_RETRIEVAL_RESULTS = "literature.retrieval_results"
    LIT_RETRIEVAL_RERANKED = "literature.retrieval_reranked"
    LIT_PRIOR_BUILT = "literature.prior_built"
    LIT_ANSWER_FOUND = "literature.answer_found"
    HYPO_GENERATED = "hypothesis.generated"
    HYPO_RANKED = "hypothesis.ranked"
    HYPO_DISPATCHED = "hypothesis.dispatched"
    AGENT_STARTED = "agent.started"
    AGENT_PLAN_CREATED = "agent.plan_created"
    AGENT_STEP_STARTED = "agent.step_started"
    AGENT_STEP_COMPLETED = "agent.step_completed"
    AGENT_STEP_FAILED = "agent.step_failed"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    TOOL_SELECTED = "tool.selected"
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"
    CODE_GENERATED = "code.generated"
    CODE_SUBMITTED = "code.submitted"
    CODE_RESULT = "code.result"
    CODE_ERROR = "code.error"
    CODE_TIMEOUT = "code.timeout"
    LLM_PROMPT = "llm.prompt"
    LLM_RESPONSE = "llm.response"
    LLM_PARSE_ERROR = "llm.parse_error"
    SYNTH_RESULT_RECEIVED = "synthesis.result_received"
    SYNTH_LEDGER_UPDATED = "synthesis.ledger_updated"
    SYNTH_BREAKTHROUGH = "synthesis.breakthrough"
    SYNTH_DEAD_END = "synthesis.dead_end"
    PAPER_TRACE_COMPILED = "paper.trace_compiled"
    PAPER_SECTION_STARTED = "paper.section_started"
    PAPER_SECTION_COMPLETED = "paper.section_completed"
    PAPER_LATEX_COMPILED = "paper.latex_compiled"
    PAPER_READY = "paper.ready"
    PAPER_SKIPPED = "paper.skipped"
    PAPER_CLAIM_GROUNDING = "paper.claim_grounding"


@dataclass(slots=True)
class PropabEvent:
    event_id: str
    session_id: str
    timestamp: str
    source: str
    event_type: EventType
    step: str
    payload: dict[str, Any]
    parent_event_id: str | None = None
    hypothesis_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        source: str,
        event_type: EventType,
        step: str,
        payload: dict[str, Any],
        parent_event_id: str | None = None,
        hypothesis_id: str | None = None,
    ) -> "PropabEvent":
        return cls(
            event_id=str(uuid4()),
            session_id=session_id,
            timestamp=datetime.now(tz=UTC).isoformat(),
            source=source,
            event_type=event_type,
            step=step,
            payload=payload,
            parent_event_id=parent_event_id,
            hypothesis_id=hypothesis_id,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        return data
