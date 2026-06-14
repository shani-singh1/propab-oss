"""NodeOperatorTrace — per-node operator usage (P1)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from propab.config import settings


def trace_ledger_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "operator_trace_ledger.json"


@dataclass
class OperatorStep:
    family: str
    operator: str
    order: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NodeOperatorTrace:
    campaign_id: str
    node_id: str
    operators_used: list[OperatorStep]
    order: int
    cost: float
    outcome: str
    retrieval: str = "hybrid"
    branching: str = "closure_aware"
    mutation: str = "local_refinement"
    verification: str = "numerical"
    model: str = "default_llm"
    decomposition: str = "confirmed_expand"
    state_vector: list[float] = field(default_factory=list)
    neighbors: list[dict[str, Any]] = field(default_factory=list)
    parent_node_id: str | None = None
    child_node_ids: list[str] = field(default_factory=list)
    expansion_type: str | None = None
    claim_type: str | None = None
    primary_theme: str | None = None
    depth: int = 0
    duration_ms: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    lineage_length: int | None = None
    source: str = "snapshot"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeOperatorTrace:
        steps = [OperatorStep(**s) for s in (data.get("operators_used") or [])]
        return cls(
            campaign_id=str(data["campaign_id"]),
            node_id=str(data["node_id"]),
            operators_used=steps,
            order=int(data.get("order") or 0),
            cost=float(data.get("cost") or 0),
            outcome=str(data.get("outcome") or "pending"),
            retrieval=str(data.get("retrieval") or "hybrid"),
            branching=str(data.get("branching") or "closure_aware"),
            mutation=str(data.get("mutation") or "local_refinement"),
            verification=str(data.get("verification") or "numerical"),
            model=str(data.get("model") or "default_llm"),
            decomposition=str(data.get("decomposition") or "confirmed_expand"),
            state_vector=list(data.get("state_vector") or []),
            neighbors=list(data.get("neighbors") or []),
            parent_node_id=data.get("parent_node_id"),
            child_node_ids=list(data.get("child_node_ids") or []),
            expansion_type=data.get("expansion_type"),
            claim_type=data.get("claim_type"),
            primary_theme=data.get("primary_theme"),
            depth=int(data.get("depth") or 0),
            duration_ms=int(data.get("duration_ms") or 0),
            tool_calls=list(data.get("tool_calls") or []),
            lineage_length=data.get("lineage_length"),
            source=str(data.get("source") or "snapshot"),
        )


@dataclass
class OperatorTraceLedger:
    traces: list[NodeOperatorTrace] = field(default_factory=list)

    def add(self, trace: NodeOperatorTrace) -> None:
        self.traces = [
            t for t in self.traces
            if not (t.campaign_id == trace.campaign_id and t.node_id == trace.node_id)
        ]
        self.traces.append(trace)

    def for_campaign(self, campaign_id: str) -> list[NodeOperatorTrace]:
        return [t for t in self.traces if t.campaign_id == campaign_id]

    def to_dict(self) -> dict[str, Any]:
        return {"traces": [t.to_dict() for t in self.traces], "count": len(self.traces)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorTraceLedger:
        return cls(traces=[NodeOperatorTrace.from_dict(t) for t in (data.get("traces") or [])])

    def save(self, path: Path | None = None) -> Path:
        p = path or trace_ledger_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> OperatorTraceLedger:
        p = path or trace_ledger_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()
