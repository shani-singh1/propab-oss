"""Explicit operator registry — every campaign action records operator choices (P0, P6)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class OperatorFamily(str, Enum):
    BRANCHING = "branching"
    MUTATION = "mutation"
    RETRIEVAL = "retrieval"
    VERIFICATION = "verification"
    MODEL = "model"
    DECOMPOSITION = "decomposition"


# P6 — operator families and instances
OPERATOR_FAMILIES: dict[OperatorFamily, tuple[str, ...]] = {
    OperatorFamily.BRANCHING: ("breadth_first", "depth_first", "closure_aware"),
    OperatorFamily.MUTATION: ("local_refinement", "contradiction", "boundary"),
    OperatorFamily.RETRIEVAL: ("bm25", "semantic", "hybrid"),
    OperatorFamily.VERIFICATION: ("symbolic", "numerical", "simulation"),
    OperatorFamily.MODEL: ("default_llm", "fast_llm", "reasoning_llm"),
    OperatorFamily.DECOMPOSITION: ("confirmed_expand", "refuted_expand", "inconclusive_retest"),
}

DEFAULT_OPERATORS: dict[OperatorFamily, str] = {
    OperatorFamily.BRANCHING: "closure_aware",
    OperatorFamily.MUTATION: "local_refinement",
    OperatorFamily.RETRIEVAL: "hybrid",
    OperatorFamily.VERIFICATION: "numerical",
    OperatorFamily.MODEL: "default_llm",
    OperatorFamily.DECOMPOSITION: "confirmed_expand",
}


@dataclass
class OperatorChoice:
    family: str
    operator: str
    model: str = "default_llm"
    decomposition: str = "confirmed_expand"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class CampaignOperatorSequence:
    """Campaign as a sequence of operator choices (P0)."""
    campaign_id: str
    choices: list[OperatorChoice] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "choices": [c.to_dict() for c in self.choices],
            "node_ids": self.node_ids,
        }


@dataclass
class OperatorRegistry:
    """Canonical registry of valid operators per family."""

    families: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {k.value: v for k, v in OPERATOR_FAMILIES.items()}
    )
    defaults: dict[str, str] = field(
        default_factory=lambda: {k.value: v for k, v in DEFAULT_OPERATORS.items()}
    )

    def is_valid(self, family: str, operator: str) -> bool:
        return operator in self.families.get(family, ())

    def default_for(self, family: str) -> str:
        return self.defaults.get(family, "")

    def all_operators(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for family, ops in self.families.items():
            for op in ops:
                out.append((family, op))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"families": self.families, "defaults": self.defaults}
