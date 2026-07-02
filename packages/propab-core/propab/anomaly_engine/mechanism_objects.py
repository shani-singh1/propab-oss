"""MechanismObject — LLM-induced explanations for detected anomalies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class MechanismObject:
    explanation: str
    candidate_features: list[str]
    supporting_anomalies: list[str]
    assumptions_challenged: list[str]
    confidence: float
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MechanismObject:
        d = dict(data)
        d.pop("id", None)
        return cls(id=str(data.get("id") or uuid4()), **d)
