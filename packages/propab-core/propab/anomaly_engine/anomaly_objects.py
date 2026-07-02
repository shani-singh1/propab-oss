"""AnomalyObject — surprising feature-subset signals from sweep results."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

ANOMALY_TYPES = frozenset({
    "family_violation",
    "threshold_effect",
    "outlier",
    "prediction_failure",
    "cluster_separation",
})


@dataclass
class AnomalyObject:
    feature_subset: list[str]
    metric_name: str
    expected_score: float
    observed_score: float
    surprise_score: float
    anomaly_type: str
    affected_families: list[str]
    neighboring_subsets: list[list[str]]
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if self.anomaly_type not in ANOMALY_TYPES:
            raise ValueError(f"Invalid anomaly_type: {self.anomaly_type}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnomalyObject:
        d = dict(data)
        d.pop("id", None)
        return cls(id=str(data.get("id") or uuid4()), **d)
