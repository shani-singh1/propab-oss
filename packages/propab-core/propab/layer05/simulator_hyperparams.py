"""Simulator hyperparameters for grid search and versioned simulators."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SimulatorHyperparams:
    retrieval_weight: float = 0.7
    k_neighbors: int = 3
    normalization: str = "none"
    distance_metric: str = "euclidean"
    feature_version: str = "v2"
    knn_weight: float = 0.5
    predictor_weight: float = 0.3
    rules_weight: float = field(default=0.2)
    direction_weight: float = 5.0
    magnitude_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.predictor_weight <= 0:
            self.knn_weight = self.retrieval_weight
            self.rules_weight = max(0.0, 1.0 - self.retrieval_weight)

    def for_v2(self) -> SimulatorHyperparams:
        return SimulatorHyperparams(
            retrieval_weight=self.retrieval_weight,
            k_neighbors=self.k_neighbors,
            normalization=self.normalization,
            distance_metric=self.distance_metric,
            feature_version=self.feature_version,
            knn_weight=self.retrieval_weight,
            predictor_weight=0.0,
            rules_weight=max(0.0, 1.0 - self.retrieval_weight),
        )

    def for_v3(self) -> SimulatorHyperparams:
        rw = max(0.0, 1.0 - self.knn_weight - self.predictor_weight)
        return SimulatorHyperparams(
            retrieval_weight=self.knn_weight,
            k_neighbors=self.k_neighbors,
            normalization=self.normalization,
            distance_metric=self.distance_metric,
            feature_version=self.feature_version,
            knn_weight=self.knn_weight,
            predictor_weight=self.predictor_weight,
            rules_weight=rw,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulatorHyperparams:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
