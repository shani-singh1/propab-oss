"""Operator priors — state pattern → operator probability (P4)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.operator_credit.operator_registry import DEFAULT_OPERATORS, OperatorFamily, OperatorRegistry
from propab.operator_credit.operator_state import state_bucket
from propab.operator_credit.operator_statistics import OperatorStatistics


def priors_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "operator_priors.json"


@dataclass
class OperatorPrior:
    family: str
    state_bucket: str
    operator_weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorPriors:
    """Replace theme boosts with state-conditioned operator probabilities."""

    priors: dict[str, OperatorPrior] = field(default_factory=dict)
    registry: OperatorRegistry = field(default_factory=OperatorRegistry)

    def _key(self, family: str, bucket: str) -> str:
        return f"{family}|{bucket}"

    def build_from_statistics(self, stats: OperatorStatistics) -> None:
        by_family_bucket: dict[str, dict[str, float]] = {}
        for cell in stats.cells.values():
            fb = self._key(cell.family, cell.state_bucket)
            score = cell.mean_contribution + cell.p_success - 0.5 * cell.p_refute - 0.3 * cell.p_timeout
            by_family_bucket.setdefault(fb, {})[cell.operator] = max(0.05, score + 0.5)

        for fb, weights in by_family_bucket.items():
            family, bucket = fb.split("|", 1)
            total = sum(weights.values()) or 1.0
            normalized = {op: round(w / total, 4) for op, w in weights.items()}
            self.priors[fb] = OperatorPrior(
                family=family,
                state_bucket=bucket,
                operator_weights=normalized,
            )

    def operator_probability(
        self,
        family: str,
        operator: str,
        state: list[float],
    ) -> float:
        bucket = state_bucket(state)
        key = self._key(family, bucket)
        prior = self.priors.get(key)
        if prior:
            return prior.operator_weights.get(operator, 0.1)
        return 1.0 / max(1, len(self.registry.families.get(family, ())))

    def recommended_operator(self, family: str, state: list[float]) -> str:
        bucket = state_bucket(state)
        key = self._key(family, bucket)
        prior = self.priors.get(key)
        if prior and prior.operator_weights:
            return max(prior.operator_weights, key=lambda k: prior.operator_weights[k])
        default = self.registry.default_for(family)
        if default:
            return default
        try:
            return DEFAULT_OPERATORS[OperatorFamily(family)]
        except ValueError:
            return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "priors": {k: v.to_dict() for k, v in self.priors.items()},
            "registry": self.registry.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorPriors:
        priors = {
            k: OperatorPrior(**v)
            for k, v in (data.get("priors") or {}).items()
        }
        reg = OperatorRegistry()
        if data.get("registry"):
            reg = OperatorRegistry(**{
                k: data["registry"][k]
                for k in ("families", "defaults")
                if k in data["registry"]
            })
        return cls(priors=priors, registry=reg)

    def save(self, path: Path | None = None) -> Path:
        p = path or priors_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> OperatorPriors:
        p = path or priors_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()
