"""State-conditioned operator statistics (P3)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.operator_credit.difference_rewards import DifferenceRewardLedger
from propab.operator_credit.operator_state import state_bucket
from propab.operator_credit.operator_trace import OperatorTraceLedger


def statistics_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "operator_statistics.json"


@dataclass
class OperatorStatCell:
    family: str
    operator: str
    state_bucket: str
    n: int = 0
    p_success: float = 0.0
    p_refute: float = 0.0
    p_timeout: float = 0.0
    mean_contribution: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorStatistics:
    cells: dict[str, OperatorStatCell] = field(default_factory=dict)

    def _key(self, family: str, operator: str, bucket: str) -> str:
        return f"{family}|{operator}|{bucket}"

    def update_from_traces(
        self,
        traces: OperatorTraceLedger,
        credits: DifferenceRewardLedger | None = None,
    ) -> None:
        credit_map: dict[tuple[str, str, str], list[float]] = {}
        if credits:
            for c in credits.credits:
                trace = next(
                    (t for t in traces.traces if t.node_id == c.node_id and t.campaign_id == c.campaign_id),
                    None,
                )
                if trace:
                    b = state_bucket(trace.state_vector)
                    credit_map.setdefault((c.family, c.operator, b), []).append(c.contribution)

        for trace in traces.traces:
            bucket = state_bucket(trace.state_vector)
            for step in trace.operators_used:
                key = self._key(step.family, step.operator, bucket)
                cell = self.cells.get(key) or OperatorStatCell(
                    family=step.family,
                    operator=step.operator,
                    state_bucket=bucket,
                )
                cell.n += 1
                if trace.outcome == "confirmed":
                    cell.p_success = round(
                        (cell.p_success * (cell.n - 1) + 1.0) / cell.n, 4
                    )
                elif trace.outcome == "refuted":
                    cell.p_refute = round(
                        (cell.p_refute * (cell.n - 1) + 1.0) / cell.n, 4
                    )
                elif trace.outcome == "inconclusive":
                    cell.p_timeout = round(
                        (cell.p_timeout * (cell.n - 1) + 1.0) / cell.n, 4
                    )
                contribs = credit_map.get((step.family, step.operator, bucket), [])
                if contribs:
                    cell.mean_contribution = round(sum(contribs) / len(contribs), 4)
                self.cells[key] = cell

    def best_operator(
        self,
        family: str,
        state: list[float],
    ) -> str | None:
        bucket = state_bucket(state)
        candidates = [
            c for k, c in self.cells.items()
            if c.family == family and c.state_bucket == bucket
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.mean_contribution + c.p_success - c.p_timeout).operator

    def to_dict(self) -> dict[str, Any]:
        return {"cells": {k: v.to_dict() for k, v in self.cells.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorStatistics:
        cells = {
            k: OperatorStatCell(**v)
            for k, v in (data.get("cells") or {}).items()
        }
        return cls(cells=cells)

    def save(self, path: Path | None = None) -> Path:
        p = path or statistics_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> OperatorStatistics:
        p = path or statistics_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()
