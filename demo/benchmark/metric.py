"""P3 — Objective metrics for demo benchmark runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from demo.benchmark.domain import PRIMARY_METRIC


@dataclass
class CampaignMetrics:
    campaign_id: str
    question: str
    status: str
    baseline_metric: float | None
    best_metric: float | None
    improvement_pct: float | None
    total_confirmed: int
    total_hypotheses: int
    compute_seconds_used: int
    has_paper: bool
    max_closure_ratio: float
    n_tree_nodes: int
    n_frontier: int
    primary_metric: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def closure_rate(self) -> float:
        tested = max(1, self.total_hypotheses)
        return round(self.total_confirmed / tested, 4)

    @property
    def beat_baseline(self) -> bool:
        if self.improvement_pct is None:
            return False
        return self.improvement_pct >= 5.0


def metrics_from_db_row(row: dict[str, Any]) -> CampaignMetrics:
    return CampaignMetrics(
        campaign_id=str(row["campaign_id"]),
        question=str(row.get("question") or ""),
        status=str(row.get("status") or "unknown"),
        baseline_metric=_float(row.get("baseline_metric")),
        best_metric=_float(row.get("best_metric")),
        improvement_pct=_float(row.get("improvement_pct")),
        total_confirmed=int(row.get("total_confirmed") or 0),
        total_hypotheses=int(row.get("total_hypotheses") or 0),
        compute_seconds_used=int(row.get("compute_seconds_used") or 0),
        has_paper=bool(int(row.get("paper_count") or 0) > 0),
        max_closure_ratio=float(row.get("max_closure_ratio") or 0),
        n_tree_nodes=int(row.get("n_tree_nodes") or 0),
        n_frontier=int(row.get("n_frontier") or 0),
        primary_metric=_float(row.get("best_metric")),
    )


def metrics_from_api_summary(campaign_id: str, blob: dict[str, Any]) -> CampaignMetrics:
    s = blob.get("summary") or blob
    tree = s.get("tree") or {}
    return CampaignMetrics(
        campaign_id=campaign_id,
        question=str(s.get("question") or ""),
        status=str(s.get("status") or "unknown"),
        baseline_metric=_float(s.get("baseline_metric")),
        best_metric=_float(s.get("best_metric")),
        improvement_pct=_float(s.get("improvement_pct")),
        total_confirmed=int(s.get("total_confirmed") or 0),
        total_hypotheses=int(s.get("total_hypotheses") or 0),
        compute_seconds_used=int(s.get("elapsed_sec") or 0),
        has_paper=bool((blob.get("event_counts_by_type") or {}).get("paper.ready", 0) > 0),
        max_closure_ratio=0.0,
        n_tree_nodes=int(tree.get("total_nodes") or 0),
        n_frontier=int(tree.get("frontier_size") or 0),
        primary_metric=_float(s.get("best_metric")),
    )


def compare_to_baseline(
    metrics: CampaignMetrics,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    b_best = _float(baseline.get("best_metric"))
    m_best = metrics.best_metric
    delta = None
    if b_best is not None and m_best is not None:
        delta = round(m_best - b_best, 4)
    return {
        "primary_metric": PRIMARY_METRIC,
        "baseline_campaign_id": baseline.get("campaign_id"),
        "baseline_best": b_best,
        "campaign_best": m_best,
        "delta": delta,
        "improvement_pct": metrics.improvement_pct,
        "beat_baseline": metrics.beat_baseline,
    }


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
