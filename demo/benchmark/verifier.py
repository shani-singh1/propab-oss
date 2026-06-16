"""P3 — Cheap verification checks for demo benchmark runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from demo.benchmark.metric import CampaignMetrics


@dataclass
class VerificationResult:
    campaign_id: str
    passed: bool
    checks: dict[str, bool]
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_campaign(
    metrics: CampaignMetrics,
    *,
    require_paper: bool = False,
    min_confirmed: int = 1,
    min_hypotheses: int = 3,
    min_tree_nodes: int = 5,
) -> VerificationResult:
    """
    Cheap offline verification — no new architecture, no LLM calls.
    Pilot runs use relaxed thresholds; main runs use stricter ones.
    """
    checks = {
        "has_hypotheses": metrics.total_hypotheses >= min_hypotheses,
        "has_confirmed": metrics.total_confirmed >= min_confirmed,
        "has_tree": metrics.n_tree_nodes >= min_tree_nodes,
        "has_best_metric": metrics.best_metric is not None,
        "terminal_status": metrics.status in (
            "breakthrough", "budget_exhausted", "completed", "active",
        ),
    }
    if require_paper:
        checks["has_paper"] = metrics.has_paper

    failures = [name for name, ok in checks.items() if not ok]
    return VerificationResult(
        campaign_id=metrics.campaign_id,
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
    )


def verify_pilot(metrics: CampaignMetrics) -> VerificationResult:
    """Relaxed checks for 10–20 min pilot runs (find bugs, not discoveries)."""
    return verify_campaign(
        metrics,
        require_paper=False,
        min_confirmed=0,
        min_hypotheses=1,
        min_tree_nodes=3,
    )


def verify_main(metrics: CampaignMetrics) -> VerificationResult:
    """Stricter checks for 2–4 h main demo runs."""
    return verify_campaign(
        metrics,
        require_paper=True,
        min_confirmed=2,
        min_hypotheses=10,
        min_tree_nodes=15,
    )
