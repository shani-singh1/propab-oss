"""
P2 — Single benchmark domain: graphs / SIS contagion spreading.

Criteria:
  * objective metric (final_outbreak_fraction, improvement_pct)
  * cheap verification (numeric_summary, literature_baseline_compare)
  * reproducibility (fixed question, gold corpus baseline)
  * understandable to outsiders (network contagion)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Canonical question — matches all 7 gold-corpus Era-4 campaigns.
DEMO_QUESTION = (
    "Investigate which structural properties of complex networks most strongly "
    "determine the speed and extent of contagion spreading under competing diffusion models."
)

DOMAIN_ID = "graphs_sis_contagion"
DOMAIN_BUCKET = "graphs"
BUDGET_BUCKET = "3h"

PRIMARY_METRIC = "final_outbreak_fraction"
BASELINE_METRIC_NAME = "final_outbreak_fraction"

BREAKTHROUGH_CRITERIA: dict[str, Any] = {
    "metric_name": PRIMARY_METRIC,
    "improvement_threshold": 0.05,
    "direction": "higher_is_better",
    "min_confidence": 0.85,
    "min_replications": 2,
}


@dataclass(frozen=True)
class DemoDomainConfig:
    domain_id: str
    question: str
    domain_bucket: str
    budget_bucket: str
    primary_metric: str
    breakthrough_criteria: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def campaign_body(self, *, compute_budget_hours: float, policy_mode: str = "accepted") -> dict[str, Any]:
        return {
            "question": self.question,
            "compute_budget_hours": compute_budget_hours,
            "policy_mode": policy_mode,
            "breakthrough_criteria": dict(self.breakthrough_criteria),
        }


DEMO_DOMAIN = DemoDomainConfig(
    domain_id=DOMAIN_ID,
    question=DEMO_QUESTION,
    domain_bucket=DOMAIN_BUCKET,
    budget_bucket=BUDGET_BUCKET,
    primary_metric=PRIMARY_METRIC,
    breakthrough_criteria=BREAKTHROUGH_CRITERIA,
)
