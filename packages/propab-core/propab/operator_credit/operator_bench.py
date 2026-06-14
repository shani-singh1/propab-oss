"""OperatorBench — offline operator family benchmarks (fixes.md #6)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from propab.operator_credit.counterfactual_replay import CounterfactualResult, run_counterfactual_suite
from propab.operator_credit.hierarchical_credit import HierarchicalCreditLedger
from propab.operator_credit.operator_registry import OPERATOR_FAMILIES, OperatorFamily
from propab.operator_credit.operator_statistics import OperatorStatistics
from propab.operator_credit.operator_trace import OperatorTraceLedger
from propab.policy_record import PolicyRecord


@dataclass
class OperatorBenchResult:
    family: str
    operator: str
    n_observations: int
    mean_contribution: float
    p_success: float
    p_refute: float
    p_timeout: float
    counterfactual_delta_mean: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorBenchSuite:
    results: list[OperatorBenchResult] = field(default_factory=list)
    weakest_family: str = ""
    strongest_family: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "weakest_family": self.weakest_family,
            "strongest_family": self.strongest_family,
        }


def _family_bench(
    family: str,
    stats: OperatorStatistics,
    hierarchical: HierarchicalCreditLedger | None,
    counterfactuals: list[CounterfactualResult],
) -> list[OperatorBenchResult]:
    results: list[OperatorBenchResult] = []
    fam_enum = next((f for f in OperatorFamily if f.value == family), None)
    operators = OPERATOR_FAMILIES.get(fam_enum, ()) if fam_enum else ()
    cf_deltas = [
        c.delta for c in counterfactuals
        if c.spec.get("remove_family") == family
        or family in str(c.spec)
    ]

    for op in operators:
        cells = [c for c in stats.cells.values() if c.family == family and c.operator == op]
        n = sum(c.n for c in cells)
        p_succ = sum(c.p_success * c.n for c in cells) / max(1, n)
        p_ref = sum(c.p_refute * c.n for c in cells) / max(1, n)
        p_to = sum(c.p_timeout * c.n for c in cells) / max(1, n)
        mean_c = sum(c.mean_contribution * c.n for c in cells) / max(1, n)

        if hierarchical:
            hc = [
                c for c in hierarchical.credits
                if c.family == family and c.operator == op
            ]
            if hc:
                mean_c = sum(c.contribution for c in hc) / len(hc)

        cf_mean = sum(cf_deltas) / max(1, len(cf_deltas)) if cf_deltas else 0.0
        passed = mean_c > 0.05 and p_succ >= p_to
        results.append(OperatorBenchResult(
            family=family,
            operator=op,
            n_observations=n,
            mean_contribution=round(mean_c, 4),
            p_success=round(p_succ, 4),
            p_refute=round(p_ref, 4),
            p_timeout=round(p_to, 4),
            counterfactual_delta_mean=round(cf_mean, 4),
            passed=passed,
        ))
    return results


def run_operator_bench_suite(
    *,
    traces: OperatorTraceLedger,
    stats: OperatorStatistics,
    snapshots_by_campaign: dict[str, list[dict[str, Any]]],
    hierarchical: HierarchicalCreditLedger | None = None,
    policy: PolicyRecord | None = None,
    trees: dict[str, Any] | None = None,
) -> OperatorBenchSuite:
    all_cf: list[CounterfactualResult] = []
    for cid in {t.campaign_id for t in traces.traces}:
        all_cf.extend(run_counterfactual_suite(
            campaign_id=cid,
            traces=traces,
            snapshots=snapshots_by_campaign.get(cid, []),
            tree=trees.get(cid) if trees else None,
            policy=policy,
        ))

    results: list[OperatorBenchResult] = []
    for family_enum in OPERATOR_FAMILIES:
        results.extend(_family_bench(
            family_enum.value,
            stats,
            hierarchical,
            all_cf,
        ))

    by_family: dict[str, float] = {}
    for r in results:
        by_family[r.family] = by_family.get(r.family, 0) + r.mean_contribution

    weakest = min(by_family, key=by_family.get) if by_family else ""
    strongest = max(by_family, key=by_family.get) if by_family else ""
    return OperatorBenchSuite(
        results=results,
        weakest_family=weakest,
        strongest_family=strongest,
    )
