"""Operator credit assignment cycle — full pipeline (fixes.md week 1–2)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.hypothesis_tree import HypothesisTree
from propab.operator_credit.bandit import OperatorBandit
from propab.operator_credit.campaign_corpus import CampaignCorpus, ingest_trajectory_file
from propab.operator_credit.campaign_family_dag import (
    build_campaign_family_dag,
    load_baselines_from_trajectory_file,
)
from propab.operator_credit.counterfactual_replay import run_counterfactual_suite
from propab.operator_credit.db_trace_loader import load_bundles_from_trajectory_file
from propab.operator_credit.difference_rewards import (
    DifferenceRewardLedger,
    build_difference_rewards,
)
from propab.operator_credit.hierarchical_credit import build_hierarchical_credits
from propab.operator_credit.operator_bench import run_operator_bench_suite
from propab.operator_credit.operator_dag import OperatorDAG
from propab.operator_credit.operator_priors import OperatorPriors
from propab.operator_credit.operator_registry import OPERATOR_FAMILIES, OperatorRegistry
from propab.operator_credit.operator_statistics import OperatorStatistics
from propab.operator_credit.operator_trace import OperatorTraceLedger
from propab.policy_store import PolicyStore


def _lifetime_dir() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base


@dataclass
class OperatorCreditReport:
    n_campaigns: int
    n_traces: int
    n_traces_from_tree: int
    n_traces_from_db: int
    n_credits: int
    n_stat_cells: int
    n_priors: int
    n_dag_edges: int
    n_hierarchical_credits: int
    n_counterfactuals: int
    n_family_nodes: int
    operator_bench: dict[str, Any]
    corpus_coverage: dict[str, Any]
    bandit: dict[str, Any]
    registry: dict[str, Any]
    weak_operators: list[dict[str, Any]]
    elapsed_ms: float
    family_dag: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_operator_credit_cycle(
    *,
    trajectory_path: Path | str | None = None,
    trees: dict[str, HypothesisTree | dict[str, Any]] | None = None,
    persist: bool = True,
) -> tuple[OperatorCreditReport, OperatorTraceLedger, DifferenceRewardLedger]:
    t0 = time.perf_counter()
    traj_path = Path(trajectory_path or "artifacts/entropy_trajectories.json")

    tree_map: dict[str, HypothesisTree] = {}
    if trees:
        for cid, t in trees.items():
            tree_map[cid] = t if isinstance(t, HypothesisTree) else HypothesisTree.from_dict(t)

    if traj_path.is_file():
        corpus, trace_ledger = ingest_trajectory_file(traj_path, trees=tree_map or None)
        bundles = load_bundles_from_trajectory_file(str(traj_path), trees=tree_map or None)
        snapshots_by_campaign = {b.campaign_id: b.snapshots for b in bundles}
        baselines = load_baselines_from_trajectory_file(traj_path)
    else:
        corpus = CampaignCorpus()
        trace_ledger = OperatorTraceLedger()
        snapshots_by_campaign = {}
        baselines = {}

    n_tree = sum(1 for t in trace_ledger.traces if t.source == "tree")
    n_snap = sum(1 for t in trace_ledger.traces if t.source == "snapshot")

    store = PolicyStore.load()
    candidate = store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")

    credit_ledger = build_difference_rewards(
        traces=trace_ledger,
        snapshots_by_campaign=snapshots_by_campaign,
        trees=tree_map or None,
        candidate_policy=candidate,
        baseline_policy=candidate,
    )

    stats = OperatorStatistics()
    stats.update_from_traces(trace_ledger, credit_ledger)

    priors = OperatorPriors()
    priors.build_from_statistics(stats)

    family_dag = build_campaign_family_dag(
        campaign_ids=list(snapshots_by_campaign.keys()),
        baseline_by_campaign=baselines,
        priors=priors,
    )
    families = family_dag.families_by_root()

    hierarchical = build_hierarchical_credits(
        traces=trace_ledger,
        snapshots_by_campaign=snapshots_by_campaign,
        diff_ledger=credit_ledger,
        campaign_families=families,
    )

    dag = OperatorDAG.from_traces(trace_ledger)

    counterfactuals: list[dict[str, Any]] = []
    for cid in snapshots_by_campaign:
        results = run_counterfactual_suite(
            campaign_id=cid,
            traces=trace_ledger,
            snapshots=snapshots_by_campaign[cid],
            tree=tree_map.get(cid),
            policy=candidate,
        )
        counterfactuals.extend([r.to_dict() for r in results])

    bench = run_operator_bench_suite(
        traces=trace_ledger,
        stats=stats,
        snapshots_by_campaign=snapshots_by_campaign,
        hierarchical=hierarchical,
        policy=candidate,
        trees=tree_map or None,
    )

    bandit = OperatorBandit()
    for family, ops in OPERATOR_FAMILIES.items():
        bandit.ensure_arms(family.value, ops)
    for credit in credit_ledger.credits:
        bandit.update(
            credit.family,
            credit.operator,
            credit.contribution,
            success=credit.contribution > 0.1,
        )

    weak = sorted(
        [
            {
                "family": c.family,
                "operator": c.operator,
                "mean_contribution": round(c.contribution, 4),
                "campaign_id": c.campaign_id,
            }
            for c in credit_ledger.credits
        ],
        key=lambda x: x["mean_contribution"],
    )[:10]

    if persist:
        trace_ledger.save()
        stats.save()
        priors.save()
        corpus.save()
        family_dag.save()
        base = _lifetime_dir()
        (base / "operator_credits.json").write_text(
            json.dumps(credit_ledger.to_dict(), indent=2), encoding="utf-8",
        )
        (base / "operator_bandit.json").write_text(
            json.dumps(bandit.to_dict(), indent=2), encoding="utf-8",
        )
        (base / "operator_dag.json").write_text(
            json.dumps(dag.to_dict(), indent=2), encoding="utf-8",
        )
        (base / "hierarchical_credits.json").write_text(
            json.dumps(hierarchical.to_dict(), indent=2), encoding="utf-8",
        )
        (base / "counterfactual_results.json").write_text(
            json.dumps(counterfactuals, indent=2), encoding="utf-8",
        )
        (base / "operator_bench.json").write_text(
            json.dumps(bench.to_dict(), indent=2), encoding="utf-8",
        )

    elapsed = (time.perf_counter() - t0) * 1000
    report = OperatorCreditReport(
        n_campaigns=len(snapshots_by_campaign),
        n_traces=len(trace_ledger.traces),
        n_traces_from_tree=n_tree,
        n_traces_from_db=n_snap,
        n_credits=len(credit_ledger.credits),
        n_stat_cells=len(stats.cells),
        n_priors=len(priors.priors),
        n_dag_edges=len(dag.edges),
        n_hierarchical_credits=len(hierarchical.credits),
        n_counterfactuals=len(counterfactuals),
        n_family_nodes=len(family_dag.nodes),
        operator_bench=bench.to_dict(),
        corpus_coverage=(corpus.coverage.to_dict() if corpus.coverage else {}),
        bandit=bandit.to_dict(),
        registry=OperatorRegistry().to_dict(),
        weak_operators=weak,
        elapsed_ms=round(elapsed, 2),
        family_dag=family_dag.to_dict(),
    )
    return report, trace_ledger, credit_ledger
