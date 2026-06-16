"""Operator credit assignment cycle — full pipeline (fixes.md week 1–2)."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.hypothesis_tree import HypothesisTree
from propab.operator_credit.bandit import OperatorBandit
from propab.operator_credit.campaign_corpus import (
    CampaignCorpus,
    harvest_from_bundles,
    ingest_trajectory_file,
)
from propab.operator_credit.campaign_family_dag import (
    build_campaign_family_dag,
    load_baselines_from_trajectory_file,
)
from propab.operator_credit.counterfactual_replay import run_counterfactual_suite
from propab.operator_credit.db_trace_loader import (
    campaign_ids_from_trajectory,
    load_bundles_from_db,
    load_bundles_from_trajectory_file,
)
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
    n_traces_from_snapshot: int
    n_traces_with_tool_calls: int
    n_credits: int
    n_stat_cells: int
    n_priors: int
    n_dag_edges: int
    n_hierarchical_credits: int
    n_counterfactuals: int
    n_family_nodes: int
    trace_source: str
    operator_bench: dict[str, Any]
    corpus_coverage: dict[str, Any]
    bandit: dict[str, Any]
    registry: dict[str, Any]
    weak_operators: list[dict[str, Any]]
    elapsed_ms: float
    family_dag: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_bundles(
    traj_path: Path,
    *,
    use_db: bool,
    trees: dict[str, HypothesisTree] | None,
    campaign_ids: list[str] | None = None,
) -> tuple[list[Any], str]:
    """Load campaign bundles; prefer Postgres when use_db=True."""
    ids = list(campaign_ids or [])
    if not ids and traj_path.is_file():
        ids = campaign_ids_from_trajectory(traj_path)

    if use_db and ids:
        try:
            bundles = asyncio.run(load_bundles_from_db(ids))
            if bundles and any(b.tree for b in bundles):
                if traj_path.is_file():
                    offline = load_bundles_from_trajectory_file(str(traj_path), trees=trees)
                    snap_by_id = {b.campaign_id: b.snapshots for b in offline}
                    for b in bundles:
                        if not b.snapshots and b.campaign_id in snap_by_id:
                            b.snapshots = snap_by_id[b.campaign_id]
                return bundles, "db"
        except Exception:
            pass

    if traj_path.is_file():
        return load_bundles_from_trajectory_file(str(traj_path), trees=trees), "offline"
    return [], "none"


def run_operator_credit_cycle(
    *,
    trajectory_path: Path | str | None = None,
    campaign_ids: list[str] | None = None,
    trees: dict[str, HypothesisTree | dict[str, Any]] | None = None,
    persist: bool = True,
    use_db: bool = True,
) -> tuple[OperatorCreditReport, OperatorTraceLedger, DifferenceRewardLedger]:
    t0 = time.perf_counter()
    traj_path = Path(trajectory_path or "artifacts/entropy_trajectories.json")

    tree_map: dict[str, HypothesisTree] = {}
    if trees:
        for cid, t in trees.items():
            tree_map[cid] = t if isinstance(t, HypothesisTree) else HypothesisTree.from_dict(t)

    bundles, trace_source = _load_bundles(
        traj_path, use_db=use_db, trees=tree_map or None, campaign_ids=campaign_ids,
    )
    if bundles:
        corpus, trace_ledger = harvest_from_bundles(bundles)
        snapshots_by_campaign = {b.campaign_id: b.snapshots for b in bundles}
        baselines = {
            b.campaign_id: b.baseline_campaign_id
            for b in bundles
            if b.baseline_campaign_id
        }
        if traj_path.is_file():
            baselines.update(load_baselines_from_trajectory_file(traj_path))
    elif traj_path.is_file():
        corpus, trace_ledger = ingest_trajectory_file(traj_path, trees=tree_map or None)
        offline_bundles = load_bundles_from_trajectory_file(str(traj_path), trees=tree_map or None)
        snapshots_by_campaign = {b.campaign_id: b.snapshots for b in offline_bundles}
        baselines = load_baselines_from_trajectory_file(traj_path)
        trace_source = "offline"
    else:
        corpus = CampaignCorpus()
        trace_ledger = OperatorTraceLedger()
        snapshots_by_campaign = {}
        baselines = {}
        trace_source = "none"

    n_tree = sum(1 for t in trace_ledger.traces if t.source == "tree")
    n_db = sum(1 for t in trace_ledger.traces if t.source == "db")
    n_snap = sum(1 for t in trace_ledger.traces if t.source == "snapshot")
    n_tools = sum(1 for t in trace_ledger.traces if t.tool_calls)

    store = PolicyStore.load()
    candidate = store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")

    credit_ledger = build_difference_rewards(
        traces=trace_ledger,
        snapshots_by_campaign=snapshots_by_campaign,
        trees={b.campaign_id: b.tree for b in bundles if b.tree} if bundles else (tree_map or None),
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

    tree_by_campaign = {b.campaign_id: b.tree for b in bundles if b.tree} if bundles else tree_map

    counterfactuals: list[dict[str, Any]] = []
    for cid in snapshots_by_campaign:
        snaps = [
            s for s in snapshots_by_campaign[cid]
            if s.get("theme_entropy") is not None
        ]
        if len(snaps) < 2:
            continue
        results = run_counterfactual_suite(
            campaign_id=cid,
            traces=trace_ledger,
            snapshots=snaps,
            tree=tree_by_campaign.get(cid) if tree_by_campaign else None,
            policy=candidate,
        )
        counterfactuals.extend([r.to_dict() for r in results])

    bench = run_operator_bench_suite(
        traces=trace_ledger,
        stats=stats,
        snapshots_by_campaign=snapshots_by_campaign,
        hierarchical=hierarchical,
        policy=candidate,
        trees=tree_by_campaign or None,
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
        n_traces_from_db=n_db,
        n_traces_from_snapshot=n_snap,
        n_traces_with_tool_calls=n_tools,
        n_credits=len(credit_ledger.credits),
        n_stat_cells=len(stats.cells),
        n_priors=len(priors.priors),
        n_dag_edges=len(dag.edges),
        n_hierarchical_credits=len(hierarchical.credits),
        n_counterfactuals=len(counterfactuals),
        n_family_nodes=len(family_dag.nodes),
        trace_source=trace_source,
        operator_bench=bench.to_dict(),
        corpus_coverage=(corpus.coverage.to_dict() if corpus.coverage else {}),
        bandit=bandit.to_dict(),
        registry=OperatorRegistry().to_dict(),
        weak_operators=weak,
        elapsed_ms=round(elapsed, 2),
        family_dag=family_dag.to_dict(),
    )
    return report, trace_ledger, credit_ledger
