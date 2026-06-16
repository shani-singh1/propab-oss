"""Tests for operator credit assignment (P0–P6)."""
from __future__ import annotations

from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.operator_credit.bandit import OperatorBandit
from propab.operator_credit.credit_cycle import run_operator_credit_cycle
from propab.operator_credit.difference_rewards import (
    build_difference_rewards,
    estimate_contribution,
    node_outcome_reward,
)
from propab.operator_credit.operator_priors import OperatorPriors
from propab.operator_credit.operator_registry import OPERATOR_FAMILIES, OperatorFamily, OperatorRegistry
from propab.operator_credit.operator_statistics import OperatorStatistics
from propab.operator_credit.operator_trace import NodeOperatorTrace, OperatorStep
from propab.operator_credit.trace_extractor import extract_traces_from_tree


def _tree() -> HypothesisTree:
    tree = HypothesisTree()
    root = HypothesisNode(
        id="n1", text="spectral gap affects diffusion", parent_id=None, depth=0,
        verdict="confirmed", confidence=0.8, primary_theme="spectral",
        verification_method="statistical", expansion_type="mechanistic",
    )
    child = HypothesisNode(
        id="n2", text="boundary at sparse graphs", parent_id="n1", depth=1,
        verdict="refuted", confidence=0.7, primary_theme="sparse_regime",
        verification_method="counterexample", expansion_type="boundary",
    )
    tree.nodes = {"n1": root, "n2": child}
    tree.frontier = []
    return tree


def test_operator_registry_families():
    reg = OperatorRegistry()
    assert reg.is_valid("branching", "closure_aware")
    assert len(reg.all_operators()) == sum(len(v) for v in OPERATOR_FAMILIES.values())


def test_extract_traces_from_tree():
    ledger = extract_traces_from_tree(campaign_id="c1", tree=_tree())
    assert len(ledger.traces) == 2
    assert ledger.traces[0].branching in ("breadth_first", "depth_first", "closure_aware")
    assert ledger.traces[0].verification in ("symbolic", "numerical", "simulation")


def test_difference_rewards():
    trace = NodeOperatorTrace(
        campaign_id="c1",
        node_id="n1",
        operators_used=[OperatorStep("branching", "closure_aware", 0)],
        order=0,
        cost=1.0,
        outcome="confirmed",
        branching="closure_aware",
    )
    credits = estimate_contribution(trace=trace, reward_all=0.7)
    assert credits[0].contribution >= 0


def test_operator_statistics_and_priors():
    from propab.operator_credit.operator_trace import OperatorTraceLedger

    ledger = extract_traces_from_tree(campaign_id="c1", tree=_tree())
    stats = OperatorStatistics()
    stats.update_from_traces(ledger)
    assert len(stats.cells) > 0

    priors = OperatorPriors()
    priors.build_from_statistics(stats)
    assert len(priors.priors) > 0
    op = priors.recommended_operator("branching", [0.1, 0.2, 0.3, 0.4, 0.5])
    assert op in OPERATOR_FAMILIES[OperatorFamily.BRANCHING]


def test_bandit_ucb_and_thompson():
    bandit = OperatorBandit()
    bandit.ensure_arms("branching", ("breadth_first", "closure_aware"))
    bandit.update("branching", "closure_aware", 0.8, success=True)
    bandit.update("branching", "breadth_first", 0.2, success=False)
    ucb = bandit.select_ucb("branching")
    ts = bandit.select_thompson("branching")
    assert ucb in ("breadth_first", "closure_aware")
    assert ts in ("breadth_first", "closure_aware")


def test_node_outcome_reward():
    assert node_outcome_reward("confirmed") > node_outcome_reward("inconclusive")


def test_credit_cycle_on_trajectories(tmp_path, monkeypatch):
    from propab.config import settings

    traj = tmp_path / "traj.json"
    traj.write_text(
        '{"campaigns":[{"campaign_id":"c1","trajectory":['
        '{"tested":1,"theme_entropy":0.5,"closure_ratio":0.1},'
        '{"tested":2,"theme_entropy":0.7,"closure_ratio":0.2}'
        "]}]}",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "propab_data_dir", str(data_dir))

    report, traces, credits = run_operator_credit_cycle(
        trajectory_path=traj, persist=True, use_db=False,
    )
    assert report.n_campaigns == 1
    assert report.n_traces >= 1
    assert report.n_credits >= 1
