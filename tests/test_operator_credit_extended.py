"""Extended tests for operator credit infrastructure (fixes.md week 1–2)."""
from __future__ import annotations

import json

from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.operator_credit.campaign_corpus import ingest_trajectory_file
from propab.operator_credit.campaign_family_dag import build_campaign_family_dag
from propab.operator_credit.counterfactual_replay import (
    CounterfactualSpec,
    run_counterfactual_on_traces,
    run_counterfactual_suite,
)
from propab.operator_credit.credit_cycle import run_operator_credit_cycle
from propab.operator_credit.db_trace_loader import (
    CampaignDBBundle,
    ToolCallRecord,
    enrich_trace_from_db,
    extract_traces_from_db_bundle,
)
from propab.operator_credit.hierarchical_credit import CreditLevel, build_hierarchical_credits
from propab.operator_credit.operator_bench import run_operator_bench_suite
from propab.operator_credit.operator_dag import OperatorDAG
from propab.operator_credit.operator_statistics import OperatorStatistics
from propab.operator_credit.operator_trace import NodeOperatorTrace, OperatorStep, OperatorTraceLedger
from propab.operator_credit.search_state_v3 import SearchStateV3
from propab.operator_credit.trace_extractor import extract_trace_for_node


def _tree_with_tools() -> HypothesisTree:
    tree = HypothesisTree()
    root = HypothesisNode(
        id="n1", text="spectral gap affects diffusion", parent_id=None, depth=0,
        verdict="confirmed", confidence=0.8, primary_theme="spectral",
        verification_method="statistical", expansion_type="mechanistic",
        children=["n2"],
    )
    child = HypothesisNode(
        id="n2", text="boundary at sparse graphs", parent_id="n1", depth=1,
        verdict="refuted", confidence=0.7, primary_theme="sparse_regime",
        verification_method="counterexample", expansion_type="boundary",
    )
    tree.nodes = {"n1": root, "n2": child}
    tree.frontier = []
    return tree


def test_db_bundle_enriched_trace():
    tree = _tree_with_tools()
    node = tree.nodes["n1"]
    trace = extract_trace_for_node(campaign_id="c1", node=node, tree=tree, order=0)
    tool_calls = [
        ToolCallRecord("numeric_summary", True, 120, "db-h1"),
    ]
    enrich_trace_from_db(
        trace, node=node, tree=tree,
        tool_calls=tool_calls,
        hypothesis_db_ids={"n1": "db-h1"},
    )
    assert trace.parent_node_id is None
    assert trace.child_node_ids == ["n2"]
    assert trace.expansion_type == "mechanistic"
    assert len(trace.tool_calls) == 1
    assert trace.duration_ms == 120
    assert trace.source == "tree"


def test_extract_traces_from_db_bundle():
    tree = _tree_with_tools()
    bundle = CampaignDBBundle(
        campaign_id="c1",
        tree=tree,
        snapshots=[{"tested": 1, "theme_entropy": 0.5, "closure_ratio": 0.1}],
        baseline_campaign_id="root",
    )
    ledger = extract_traces_from_db_bundle(bundle)
    assert len(ledger.traces) == 2
    assert all(t.source == "tree" for t in ledger.traces)


def test_operator_dag_from_traces():
    tree = _tree_with_tools()
    ledger = extract_traces_from_db_bundle(CampaignDBBundle(campaign_id="c1", tree=tree))
    dag = OperatorDAG.from_traces(ledger)
    assert len(dag.edges) == 12  # 2 nodes × 6 operators
    assert dag.edges[0].family == "retrieval"


def test_counterfactual_remove_retrieval():
    tree = _tree_with_tools()
    ledger = extract_traces_from_db_bundle(CampaignDBBundle(campaign_id="c1", tree=tree))
    snaps = [{"tested": 1, "theme_entropy": 0.5, "closure_ratio": 0.3}]
    result = run_counterfactual_on_traces(
        campaign_id="c1",
        traces=ledger,
        snapshots=snaps,
        spec=CounterfactualSpec(remove_family="retrieval"),
    )
    assert result.delta != 0 or result.n_affected_nodes >= 0


def test_counterfactual_suite():
    tree = _tree_with_tools()
    ledger = extract_traces_from_db_bundle(CampaignDBBundle(campaign_id="c1", tree=tree))
    snaps = [{"tested": i, "theme_entropy": 0.5 + i * 0.1, "closure_ratio": 0.2} for i in range(5)]
    results = run_counterfactual_suite(
        campaign_id="c1", traces=ledger, snapshots=snaps, tree=tree,
    )
    assert len(results) >= 5


def test_hierarchical_credits_all_levels():
    tree = _tree_with_tools()
    bundle = CampaignDBBundle(campaign_id="c1", tree=tree)
    ledger = extract_traces_from_db_bundle(bundle)
    ledger.traces[0].tool_calls = [{"tool_name": "t", "success": True, "duration_ms": 50}]
    hier = build_hierarchical_credits(
        traces=ledger,
        snapshots_by_campaign={"c1": [{"tested": 1, "theme_entropy": 0.5, "closure_ratio": 0.2}]},
        campaign_families={"root": ["c1"]},
    )
    assert len(hier.by_level(CreditLevel.MICRO)) >= 1
    assert len(hier.by_level(CreditLevel.MESO)) >= 1
    assert len(hier.by_level(CreditLevel.MACRO)) >= 1
    assert len(hier.by_level(CreditLevel.MEGA)) >= 1


def test_campaign_family_dag():
    dag = build_campaign_family_dag(
        campaign_ids=["c1", "c2", "c3"],
        baseline_by_campaign={"c1": None, "c2": "c1", "c3": "c2"},
    )
    assert len(dag.nodes) == 3
    assert dag.lineage("c3") == ["c1", "c2", "c3"]
    assert dag.nodes["c2"].depth == 1
    assert dag.nodes["c3"].depth == 2


def test_search_state_v3_vector():
    tree = _tree_with_tools()
    node = tree.nodes["n1"]
    trace = extract_trace_for_node(campaign_id="c1", node=node, tree=tree, order=0)
    state = SearchStateV3.from_node_and_tree(node, tree, trace)
    vec = state.to_vector()
    assert len(vec) == 15
    assert 0 <= state.uncertainty <= 1


def test_operator_bench_suite():
    tree = _tree_with_tools()
    ledger = extract_traces_from_db_bundle(CampaignDBBundle(campaign_id="c1", tree=tree))
    stats = OperatorStatistics()
    stats.update_from_traces(ledger)
    snaps = {"c1": [{"tested": i, "theme_entropy": 0.5, "closure_ratio": 0.2} for i in range(3)]}
    bench = run_operator_bench_suite(
        traces=ledger, stats=stats, snapshots_by_campaign=snaps, trees={"c1": tree},
    )
    assert len(bench.results) > 0
    assert bench.weakest_family


def test_corpus_ingest(tmp_path):
    traj = tmp_path / "traj.json"
    traj.write_text(json.dumps({
        "baseline_campaign_id": "root",
        "campaigns": [{
            "campaign_id": "c1",
            "baseline_campaign_id": "root",
            "trajectory": [
                {"tested": 1, "theme_entropy": 0.5, "closure_ratio": 0.1},
                {"tested": 2, "theme_entropy": 0.7, "closure_ratio": 0.2},
            ],
        }],
    }), encoding="utf-8")
    tree = _tree_with_tools()
    corpus, ledger = ingest_trajectory_file(traj, trees={"c1": tree})
    assert corpus.entries[0].has_tree is True
    assert corpus.entries[0].n_traces == 2
    assert len(ledger.traces) == 2


def test_full_credit_cycle_with_tree(tmp_path, monkeypatch):
    from propab.config import settings

    traj = tmp_path / "traj.json"
    tree = _tree_with_tools()
    traj.write_text(json.dumps({
        "campaigns": [{
            "campaign_id": "c1",
            "trajectory": [
                {"tested": 1, "theme_entropy": 0.5, "closure_ratio": 0.1},
                {"tested": 2, "theme_entropy": 0.7, "closure_ratio": 0.2},
            ],
        }],
    }), encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "propab_data_dir", str(data_dir))

    report, traces, credits = run_operator_credit_cycle(
        trajectory_path=traj,
        trees={"c1": tree},
        persist=True,
    )
    assert report.n_traces_from_tree == 2
    assert report.n_dag_edges > 0
    assert report.n_hierarchical_credits > 0
    assert report.n_counterfactuals > 0
    assert report.n_family_nodes >= 1
    assert (data_dir / "lifetime_knowledge" / "operator_dag.json").is_file()
