"""Tests for demo benchmark harness (no API, no campaigns)."""
from __future__ import annotations

from demo.benchmark.domain import DEMO_DOMAIN, DOMAIN_ID
from demo.benchmark.gold import BASELINE_CAMPAIGN_ID, require_gold_campaign
from demo.benchmark.metric import CampaignMetrics, compare_to_baseline, metrics_from_db_row
from demo.benchmark.report import build_campaign_asset, build_demo_report
from demo.benchmark.verifier import verify_main, verify_pilot


def test_single_domain():
    assert DEMO_DOMAIN.domain_id == DOMAIN_ID
    assert "contagion" in DEMO_DOMAIN.question.lower()
    body = DEMO_DOMAIN.campaign_body(compute_budget_hours=0.25)
    assert body["compute_budget_hours"] == 0.25
    assert body["breakthrough_criteria"]["metric_name"] == "final_outbreak_fraction"


def test_gold_corpus_enforcement():
    require_gold_campaign(BASELINE_CAMPAIGN_ID)
    try:
        require_gold_campaign("00000000-0000-0000-0000-000000000000")
        assert False, "should raise"
    except ValueError:
        pass


def test_metrics_and_baseline_comparison():
    row = {
        "campaign_id": "test-id",
        "question": DEMO_DOMAIN.question,
        "status": "budget_exhausted",
        "baseline_metric": 0.8,
        "best_metric": 0.95,
        "improvement_pct": 18.75,
        "total_confirmed": 8,
        "total_hypotheses": 50,
        "compute_seconds_used": 3600,
        "paper_count": 1,
        "max_closure_ratio": 0.7,
        "n_tree_nodes": 100,
        "n_frontier": 5,
    }
    m = metrics_from_db_row(row)
    assert m.beat_baseline
    cmp = compare_to_baseline(m, {"campaign_id": BASELINE_CAMPAIGN_ID, "best_metric": 0.8})
    assert cmp["delta"] == 0.15


def test_verifier_pilot_vs_main():
    m = CampaignMetrics(
        campaign_id="c1", question="q", status="active",
        baseline_metric=0.5, best_metric=0.6, improvement_pct=10,
        total_confirmed=1, total_hypotheses=5, compute_seconds_used=600,
        has_paper=False, max_closure_ratio=0.3, n_tree_nodes=10, n_frontier=2,
    )
    assert verify_pilot(m).passed
    assert not verify_main(m).passed


def test_demo_report_markdown():
    m = metrics_from_db_row({
        "campaign_id": BASELINE_CAMPAIGN_ID,
        "question": DEMO_DOMAIN.question,
        "status": "budget_exhausted",
        "baseline_metric": 0.8, "best_metric": 1.0, "improvement_pct": 25,
        "total_confirmed": 18, "total_hypotheses": 96,
        "compute_seconds_used": 5000, "paper_count": 1,
        "max_closure_ratio": 1.0, "n_tree_nodes": 96, "n_frontier": 0,
    })
    from demo.benchmark.verifier import verify_main

    asset = build_campaign_asset(
        m, verify_main(m),
        {"campaign_id": BASELINE_CAMPAIGN_ID, "best_metric": 0.8},
        tree_summary={"total_nodes": 96, "max_depth": 8},
    )
    report = build_demo_report([asset], gold_corpus_size=7, archive_size=48)
    md = report.to_markdown()
    assert "Graph Contagion" in md
    assert "contagion" in md.lower()
