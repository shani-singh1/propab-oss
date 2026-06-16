"""Tests for campaign era partitioning (fixes.md P0–P6)."""
from __future__ import annotations

from datetime import date

from propab.operator_credit.campaign_era import (
    EraId,
    build_campaign_metadata,
    build_experience_archive,
    classify_era,
    compute_cross_era_comparisons,
    compute_quality_score,
    select_gold_corpus,
    trust_weight,
    CampaignEraMetadata,
    EraLocalStatistics,
    OperatorStatCell,
)


def _meta(**kwargs) -> CampaignEraMetadata:
    base = {
        "campaign_id": "c1",
        "era_id": 0,
        "started_at": "2026-06-06",
        "total_confirmed": 10,
        "total_hypotheses": 50,
        "has_paper": True,
        "max_closure_ratio": 0.6,
        "n_tool_calls": 200,
        "n_snapshots": 25,
        "policy_id": "pol-abc",
        "architecture_features": {
            "literature_repaired": True,
            "claim_typing": True,
            "simulator_eval": True,
            "operator_credit": True,
            "db_traces": True,
        },
    }
    base.update(kwargs)
    m = CampaignEraMetadata(**base)
    m.quality_score = compute_quality_score(m)
    m.trust_weight = trust_weight(m, reference=date(2026, 6, 7))
    return m


def test_classify_era_0_early():
    m = _meta(started_at="2026-05-03", policy_id=None, n_tool_calls=0, n_snapshots=0)
    m.architecture_features = {}
    assert classify_era(m) == EraId.ERA_0


def test_classify_era_1_mid_may():
    m = _meta(started_at="2026-05-12", policy_id=None, n_tool_calls=30, n_snapshots=5)
    m.architecture_features = {"literature_repaired": True}
    assert classify_era(m) == EraId.ERA_1


def test_classify_era_2_june_early():
    m = _meta(started_at="2026-06-01", policy_id=None, n_tool_calls=100, n_snapshots=15)
    m.architecture_features = {"simulator_eval": True}
    assert classify_era(m) == EraId.ERA_2


def test_classify_era_4_db_traces():
    m = _meta(started_at="2026-06-06", policy_id="pol-x", n_tool_calls=200, n_snapshots=25)
    m.architecture_features = {"db_traces": True, "operator_credit": True}
    assert classify_era(m) == EraId.ERA_4


def test_trust_weight_recent_era_dominates():
    old = _meta(started_at="2026-05-03", era_id=0)
    old.trust_weight = trust_weight(old, reference=date(2026, 6, 7))
    new = _meta(started_at="2026-06-06", era_id=4)
    new.trust_weight = trust_weight(new, reference=date(2026, 6, 7))
    assert new.trust_weight > old.trust_weight


def test_select_gold_corpus():
    campaigns = [
        _meta(campaign_id="a", era_id=4, quality_score=5.0),
        _meta(campaign_id="b", era_id=4, quality_score=4.0),
        _meta(campaign_id="c", era_id=2, quality_score=6.0),
        _meta(campaign_id="d", era_id=0, quality_score=1.0),
    ]
    for c in campaigns:
        c.quality_score = compute_quality_score(c)
    gold = select_gold_corpus(campaigns, min_era=3, max_size=2, min_quality=1.0)
    assert len(gold.campaign_ids) == 2
    assert "c" not in gold.campaign_ids
    assert "d" not in gold.campaign_ids
    archive = build_experience_archive(campaigns)
    assert len(archive.campaign_ids) == 2


def test_build_campaign_metadata_from_row():
    row = {
        "campaign_id": "abc",
        "started_at": "2026-06-06T12:00:00+00:00",
        "total_confirmed": 18,
        "total_hypotheses": 96,
        "policy_id": "pol-60f1d9427b67",
        "budget_bucket": "3h",
        "paper_count": 1,
        "n_snapshots": 28,
        "n_tool_calls": 432,
        "max_closure_ratio": 0.85,
    }
    meta = build_campaign_metadata(row)
    assert meta.era_id >= EraId.ERA_3
    assert meta.trust_weight > 0
    assert meta.quality_score > 0


def test_cross_era_ranking_stability():
    era_stats = [
        EraLocalStatistics(
            era_id=3,
            cells={
                "branching|closure_aware|growth": OperatorStatCell(
                    family="branching", operator="closure_aware", state_bucket="growth",
                    n=10, p_success=0.8, mean_contribution=0.5,
                ),
                "branching|breadth_first|growth": OperatorStatCell(
                    family="branching", operator="breadth_first", state_bucket="growth",
                    n=10, p_success=0.2, mean_contribution=0.1,
                ),
            },
        ),
        EraLocalStatistics(
            era_id=4,
            cells={
                "branching|closure_aware|growth": OperatorStatCell(
                    family="branching", operator="closure_aware", state_bucket="growth",
                    n=10, p_success=0.75, mean_contribution=0.45,
                ),
                "branching|breadth_first|growth": OperatorStatCell(
                    family="branching", operator="breadth_first", state_bucket="growth",
                    n=10, p_success=0.25, mean_contribution=0.15,
                ),
            },
        ),
    ]
    comparisons = compute_cross_era_comparisons(era_stats)
    assert comparisons
    stable = [c for c in comparisons if c.stable_operators]
    assert any("closure_aware" in c.stable_operators for c in stable)
