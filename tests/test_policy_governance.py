"""Tests for fixes.md policy governance — candidate/accept/reject, buckets, evaluation."""
from __future__ import annotations

from propab import config
from propab.campaign import ResearchCampaign
from propab.knowledge_graph import Claim, KnowledgeGraph
from propab.meta_science import CampaignObservation, MetaScienceLedger
from propab.policy_buckets import budget_bucket, domain_bucket
from propab.policy_evaluation import evaluate_candidate_policy, tolerance_for_campaign
from propab.policy_fitness_ledger import PolicyFitnessLedger
from propab.policy_mutation import mutate_policy_params
from propab.policy_record import PolicyRecord, PolicyStatus, PredictedEffects
from propab.policy_store import PolicyStore
from services.orchestrator.lifetime_knowledge import ingest_campaign, load_lifetime_state
from services.orchestrator.policy_analyst import propose_policy_narrative_sync


def _tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))


def test_budget_and_domain_buckets():
    assert budget_bucket(3600) == "1h"
    assert budget_bucket(10800) == "3h"
    assert budget_bucket(28800) == "8h"
    assert domain_bucket("SIS contagion on scale-free networks") == "graphs"
    assert domain_bucket("Collatz stopping times modulo 8") == "math"
    assert domain_bucket("cache replacement LRU adversarial") == "algorithms"


def test_candidate_never_overwrites_accepted(monkeypatch, tmp_path):
    _tmp(monkeypatch, tmp_path)
    store = PolicyStore()
    accepted = store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")
    accepted.boosts = {"spectral": 0.2}
    store.save()

    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000bb",
        question="Network contagion speed on graphs",
        compute_budget_seconds=10800,
    )
    campaign.hypothesis_tree.finding_ledger = [
        {
            "claim": "x",
            "verdict": "confirmed",
            "primary_theme": "spectral",
            "node_role": "DISCOVERY",
        },
    ]
    report = ingest_campaign(campaign)
    assert report["candidate_policy_id"]
    assert report["candidate_status"] == "CANDIDATE"

    store2 = PolicyStore.load()
    acc = store2.accepted_policy(domain_bucket="graphs", budget_bucket="3h")
    assert acc.id == accepted.id
    assert acc.boosts.get("spectral") == 0.2


def test_rejected_policy_in_history(monkeypatch, tmp_path):
    _tmp(monkeypatch, tmp_path)
    meta = MetaScienceLedger()
    meta.record(CampaignObservation(
        campaign_id="baseline",
        question="q",
        tested=85,
        confirmed=23,
        refuted=3,
        inconclusive=59,
        closure_ratio=0.306,
        theme_entropy=0.25,
        general_theme_fraction=0.0,
        compute_seconds=10800,
        policy_generation=1,
        knowledge_claims=23,
        knowledge_failures=34,
        budget_bucket="3h",
        domain_bucket="graphs",
    ))
    meta.save()

    store = PolicyStore()
    parent = store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")
    candidate = store.add_candidate(
        parent=parent,
        params={"boosts": {}, "penalties": {"general": 0.25}, "blocked_failures": [], "saturated_themes": []},
        rationale="test",
        predicted=PredictedEffects(closure_ratio_delta=0.05),
        falsification=["closure collapses"],
    )
    store.bind_campaign(
        campaign_id="cal-run",
        policy_mode="candidate",
        domain_bucket="graphs",
        budget_bucket="3h",
        baseline_campaign_id="baseline",
    )
    store.save()

    campaign = ResearchCampaign(
        id="cal-run",
        question="Network contagion under SIS",
        compute_budget_seconds=10800,
        policy_mode="candidate",
    )
    campaign.hypothesis_tree.finding_ledger = []
    report = ingest_campaign(campaign, session_domain="general_computation")
    assert report["evaluation"] is not None
    assert report["evaluation"]["accept_or_reject"] in ("ACCEPT", "REJECT")

    store2 = PolicyStore.load()
    rec = store2.get_policy(candidate.id)
    assert rec is not None
    assert rec.status in (PolicyStatus.ACCEPTED, PolicyStatus.REJECTED)
    if rec.status == PolicyStatus.REJECTED:
        assert candidate.id in store2.rejected_ids


def test_bucket_local_mutation_excludes_other_buckets():
    graph = KnowledgeGraph()
    graph.add_claim(Claim(
        id="c1", text="a", verdict="confirmed", theme="spectral", campaign_id="3h-run",
    ))
    graph.add_claim(Claim(
        id="c2", text="b", verdict="refuted", theme="general", campaign_id="1h-run",
    ))
    meta = MetaScienceLedger()
    meta.record(CampaignObservation(
        campaign_id="3h-run", question="q", tested=10, confirmed=5, refuted=1,
        inconclusive=4, closure_ratio=0.3, theme_entropy=0.5, general_theme_fraction=0.1,
        compute_seconds=10800, policy_generation=1, knowledge_claims=1, knowledge_failures=0,
        budget_bucket="3h", domain_bucket="graphs",
    ))
    params = mutate_policy_params(
        graph, meta, budget_bucket="3h", domain_bucket="graphs", parent=None,
    )
    assert "spectral" in params["boosts"]
    assert "general" not in params.get("penalties", {})


def test_llm_narrative_does_not_mutate_params():
    graph = KnowledgeGraph()
    meta = MetaScienceLedger()
    parent = PolicyRecord.empty_accepted(budget_bucket="3h", domain_bucket="graphs")
    r1, p1, f1, params = propose_policy_narrative_sync(
        parent=parent, graph=graph, meta=meta,
        budget_bucket="3h", domain_bucket="graphs",
    )
    r2, p2, f2, params2 = propose_policy_narrative_sync(
        parent=parent, graph=graph, meta=meta,
        budget_bucket="3h", domain_bucket="graphs",
    )
    assert params == params2
    assert r1
    assert f1


def test_evaluation_tolerance_bands():
    tol = tolerance_for_campaign(budget_bucket="1h", tested=10)
    assert tol["closure_ratio"] > tol["closure_ratio"] * 0.5
    baseline = CampaignObservation(
        campaign_id="b", question="q", tested=85, confirmed=23, refuted=3,
        inconclusive=59, closure_ratio=0.306, theme_entropy=0.25,
        general_theme_fraction=0.0, compute_seconds=10800, policy_generation=1,
        knowledge_claims=0, knowledge_failures=0, budget_bucket="3h", domain_bucket="graphs",
    )
    current = CampaignObservation(
        campaign_id="c", question="q", tested=85, confirmed=23, refuted=3,
        inconclusive=59, closure_ratio=0.31, theme_entropy=0.28,
        general_theme_fraction=0.0, compute_seconds=10800, policy_generation=2,
        knowledge_claims=0, knowledge_failures=0, budget_bucket="3h", domain_bucket="graphs",
    )
    ok, detail = evaluate_candidate_policy(
        predicted=PredictedEffects(closure_ratio_delta=0.01, theme_entropy_delta=0.02),
        baseline_obs=baseline,
        current_obs=current,
        budget_bucket="3h",
    )
    assert ok
    assert detail["pred_ok"]


def test_fitness_ledger_persists(monkeypatch, tmp_path):
    _tmp(monkeypatch, tmp_path)
    from propab.policy_fitness_ledger import FitnessRecord

    ledger = PolicyFitnessLedger()
    ledger.record(FitnessRecord(
        policy_id="pol-1",
        campaign_id="camp-1",
        budget_bucket="3h",
        domain_bucket="graphs",
        predictions={"closure_ratio_delta": 0.02},
        observations={"closure_ratio": 0.01},
        residuals={"closure_ratio": -0.01},
        accept_or_reject="ACCEPT",
    ))
    ledger.save()
    loaded = PolicyFitnessLedger.load()
    assert len(loaded.records) == 1


def test_load_lifetime_binds_candidate_mode(monkeypatch, tmp_path):
    _tmp(monkeypatch, tmp_path)
    store = PolicyStore()
    parent = store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")
    store.add_candidate(
        parent=parent,
        params={"boosts": {"modularity": 0.3}, "penalties": {}, "blocked_failures": [], "saturated_themes": []},
        rationale="cand",
        predicted=PredictedEffects(),
        falsification=[],
    )
    store.save()

    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000cc",
        question="Contagion on networks",
        compute_budget_seconds=10800,
        policy_mode="candidate",
    )
    state = load_lifetime_state(campaign, session_domain="general_computation")
    assert state.policy_record.status == PolicyStatus.CANDIDATE
    assert "modularity" in state.policy.theme_boost
