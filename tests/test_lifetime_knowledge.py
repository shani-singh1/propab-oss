"""Tests for roadmap lifetime knowledge + policy persistence."""
from __future__ import annotations

from propab import config
from propab.campaign import ResearchCampaign
from propab.knowledge_graph import Claim, FailureRecord, KnowledgeGraph, new_id
from propab.meta_science import MetaScienceLedger
from propab.negative_knowledge import extract_confirmed_claims, extract_failures_from_campaign
from propab.policy_store import PolicyStore
from propab.search_policy import update_policy_from_graph
from propab.theory_objects import form_theories_from_claims
from services.orchestrator.lifetime_knowledge import (
    enrich_prior_from_lifetime,
    ingest_campaign,
    lifetime_context_for_seeds,
    load_lifetime_state,
)


def _tmp_lifetime(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))


def test_ingest_campaign_builds_claims_and_candidate(monkeypatch, tmp_path):
    _tmp_lifetime(monkeypatch, tmp_path)
    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000aa",
        question="Which network metrics predict contagion speed on graphs?",
        compute_budget_seconds=10800,
    )
    campaign.hypothesis_tree.finding_ledger = [
        {
            "claim_id": "c1",
            "claim": "Spectral gap predicts outbreak speed on scale-free graphs",
            "verdict": "confirmed",
            "primary_theme": "spectral",
            "confidence": 0.9,
            "replication_level": "T2",
            "node_role": "DISCOVERY",
        },
        {
            "claim_id": "c2",
            "claim": "K-shell index correlates with peak infection under SIS",
            "verdict": "confirmed",
            "primary_theme": "spectral",
            "confidence": 0.85,
            "replication_level": "T2",
            "node_role": "DISCOVERY",
        },
    ]
    report = ingest_campaign(campaign)
    assert report["claims_added"] == 2
    assert report["candidate_policy_id"]
    assert report["candidate_status"] == "CANDIDATE"

    graph = KnowledgeGraph.load()
    store = PolicyStore.load()
    assert len(graph.claims) == 2
    assert campaign.id in graph.campaign_ids
    assert len(store.policies) >= 2  # accepted + candidate


def test_campaign_nplus1_prior_differs(monkeypatch, tmp_path):
    _tmp_lifetime(monkeypatch, tmp_path)
    graph = KnowledgeGraph()
    graph.add_claim(Claim(
        id=new_id("claim"),
        text="Modularity slows cross-community contagion",
        verdict="confirmed",
        theme="modularity",
        confidence=0.88,
    ))
    graph.add_failure(FailureRecord(
        id=new_id("fail"),
        text="Random rewiring alone explains outbreak size",
        reason="refuted",
        failure_signature="sig-random",
        theme="random_graph",
        verdict="refuted",
    ))
    graph.save()
    store = PolicyStore()
    rec = store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")
    rec.boosts = {"modularity": 0.2}
    store.save()
    policy = rec.to_search_policy()

    prior = enrich_prior_from_lifetime(
        {"established_facts": [], "dead_ends": []},
        graph,
        policy,
        policy_record=rec,
    )
    assert any("Modularity" in f["text"] for f in prior["established_facts"])
    assert any("Random rewiring" in d["text"] for d in prior["dead_ends"])
    assert prior["policy_status"] == "ACCEPTED"
    ctx = lifetime_context_for_seeds(graph, policy)
    assert "modularity" in ctx.lower()


def test_extract_failures_skips_controls():
    nodes = {
        "n1": {
            "text": "Hub degree drives IC spread",
            "verdict": "refuted",
            "node_role": "DISCOVERY",
            "primary_theme": "degree",
        },
        "n2": {
            "text": "baseline control",
            "verdict": "inconclusive",
            "node_role": "CONTROL",
            "primary_theme": "general",
        },
    }
    failures = extract_failures_from_campaign("cid", nodes)
    assert len(failures) == 1
    assert failures[0].verdict == "refuted"


def test_legacy_update_policy_from_graph_still_works():
    graph = KnowledgeGraph()
    for i in range(4):
        graph.add_claim(Claim(
            id=f"c{i}",
            text=f"claim {i}",
            verdict="confirmed",
            theme="spectral",
            confidence=0.8,
        ))
    from propab.search_policy import SearchPolicy
    policy = SearchPolicy()
    updated = update_policy_from_graph(policy, graph, campaign_metrics={"closure_ratio": 0.3})
    assert updated.generation == 1
    assert updated.theme_boost.get("spectral", 0) > 0


def test_meta_science_learning_curve(monkeypatch, tmp_path):
    _tmp_lifetime(monkeypatch, tmp_path)
    from propab.meta_science import CampaignObservation

    meta = MetaScienceLedger()
    meta.record(CampaignObservation(
        campaign_id="a",
        question="q1",
        tested=10,
        confirmed=3,
        refuted=1,
        inconclusive=6,
        closure_ratio=0.3,
        theme_entropy=1.2,
        general_theme_fraction=0.2,
        compute_seconds=100,
        policy_generation=1,
        knowledge_claims=3,
        knowledge_failures=1,
        budget_bucket="3h",
        domain_bucket="graphs",
    ))
    meta.save()
    loaded = MetaScienceLedger.load()
    curve = loaded.learning_curve(budget_bucket="3h", domain_bucket="graphs")
    assert curve["closure_ratio"] == [0.3]
    assert curve["confirmed_rate"][0] == 0.3


def test_theories_need_min_support():
    claims = extract_confirmed_claims("cid", [
        {"claim": "a", "verdict": "confirmed", "primary_theme": "spectral", "node_role": "DISCOVERY"},
        {"claim": "b", "verdict": "confirmed", "primary_theme": "spectral", "node_role": "DISCOVERY"},
    ])
    theories = form_theories_from_claims(claims, min_support=2)
    assert len(theories) == 1
    assert "spectral" in theories[0].name
