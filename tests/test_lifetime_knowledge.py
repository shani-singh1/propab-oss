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


def _spectral_claims():
    return extract_confirmed_claims("cid", [
        {"claim": "a", "verdict": "confirmed", "primary_theme": "spectral", "node_role": "DISCOVERY"},
        {"claim": "b", "verdict": "confirmed", "primary_theme": "spectral", "node_role": "DISCOVERY"},
    ])


def test_ll5_non_network_theme_is_domain_neutral():
    """A `spectral` theme in a math/materials campaign gets no contagion framing."""
    theories = form_theories_from_claims(
        _spectral_claims(), min_support=2, domain="math_combinatorics",
    )
    assert len(theories) == 1
    th = theories[0]
    assert th.name == "spectral_theory"
    blob = " ".join([th.name, *th.assumptions]).lower()
    assert "contagion" not in blob
    assert "diffusion" not in blob


def test_ll5_no_domain_defaults_to_domain_neutral():
    """Without a domain hint, a non-diffusion theme still avoids contagion wording."""
    theories = form_theories_from_claims(_spectral_claims(), min_support=2)
    th = theories[0]
    blob = " ".join([th.name, *th.assumptions]).lower()
    assert "contagion" not in blob and "diffusion" not in blob


def test_ll5_network_diffusion_theme_keeps_framing():
    """A genuine network-diffusion campaign still gets contagion/diffusion framing."""
    theories = form_theories_from_claims(
        _spectral_claims(), min_support=2, domain="network_diffusion",
    )
    th = theories[0]
    assert th.name == "spectral_contagion_theory"
    blob = " ".join(th.assumptions).lower()
    assert "diffusion" in blob


def test_ll5_diffusion_theme_keeps_framing_without_domain():
    """A theme that is itself about diffusion keeps network framing even w/o domain."""
    claims = extract_confirmed_claims("cid", [
        {"claim": "a", "verdict": "confirmed", "primary_theme": "diffusion_dynamics", "node_role": "DISCOVERY"},
        {"claim": "b", "verdict": "confirmed", "primary_theme": "diffusion_dynamics", "node_role": "DISCOVERY"},
    ])
    theories = form_theories_from_claims(claims, min_support=2)
    assert theories[0].name == "diffusion_dynamics_contagion_theory"


def test_ll6_underevidenced_claim_not_promoted():
    """A T1, confidence-0 single-campaign claim is NOT an established fact."""
    graph = KnowledgeGraph()
    graph.add_claim(Claim(
        id="weak-1",
        text="Weak unreplicated claim",
        verdict="confirmed",
        theme="spectral",
        confidence=0.0,
        replication_level="T1",
        campaign_id="camp-weak",
    ))
    assert graph.established_fact_texts() == []


def test_ll6_replicated_claim_promoted_with_provenance():
    """A T2 (replicated) claim IS promoted and carries traceable provenance."""
    graph = KnowledgeGraph()
    graph.add_claim(Claim(
        id="strong-1",
        text="Replicated claim",
        verdict="confirmed",
        theme="spectral",
        confidence=0.3,  # low confidence but replicated → still promoted
        replication_level="T2",
        campaign_id="camp-strong",
    ))
    facts = graph.established_fact_texts()
    assert len(facts) == 1
    assert facts[0]["paper_ids"]  # non-empty provenance, not []
    assert "camp-strong" in facts[0]["paper_ids"]
    assert facts[0]["campaign_id"] == "camp-strong"
    assert facts[0]["claim_id"] == "strong-1"


def test_ll6_high_confidence_t1_promoted():
    """A T1 claim above the confidence floor is still promoted (OR-gate)."""
    graph = KnowledgeGraph()
    graph.add_claim(Claim(
        id="hc-1",
        text="High-confidence single-run claim",
        verdict="confirmed",
        theme="spectral",
        confidence=0.85,
        replication_level="T1",
        campaign_id="camp-hc",
    ))
    facts = graph.established_fact_texts()
    assert len(facts) == 1
    assert "camp-hc" in facts[0]["paper_ids"]


def test_ll6_gate_filters_mix():
    """Established facts exclude the weak claim and include the strong one."""
    graph = KnowledgeGraph()
    graph.add_claim(Claim(
        id="weak", text="weak", verdict="confirmed", theme="t",
        confidence=0.0, replication_level="T1", campaign_id="cw",
    ))
    graph.add_claim(Claim(
        id="strong", text="strong", verdict="confirmed", theme="t",
        confidence=0.9, replication_level="T3", campaign_id="cs",
    ))
    texts = {f["text"] for f in graph.established_fact_texts()}
    assert texts == {"strong"}
