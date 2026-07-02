"""Tests for V1 domain profiles (fixes.md Step 3)."""
from propab.artifact_verification import EvidenceContext, run_artifact_gate
from propab.domain_profiles import get_profile, resolve_domain_profile


def test_resolve_from_question_tag():
    ctx = EvidenceContext(hypothesis_text="test")
    p = resolve_domain_profile(ctx, question="[domain_profile:materials] band gap study")
    assert p is not None
    assert p.profile_id == "materials"


def test_resolve_from_payload():
    ctx = EvidenceContext(hypothesis_text="x")
    p = resolve_domain_profile(ctx, payload={"domain_profile": "enzyme_kinetics"})
    assert p is not None
    assert p.profile_id == "enzyme_kinetics"


def test_artifact_gate_uses_profile():
    profile = get_profile("graph_invariants")
    assert profile is not None
    ctx = EvidenceContext(
        hypothesis_text="clustering coefficient correlates with degree across SNAP categories",
        n_samples=500,
        n_groups=5,
        p_value=0.01,
    )
    gate = run_artifact_gate(ctx, None, question="[domain_profile:graph_invariants] SNAP study")
    assert gate.verdict in ("confirmed", "refuted", "inconclusive")
    assert gate.ranked_artifacts
