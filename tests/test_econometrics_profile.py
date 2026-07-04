"""Econometrics domain profile (T3-002)."""
from propab.artifact_verification import EvidenceContext, run_artifact_gate
from propab.domain_profiles import get_profile, resolve_domain_profile


def test_resolve_econometrics_profile():
    ctx = EvidenceContext(hypothesis_text="panel FE study")
    p = resolve_domain_profile(ctx, question="[domain_profile:econometrics] wage panel fixed effects")
    assert p is not None
    assert p.profile_id == "econometrics"


def test_panel_fe_gate_confirms_with_within_group_r2():
    profile = get_profile("econometrics")
    assert profile is not None
    ctx = EvidenceContext(
        hypothesis_text="Within-group R² exceeds baseline under entity FE",
        n_samples=500,
        n_groups=10,
        p_value=0.02,
    )
    exp = {
        "within_group_r2": 0.35,
        "baseline_r2": 0.12,
        "permutation_p": 0.02,
        "fe_coefficient": 0.45,
        "verification_method": "within_group_FE",
    }
    gate = run_artifact_gate(ctx, exp, question="[domain_profile:econometrics] panel OLS")
    assert gate.verdict == "confirmed"
    assert gate.top_artifact_survived
