"""Paper ledger respects deterministic verification (fixes.md P0.2)."""

from propab.paper_compiler import _effective_verdict


def test_confirmed_without_metric_but_with_verification_stays_confirmed() -> None:
    row = {
        "step_count": 3,
        "verdict": "confirmed",
        "evidence_summary": (
            'evidence={"verified_true_steps": 2, "n_metric_steps": 0, "verdict_reason": "scan"}; '
            "significance={}; plan_origin=heuristic; any_success=True; sandbox_ok=True; steps=3."
        ),
    }
    assert _effective_verdict(row) == "confirmed"


def test_confirmed_without_metric_or_verification_downgraded() -> None:
    row = {
        "step_count": 1,
        "verdict": "confirmed",
        "evidence_summary": 'evidence={"n_metric_steps": 0}; significance={}; steps=1.',
    }
    assert _effective_verdict(row) == "inconclusive"
