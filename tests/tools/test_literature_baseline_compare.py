from propab.tools.ml_research.literature_baseline_compare import literature_baseline_compare


def test_literature_baseline_lower_is_better() -> None:
    r = literature_baseline_compare(
        our_results=[0.4, 0.41, 0.39],
        baseline_value=0.5,
        metric_direction="lower_is_better",
    )
    assert r.success
    out = r.output or {}
    assert out["improvement_pct"] > 0
    assert "our_mean" in out


def test_literature_baseline_with_std() -> None:
    r = literature_baseline_compare(
        our_results=[1.0, 1.02, 0.99],
        baseline_value=0.9,
        baseline_std=0.05,
        metric_direction="higher_is_better",
    )
    assert r.success
