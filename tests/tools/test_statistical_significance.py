from propab.tools.ml_research.statistical_significance import statistical_significance


def test_statistical_significance_rejects_identical_replicates() -> None:
    r = statistical_significance(
        results_a=[1.0, 1.0, 1.0],
        results_b=[2.0, 2.0, 2.0],
        test="t_test",
    )
    assert not r.success
    assert r.error is not None
    assert r.error.type == "zero_variance"


def test_statistical_significance_wilcoxon_identical_pairs() -> None:
    r = statistical_significance(
        results_a=[0.5, 0.5, 0.5],
        results_b=[0.5, 0.5, 0.5],
        test="wilcoxon",
    )
    assert not r.success
    assert r.error is not None
