from propab.tools.ml_research.bootstrap_confidence import bootstrap_confidence
from propab.tools.ml_research.statistical_significance import statistical_significance


def test_statistical_significance_t_test() -> None:
    r = statistical_significance([1.0, 1.1, 1.05], [0.5, 0.55, 0.52], test="t_test")
    assert r.success
    assert r.output["p_value"] < 0.05
    assert r.output["significant"] is True


def test_bootstrap_confidence_mean() -> None:
    r = bootstrap_confidence([1.0, 2.0, 3.0, 4.0], metric="mean", n_bootstrap=800)
    assert r.success
    assert r.output["ci_lower"] <= r.output["point_estimate"] <= r.output["ci_upper"]
