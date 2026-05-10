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


def test_statistical_significance_rejects_known_placeholder_vector() -> None:
    leak = [
        79.648735,
        79.648735,
        79.648735,
        54.044083,
        54.044083,
        54.044083,
        63.826187,
        63.826187,
        63.826187,
    ]
    other = [1.0, 1.01, 1.02, 2.0, 2.01, 2.02, 3.0, 3.01, 3.02]
    r = statistical_significance(leak, other)
    assert r.success is False
    assert r.error is not None
