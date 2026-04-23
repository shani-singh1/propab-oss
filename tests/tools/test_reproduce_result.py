from propab.tools.ml_research.reproduce_result import reproduce_result


def test_reproduce_result_score_range() -> None:
    r = reproduce_result("#ignored", n_runs=10, fixed_seed=99)
    assert r.success
    assert 0.0 <= r.output["reproducibility_score"] <= 1.0
    assert r.output["fixed_variance"] < r.output["random_variance"] + 1e-6
