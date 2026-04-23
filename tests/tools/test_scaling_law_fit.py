from propab.tools.deep_learning.scaling_law_fit import scaling_law_fit


def test_scaling_law_fit_basic() -> None:
    r = scaling_law_fit(
        [
            {"model_params": 1e6, "loss": 2.5},
            {"model_params": 4e6, "loss": 2.0},
            {"model_params": 16e6, "loss": 1.7},
        ]
    )
    assert r.success
    assert "intercept" in r.output["fit_params"]
    assert r.output["r_squared"] <= 1.0
