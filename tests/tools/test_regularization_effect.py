from propab.tools.algorithm_optimization.regularization_effect import regularization_effect


def test_regularization_effect_runs() -> None:
    r = regularization_effect("m1", ["none", "l2", "dropout"], n_steps=50)
    assert r.success
    assert r.output["best_strategy"] in ("none", "l2", "dropout")
