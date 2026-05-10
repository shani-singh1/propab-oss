from propab.tools.algorithm_optimization.compare_gradient_methods import compare_gradient_methods


def test_compare_gradient_methods_long_adam_run_stable_bias() -> None:
    """Regression: long Adam runs should not fail in bias correction / step updates."""
    r = compare_gradient_methods(
        methods=["adam"],
        learning_rate=0.01,
        n_steps=5500,
        init_point=[-1.0, 1.0],
    )
    assert r.success
    out = r.output or {}
    assert len((out["trajectories"] or [{}])[0].get("loss_curve", [])) >= 1


def test_compare_gradient_methods_empty_methods_validation() -> None:
    r = compare_gradient_methods(methods=[])
    assert not r.success
