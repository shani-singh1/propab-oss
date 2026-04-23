from propab.tools.algorithm_optimization.gradient_noise_scale import gradient_noise_scale


def test_gradient_noise_scale() -> None:
    r = gradient_noise_scale("m", batch_sizes=[8, 16])
    assert r.success
    assert r.output["optimal_batch_size"] in (8, 16)
