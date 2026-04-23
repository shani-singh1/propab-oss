from propab.tools.algorithm_optimization.hessian_analysis import hessian_analysis


def test_hessian_analysis_smoke() -> None:
    r = hessian_analysis(model_id="test-model", n_samples=128, top_k_eigenvalues=5)
    assert r.success
    out = r.output or {}
    assert len(out["top_eigenvalues"]) <= 5
    assert out["critical_point_type"] in ("local_minimum", "saddle_point", "local_maximum")
    assert out["condition_number"] >= 1.0
