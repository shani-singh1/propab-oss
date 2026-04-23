from propab.tools.algorithm_optimization.compare_implementations import compare_implementations


def test_compare_implementations_smoke() -> None:
    r = compare_implementations(
        implementations=[{"name": "impl_a", "code": "x"}, {"name": "impl_b", "code": "y"}],
        test_inputs=[1, 2, 3],
        n_runs=3,
    )
    assert r.success
    out = r.output or {}
    assert len(out["performance"]) == 2
    assert out["fastest"] in ("impl_a", "impl_b")
