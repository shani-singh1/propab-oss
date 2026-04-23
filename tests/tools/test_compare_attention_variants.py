from propab.tools.deep_learning.compare_attention_variants import compare_attention_variants


def test_compare_attention_variants_smoke() -> None:
    r = compare_attention_variants(variants=["standard", "linear"], seq_lengths=[16, 32], d_model=64, n_heads=2)
    assert r.success
    out = r.output or {}
    assert len(out["comparison"]) == 4
    assert set(out["pareto_front"]).issubset({"standard", "linear"})


def test_compare_attention_variants_rejects_unknown() -> None:
    r = compare_attention_variants(variants=["nope"], seq_lengths=[8])
    assert not r.success
