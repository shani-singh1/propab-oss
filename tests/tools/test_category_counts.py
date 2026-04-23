from propab.tools.statistics.category_counts import category_counts


def test_category_counts_full() -> None:
    r = category_counts(["x", "y", "x"])
    assert r.success
    assert r.output["counts"] == {"x": 2, "y": 1}
    assert r.output["distinct"] == 2
    assert r.output["total"] == 3


def test_category_counts_top_k() -> None:
    r = category_counts(["a", "b", "a", "c", "a"], top_k=2)
    assert r.success
    assert r.output["counts"]["a"] == 3
    assert len(r.output["counts"]) == 2
