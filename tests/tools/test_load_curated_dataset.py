from propab.tools.data_analysis.load_curated_dataset import load_curated_dataset


def test_load_synthetic_gaussian() -> None:
    r = load_curated_dataset("synthetic_gaussian", max_rows=15)
    assert r.success
    out = r.output or {}
    assert out["n_rows"] == 15
    assert "f0" in out["columns"]


def test_load_unknown_dataset() -> None:
    r = load_curated_dataset("not-a-real-id")
    assert not r.success
