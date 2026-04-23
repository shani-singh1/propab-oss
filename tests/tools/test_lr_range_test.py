from propab.tools.deep_learning.lr_range_test import lr_range_test


def test_lr_range_test_suggested_positive() -> None:
    r = lr_range_test("my-model", lr_min=1e-5, lr_max=0.2, n_steps=30)
    assert r.success
    assert r.output["suggested_lr"] > 0
    assert len(r.output["lr_loss_curve"]) >= 2
