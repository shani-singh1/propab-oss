from propab.metric_normalize import normalize_accuracy_metric


def test_accuracy_percentage_rescale() -> None:
    v = normalize_accuracy_metric("val_accuracy", 95.2)
    assert v is not None
    assert abs(v - 0.952) < 1e-9


def test_accuracy_fraction_unchanged() -> None:
    v = normalize_accuracy_metric("val_accuracy", 0.952)
    assert v is not None
    assert abs(v - 0.952) < 1e-9


def test_non_accuracy_untouched() -> None:
    assert normalize_accuracy_metric("val_loss", 2.5) == 2.5
