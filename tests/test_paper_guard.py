from services.orchestrator.paper import _ensure_nonempty_trace


def test_ensure_nonempty_trace_raises_on_zero() -> None:
    try:
        _ensure_nonempty_trace(0)
    except RuntimeError as exc:
        assert "zero experiment steps" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for zero-step trace")


def test_ensure_nonempty_trace_passes_positive() -> None:
    _ensure_nonempty_trace(3)
