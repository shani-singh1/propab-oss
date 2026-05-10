from services.worker.sandbox import _sandbox_prepended_budget


def test_sandbox_prepends_wall_budget_helpers() -> None:
    hdr = _sandbox_prepended_budget(220)
    assert "SANDBOX_WALL_SEC = 220" in hdr
    assert "SANDBOX_REMAINING_SEC" in hdr
    assert "_PROPAB_T0_MONO" in hdr
