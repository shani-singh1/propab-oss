from propab.sandbox_profiles import effective_sandbox_timeout_sec


def test_domain_default_deep_learning() -> None:
    assert effective_sandbox_timeout_sec("deep_learning", 30) == 300


def test_unknown_domain_uses_global() -> None:
    assert effective_sandbox_timeout_sec("chemistry", 45) == 45


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_TIMEOUT_DEEP_LEARNING", "400")
    try:
        assert effective_sandbox_timeout_sec("deep_learning", 30) == 400
    finally:
        monkeypatch.delenv("SANDBOX_TIMEOUT_DEEP_LEARNING", raising=False)


def test_env_override_invalid_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_TIMEOUT_DEEP_LEARNING", "not-a-number")
    try:
        assert effective_sandbox_timeout_sec("deep_learning", 30) == 300
    finally:
        monkeypatch.delenv("SANDBOX_TIMEOUT_DEEP_LEARNING", raising=False)
