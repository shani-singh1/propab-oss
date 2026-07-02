"""Tests for campaign client terminal wait (fixes.md Step 2)."""
from integrations.astabench.campaign_client import (
    _campaign_terminal,
    default_max_wait_sec,
)


def test_terminal_on_stop_reason_only():
    assert _campaign_terminal({"status": "active", "stop_reason": "TIME_BUDGET_EXHAUSTED"})


def test_not_terminal_while_active_without_stop():
    assert not _campaign_terminal({"status": "active", "stop_reason": None})


def test_default_max_wait_includes_grace_beyond_budget():
    assert default_max_wait_sec(1.0) > 3600 + 600
