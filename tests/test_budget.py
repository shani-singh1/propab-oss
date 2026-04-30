from __future__ import annotations

import time

import pytest

from services.orchestrator.budget import ResearchBudget


def test_not_exhausted_initially():
    budget = ResearchBudget(max_rounds=5, max_hours=1.0, max_hypotheses_total=50)
    assert budget.exhausted() is False


def test_exhausted_by_rounds():
    budget = ResearchBudget(max_rounds=2)
    budget.record_round(confirmed=0, refuted=0, inconclusive=2, n_hypotheses=2)
    budget.record_round(confirmed=0, refuted=0, inconclusive=2, n_hypotheses=2)
    assert budget.exhausted() is True
    assert "round" in budget.stop_reason()


def test_exhausted_by_confirmed_target():
    budget = ResearchBudget(target_confirmed=2)
    budget.record_round(confirmed=3, refuted=0, inconclusive=0, n_hypotheses=3)
    assert budget.exhausted() is True
    assert "confirmed" in budget.stop_reason()


def test_exhausted_by_hypothesis_total():
    budget = ResearchBudget(max_hypotheses_total=3)
    budget.record_round(confirmed=0, refuted=0, inconclusive=3, n_hypotheses=3)
    assert budget.exhausted() is True
    assert "hypothesis" in budget.stop_reason()


def test_stale_rounds_accumulate():
    budget = ResearchBudget(max_stale_rounds=2)
    budget.record_round(confirmed=0, refuted=1, inconclusive=2, n_hypotheses=3)
    assert budget.stale_rounds == 1
    budget.record_round(confirmed=0, refuted=0, inconclusive=2, n_hypotheses=2)
    assert budget.stale_rounds == 2
    assert budget.exhausted() is True


def test_stale_rounds_reset_on_confirmed():
    budget = ResearchBudget(max_stale_rounds=2)
    budget.record_round(confirmed=0, refuted=1, inconclusive=2, n_hypotheses=3)
    assert budget.stale_rounds == 1
    budget.record_round(confirmed=1, refuted=0, inconclusive=1, n_hypotheses=2)
    assert budget.stale_rounds == 0


def test_round_budget_returns_budget():
    budget = ResearchBudget(max_rounds=5, agent_max_steps=12, agent_min_steps=4)
    rb = budget.round_budget(0)
    assert rb.max_hypotheses > 0
    assert rb.agent_budget.max_steps == 12
    assert rb.agent_budget.min_steps == 4


def test_summary_keys():
    budget = ResearchBudget()
    summary = budget.summary()
    assert "rounds_completed" in summary
    assert "elapsed_sec" in summary
    assert "remaining_sec" in summary


def test_to_dict_serializable():
    import json
    budget = ResearchBudget()
    d = budget.to_dict()
    json.dumps(d)  # should not raise
