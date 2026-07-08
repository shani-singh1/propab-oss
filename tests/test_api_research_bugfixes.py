"""Regression tests for latent API-layer bugs in services/api/app/routes/research.py.

Covered:
- Bug 1: a plain resume (empty body) rebases the budget clock so the resumed run
  actually executes instead of no-opping as ``budget_exhausted``.
- Bug 3: an EMPTY injected literature prior is labeled ``INSUFFICIENT_EVIDENCE``,
  not stamped ``READY``.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import BackgroundTasks

import services.api.app.routes.research as research
from propab.campaign import (
    STATUS_ACTIVE,
    STATUS_BUDGET_EXHAUSTED,
    ResearchCampaign,
)
from services.orchestrator.campaign_loop import _derive_prior_evidence_status


# ── Test doubles ─────────────────────────────────────────────────────────────

class _FakeDB:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *a, **k):
        return None

    async def commit(self):
        return None


def _fake_session_factory():
    return _FakeDB()


class _FakeEmitter:
    def __init__(self):
        self.events: list[dict] = []

    async def emit(self, **kwargs):
        self.events.append(kwargs)


# ── Bug 1: plain resume rebases the budget clock ─────────────────────────────

def test_plain_resume_rebases_budget_clock_and_runs(monkeypatch):
    """A resume with an empty body must rebase started_at so should_stop() is False.

    The pre-fix code only rebased ``started_at`` inside ``if budget_hours is not
    None``. Budget is enforced by wall clock (should_stop -> elapsed_seconds =
    now - started_at), so a campaign launched long ago and resumed with an empty
    body kept its original ``started_at``, saw ``elapsed >> budget``, and the
    ``while not should_stop()`` loop body never ran — a silent no-op.
    """
    # Campaign launched in 2020 with a 1-hour budget, currently stopped.
    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-00000000c0de",
        question="q?",
        compute_budget_seconds=3600,
        started_at="2020-01-01T00:00:00+00:00",
        status=STATUS_BUDGET_EXHAUSTED,
    )
    # Sanity: before resume this campaign would stop immediately.
    campaign.status = STATUS_ACTIVE
    assert campaign.should_stop() is True
    campaign.status = STATUS_BUDGET_EXHAUSTED

    dispatched: dict = {}

    async def _fake_load(cid, session_factory):
        return campaign

    async def _fake_save(c, session_factory):
        return None

    async def _fake_events(cid, session_factory, limit=2000):
        return []

    async def _fake_dispatch(*, campaign, **kwargs):
        # Capture the campaign state at dispatch time — this is exactly what the
        # loop would see on entry.
        dispatched["should_stop"] = campaign.should_stop()
        dispatched["started_at"] = campaign.started_at

    monkeypatch.setattr(research, "db_load_campaign", _fake_load)
    monkeypatch.setattr(research, "db_save_campaign", _fake_save)
    monkeypatch.setattr(research, "db_load_session_events_tail", _fake_events)
    monkeypatch.setattr(research, "_dispatch_campaign", _fake_dispatch)

    resp = asyncio.run(
        research.resume_campaign(
            campaign_id=campaign.id,
            background_tasks=BackgroundTasks(),
            request=research.CampaignResumeRequest(),  # empty body
            emitter=_FakeEmitter(),
            session_factory=_fake_session_factory,
        )
    )

    assert resp.status == "resumed"
    # The core assertion: the resumed run is NOT already stopped. Pre-fix this was
    # True (elapsed since 2020 >> 3600s budget) and the loop no-op'd.
    assert dispatched["should_stop"] is False
    # started_at was rebased away from the original 2020 launch time.
    assert dispatched["started_at"] != "2020-01-01T00:00:00+00:00"
    assert campaign.should_stop() is False


def test_resume_with_new_budget_also_rebases_clock(monkeypatch):
    """A resume that DOES supply a new budget still rebases and runs (unchanged path)."""
    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-00000000beef",
        question="q?",
        compute_budget_seconds=3600,
        started_at="2019-06-01T00:00:00+00:00",
        status=STATUS_BUDGET_EXHAUSTED,
    )
    dispatched: dict = {}

    async def _fake_load(cid, session_factory):
        return campaign

    async def _noop(*a, **k):
        return None

    async def _fake_events(cid, session_factory, limit=2000):
        return []

    async def _fake_dispatch(*, campaign, **kwargs):
        dispatched["should_stop"] = campaign.should_stop()
        dispatched["budget"] = campaign.compute_budget_seconds

    monkeypatch.setattr(research, "db_load_campaign", _fake_load)
    monkeypatch.setattr(research, "db_save_campaign", _noop)
    monkeypatch.setattr(research, "db_load_session_events_tail", _fake_events)
    monkeypatch.setattr(research, "_dispatch_campaign", _fake_dispatch)

    asyncio.run(
        research.resume_campaign(
            campaign_id=campaign.id,
            background_tasks=BackgroundTasks(),
            request=research.CampaignResumeRequest(compute_budget_hours=2.0),
            emitter=_FakeEmitter(),
            session_factory=_fake_session_factory,
        )
    )
    assert dispatched["budget"] == 2 * 3600
    assert dispatched["should_stop"] is False


# ── Bug 3: empty injected prior is INSUFFICIENT_EVIDENCE, not READY ──────────

def test_empty_injected_prior_is_insufficient_evidence():
    """An injected prior with empty content and no evidence_status is NOT READY."""
    empty_prior = {
        "established_facts": [],
        "contested_claims": [],
        "open_gaps": [],
        "dead_ends": [],
        "key_papers": [],
    }
    assert _derive_prior_evidence_status(empty_prior) == "INSUFFICIENT_EVIDENCE"
    # Even a totally bare dict must not be stamped READY.
    assert _derive_prior_evidence_status({}) == "INSUFFICIENT_EVIDENCE"


def test_nonempty_injected_prior_is_ready():
    """A prior carrying real facts (and no explicit status) is READY."""
    prior = {"established_facts": [{"text": "X predicts Y"}]}
    assert _derive_prior_evidence_status(prior) == "READY"


def test_explicit_evidence_status_is_honored():
    """An explicitly declared status is always preserved verbatim."""
    assert (
        _derive_prior_evidence_status({"evidence_status": "READY"})
        == "READY"
    )
    assert (
        _derive_prior_evidence_status(
            {"evidence_status": "PARTIAL", "established_facts": [{"text": "z"}]}
        )
        == "PARTIAL"
    )


# ── Bug 6: an invalid breakthrough direction is rejected, not silently mis-scored ──

def test_invalid_breakthrough_direction_is_rejected():
    """A typo'd direction must 422 at the boundary, not silently score backwards.

    campaign.py treats any non-'higher_is_better' string as lower_is_better in its
    sort key but matches NEITHER in record detection, so 'higher' (a plausible typo)
    would either invert scoring or register no records at all — a silently broken run.
    """
    from pydantic import ValidationError

    # Valid values still construct.
    for good in ("higher_is_better", "lower_is_better"):
        assert research.BreakthroughCriteriaRequest(direction=good).direction == good

    for bad in ("higher", "lower", "higher_is_worse", "maximize", ""):
        with pytest.raises(ValidationError):
            research.BreakthroughCriteriaRequest(direction=bad)
