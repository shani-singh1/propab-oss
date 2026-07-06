"""Tests for domain preflight enforcement (Item 1) and health-metric computation (Item 2).

These cover the load-bearing logic without a live database:
- ``_enforce_domain_preflight`` blocks a fresh campaign when the owning domain's
  preflight fails, and records a ``DOMAIN_PREFLIGHT_FAILED`` stop reason.
- The pure health-metric computations (experiment success rate, worker utilization,
  literature citation counts, confirmed-finding audit counts).
"""
from __future__ import annotations

import asyncio

from propab import health_metrics as hm
from propab.campaign import STATUS_ACTIVE, STOP_REASON_DOMAIN_PREFLIGHT_FAILED, ResearchCampaign
from propab.domain_modules import registry
from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from services.orchestrator.campaign_loop import _enforce_domain_preflight


# ── Test doubles ─────────────────────────────────────────────────────────────

class _FakeDB:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *a, **k):
        class _R:
            def scalar_one_or_none(self_inner):
                return None

            def one(self_inner):
                return (0, 0)

        return _R()

    async def commit(self):
        return None


def _fake_session_factory():
    return _FakeDB()


class _FakeEmitter:
    def __init__(self):
        self.events: list[dict] = []

    async def emit(self, **kwargs):
        self.events.append(kwargs)


class _FailPlugin(DomainPlugin):
    domain_id = "faildomain"
    display_name = "fail domain"

    def matches(self, *, question: str = "", payload=None) -> bool:
        return "faildomain" in (question or "").lower()

    def available_features(self):
        return []

    def preflight(self) -> PreflightResult:
        return PreflightResult(False, "underpowered: only 3 samples", {"n_samples": 3})


class _PassPlugin(_FailPlugin):
    domain_id = "passdomain"

    def matches(self, *, question: str = "", payload=None) -> bool:
        return "passdomain" in (question or "").lower()

    def preflight(self) -> PreflightResult:
        return PreflightResult(True, "ok", {"n_samples": 5000})


def _register(plugin: DomainPlugin):
    registry.register_plugin(plugin)


def _unregister(domain_id: str):
    registry._BY_ID.pop(domain_id, None)
    registry._PLUGINS[:] = [p for p in registry._PLUGINS if p.domain_id != domain_id]


# ── Item 1: preflight enforcement ────────────────────────────────────────────

def test_preflight_failure_blocks_campaign():
    _register(_FailPlugin())
    try:
        campaign = ResearchCampaign(
            id="00000000-0000-0000-0000-0000000000f1",
            question="probe [domain_profile:faildomain]",
            hypothesis_tree=HypothesisTree(),
        )
        proceed = asyncio.run(
            _enforce_domain_preflight(campaign, _fake_session_factory, _FakeEmitter())
        )
        assert proceed is False
        assert campaign.stop_reason == STOP_REASON_DOMAIN_PREFLIGHT_FAILED
        assert campaign.status != STATUS_ACTIVE
    finally:
        _unregister("faildomain")


def test_preflight_pass_allows_campaign():
    _register(_PassPlugin())
    try:
        campaign = ResearchCampaign(
            id="00000000-0000-0000-0000-0000000000f2",
            question="probe [domain_profile:passdomain]",
            hypothesis_tree=HypothesisTree(),
        )
        emitter = _FakeEmitter()
        proceed = asyncio.run(_enforce_domain_preflight(campaign, _fake_session_factory, emitter))
        assert proceed is True
        assert campaign.stop_reason is None
        assert campaign.status == STATUS_ACTIVE
        assert any(e.get("step") == "campaign.preflight_ok" for e in emitter.events)
    finally:
        _unregister("passdomain")


def test_preflight_no_domain_proceeds():
    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000f3",
        question="a question with no scientific domain owner at all",
        hypothesis_tree=HypothesisTree(),
    )
    proceed = asyncio.run(_enforce_domain_preflight(campaign, _fake_session_factory, _FakeEmitter()))
    assert proceed is True
    assert campaign.status == STATUS_ACTIVE


# ── Item 2: health-metric computations ───────────────────────────────────────

def _node(nid: str, verdict: str, *, reason: str | None = None, evidence: str = "") -> HypothesisNode:
    return HypothesisNode(
        id=nid, text=nid, parent_id=None, depth=0, verdict=verdict,
        inconclusive_reason=reason, evidence_summary=evidence,
    )


def test_experiment_success_rate():
    tree = HypothesisTree()
    tree.nodes = {
        "a": _node("a", "confirmed"),
        "b": _node("b", "refuted"),
        "c": _node("c", "inconclusive", reason="timeout"),      # execution failure
        "d": _node("d", "inconclusive", reason="weak signal"),  # genuine
        "e": _node("e", "pending"),                             # not tested → excluded
    }
    rate, counts = hm.compute_experiment_success_rate(tree)
    assert counts == {"definitive": 2, "tested": 4, "execution_failures": 1}
    assert rate == 0.5


def test_worker_utilization_capped_and_none():
    assert hm.compute_worker_utilization(300.0, 600.0, 2) == 0.25
    assert hm.compute_worker_utilization(10_000.0, 100.0, 1) == 1.0  # capped
    assert hm.compute_worker_utilization(100.0, 0.0, 2) is None      # no elapsed


def test_confirmed_audit_counts():
    tree = HypothesisTree()
    tree.nodes = {
        "a": _node("a", "confirmed", evidence="lofo_r2=0.4 beats label_shuffle null p95"),
        "b": _node("b", "confirmed", evidence="descriptive summary, no null test"),
        "c": _node("c", "refuted", evidence="permutation test"),  # not confirmed → ignored
    }
    confirmed, survived = hm.compute_confirmed_audit_counts(tree)
    assert confirmed == 2
    assert survived == 1


def test_count_established_verified():
    facts = [
        {"claim": "x", "doi": "10.1/x"},
        {"claim": "y", "citation": "Smith 2020"},
        {"claim": "z"},  # no citation → unverifiable
    ]
    total, verified = hm.count_established_verified(facts)
    assert total == 3
    assert verified == 2
    assert hm.count_established_verified(None) == (0, 0)


def test_rate_and_norm_helpers():
    assert hm._rate(2, 10) == 0.2
    assert hm._rate(1, 0) is None
    assert hm._norm_statement("  Hello   World  ") == "hello world"


# ── Evidence-binding health warnings (too-loose vs symmetric too-strict) ──────

class _TotalsDB(_FakeDB):
    """Fake DB whose cumulative-totals query returns configurable (rej, acc)."""

    def __init__(self, tot_rej: int, tot_acc: int):
        self._tot = (tot_rej, tot_acc)

    async def execute(self, *a, **k):
        totals = self._tot

        class _R:
            def scalar_one_or_none(self_inner):
                return None  # no previous round → stability path is a no-op

            def one(self_inner):
                return totals

        return _R()


def _run_synth_health(tot_rej: int, tot_acc: int, *, rej: int = 1, acc: int = 1):
    def factory():
        return _TotalsDB(tot_rej, tot_acc)

    return asyncio.run(
        hm.log_synthesis_health(
            factory,
            campaign_id="00000000-0000-0000-0000-0000000000aa",
            generation=1,
            metrics={
                "n_candidates_raw": 0,
                "n_added": 0,
                "n_rejected_duplicate": 0,
                "binding_rejected_count": rej,
                "binding_accepted_count": acc,
            },
            active_belief_statements=[],
        )
    )


def test_binding_too_strict_warns_at_high_cumulative_rejection(caplog):
    """Near-100% rejection over >= min calls fires the over-strict warning."""
    with caplog.at_level("WARNING", logger="propab.health_metrics"):
        _run_synth_health(tot_rej=95, tot_acc=5)  # 95% over 100 calls
    msgs = [r.getMessage() for r in caplog.records]
    assert any("over-strict" in m for m in msgs), msgs
    assert not any("0 rejections" in m for m in msgs)


def test_binding_normal_rejection_does_not_warn_too_strict(caplog):
    """A healthy mid-range rejection rate must NOT fire the over-strict warning."""
    with caplog.at_level("WARNING", logger="propab.health_metrics"):
        _run_synth_health(tot_rej=10, tot_acc=90)  # 10% over 100 calls
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("over-strict" in m for m in msgs), msgs
    assert not any("0 rejections" in m for m in msgs)


def test_binding_high_rate_below_min_calls_does_not_warn(caplog):
    """High rejection rate but too few calls must not fire (mirror the zero-check gate)."""
    with caplog.at_level("WARNING", logger="propab.health_metrics"):
        _run_synth_health(tot_rej=9, tot_acc=1)  # 90% but only 10 calls < 50
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("over-strict" in m for m in msgs), msgs


def test_binding_zero_rejection_still_warns_too_loose(caplog):
    """The pre-existing too-loose warning is unchanged and mutually exclusive."""
    with caplog.at_level("WARNING", logger="propab.health_metrics"):
        _run_synth_health(tot_rej=0, tot_acc=60)  # 0 rejections over 60 calls
    msgs = [r.getMessage() for r in caplog.records]
    assert any("0 rejections" in m for m in msgs), msgs
    assert not any("over-strict" in m for m in msgs)
