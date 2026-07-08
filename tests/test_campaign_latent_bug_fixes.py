"""Regression tests for four HIGH/MEDIUM latent bugs in the campaign core.

Each test fails against the pre-fix code and passes after the fix:

1. ``min_confirmed_findings`` breakthrough path was DEAD — the producer never injected
   ``confirmed_nodes`` into the finding, so a baseline≈0 verification/math campaign could
   never declare BREAKTHROUGH on confirmed-count alone.
2. A silently-FAILED ML baseline (worker produced no metric / exception / timeout) used to
   return ``0.0``, collapsing an ML campaign into the verification frame and handing a
   zero-gain run a false breakthrough. ``measure_baseline`` now returns ``None`` on failure
   and the criteria refuse the verification frame when ``baseline_failed`` is set.
3. ``lower_is_better`` treated a real ``0.0`` optimum as "unset" and a legit ``0.0``
   ``metric_value`` was dropped as falsy. Both now use explicit ``is None`` sentinels.
4. ``_is_ml_campaign`` default is inverted: a campaign that resolves to a domain plugin is
   NON-ML unless the plugin explicitly asserts ``is_ml is True`` — even when the plugin's
   ``objective_spec()`` is ``None``.
"""
from __future__ import annotations

import asyncio

from propab.campaign import BreakthroughCriteria, ResearchCampaign


# ── Bug 1: min_confirmed_findings breakthrough via the producer ──────────────

def test_min_confirmed_findings_fires_after_n_confirms_via_producer() -> None:
    """A baseline=0 campaign with min_confirmed_findings=N declares breakthrough once N
    distinct confirmed findings exist — because the producer injects confirmed_nodes."""
    from services.orchestrator.campaign_loop import build_breakthrough_finding

    crit = BreakthroughCriteria(
        metric_name="sidon_density",
        baseline_value=0.0,
        min_confidence=0.85,
        min_replications=3,
        min_confirmed_findings=5,
    )
    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000b1",
        question="Maximize Sidon set density [domain_profile:math_combinatorics]",
        breakthrough_criteria=crit,
        compute_budget_seconds=60,
    )

    # A deliberately LOW-confidence, parentless (reps==1) confirmed finding: only the
    # confirmed-count frame can grant breakthrough, so this isolates the fixed path.
    result = {"verdict": "confirmed", "confidence": 0.40}

    campaign.total_confirmed = 4
    finding = build_breakthrough_finding(campaign, "node-4", result)
    assert finding["confirmed_nodes"] == 4  # producer injects the count (was absent before)
    assert crit.is_breakthrough(finding) is False  # 4 < 5

    campaign.total_confirmed = 5
    finding = build_breakthrough_finding(campaign, "node-5", result)
    assert finding["confirmed_nodes"] == 5
    assert crit.is_breakthrough(finding) is True  # DEAD before the producer injected it


# ── Bug 2: a FAILED ML baseline must not become a false breakthrough ─────────

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
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, **kwargs) -> None:
        self.events.append(kwargs)


class _FakeAsyncResult:
    """Celery result whose worker produced NO usable metric (the failure case)."""

    def get(self, timeout=None):  # noqa: ARG002
        return {"status": "ok", "metrics": {}}


class _FakeTask:
    def delay(self, payload):  # noqa: ARG002
        return _FakeAsyncResult()


def test_failed_ml_baseline_returns_none_not_zero(monkeypatch) -> None:
    """measure_baseline signals failure with None (not a misleading real 0.0)."""
    from services.orchestrator import campaign_loop as cl

    # fast_tool mode avoids the LLM config round-trip; the worker still yields no metric.
    monkeypatch.setattr(cl.settings, "campaign_baseline_mode", "fast_tool", raising=False)
    monkeypatch.setattr(cl, "run_sub_agent_task", _FakeTask(), raising=True)

    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000b2",
        question="Find the optimal MLP architecture for MNIST classification",
        breakthrough_criteria=BreakthroughCriteria(metric_name="val_accuracy"),
        compute_budget_seconds=60,
    )
    assert cl._is_ml_campaign(campaign) is True

    measured = asyncio.run(
        cl.measure_baseline(campaign, object(), _FakeEmitter(), _fake_session_factory)
    )
    assert measured is None  # was 0.0 before the fix


def test_failed_ml_baseline_refuses_verification_frame() -> None:
    """With baseline_failed set, even a high-confidence/high-confirm finding is NOT a
    breakthrough — the verification frame is reserved for genuine baseline=0 campaigns."""
    failed = BreakthroughCriteria(
        metric_name="val_accuracy",
        baseline_value=0.0,
        min_confidence=0.85,
        min_replications=3,
        min_confirmed_findings=1,
        baseline_failed=True,
    )
    strong = {"confidence": 0.99, "replication_count": 9, "confirmed_nodes": 25, "metric_value": 0.9}
    assert failed.is_breakthrough(strong) is False

    # A genuine verification campaign (baseline legitimately 0, measurement did not fail)
    # must still be able to declare breakthrough — the fix does not break that path.
    genuine = BreakthroughCriteria(
        metric_name="sidon_density",
        baseline_value=0.0,
        min_confidence=0.85,
        min_replications=3,
        baseline_failed=False,
    )
    assert genuine.is_breakthrough({"confidence": 0.99, "replication_count": 5}) is True


def test_baseline_failed_survives_serialization_roundtrip() -> None:
    crit = BreakthroughCriteria(metric_name="val_accuracy", baseline_failed=True)
    assert BreakthroughCriteria.from_dict(crit.to_dict()).baseline_failed is True


# ── Bug 3: lower_is_better 0.0 optimum + falsy-0.0 metric drop ───────────────

def test_lower_is_better_keeps_zero_optimum_and_accepts_zero_metric() -> None:
    crit = BreakthroughCriteria(
        metric_name="val_loss", direction="lower_is_better", improvement_threshold=0.05
    )
    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000b3",
        question="Minimize reconstruction loss",
        breakthrough_criteria=crit,
        compute_budget_seconds=60,
    )

    # A legit 0.0 metric_value must be recorded (before: dropped as falsy → returned False).
    assert campaign.update_best_metric({"metric_value": 0.0}) is True
    assert campaign.best_metric == 0.0
    assert campaign.best_finding is not None

    # A WORSE finding must NOT overwrite the 0.0 optimum (before: ==0.0 sentinel let it in).
    assert campaign.update_best_metric({"metric_value": 0.5}) is False
    assert campaign.best_metric == 0.0

    # A genuinely BETTER (more negative) finding still wins.
    assert campaign.update_best_metric({"metric_value": -0.2}) is True
    assert campaign.best_metric == -0.2


def test_is_breakthrough_does_not_drop_zero_metric_value() -> None:
    """A finding whose metric_value is a real 0.0 is not silently replaced by the
    fallback metric_name key (falsy-or bug)."""
    crit = BreakthroughCriteria(
        metric_name="val_loss",
        baseline_value=1.0,
        direction="lower_is_better",
        improvement_threshold=0.05,
        min_confidence=0.85,
        min_replications=3,
    )
    # metric_value=0.0 (a huge improvement over baseline 1.0); the stale fallback key would
    # have been read instead before the fix. With the fix, 0.0 is used → breakthrough.
    finding = {"confidence": 0.99, "replication_count": 3, "metric_value": 0.0, "val_loss": 1.0}
    assert crit.is_breakthrough(finding) is True


# ── Bug 4: inverted _is_ml_campaign default for plugin-owned campaigns ───────

def test_plugin_with_none_objective_spec_is_non_ml(monkeypatch) -> None:
    """A domain plugin whose objective_spec() is None must STILL be classified NON-ML by
    the inverted _is_ml_campaign (domain-owned ⇒ ML only if the plugin explicitly asserts
    is_ml=True), even when the metric label carries an ML token. Every shipped domain now
    declares an objective_spec, so this uses a synthetic None-objective plugin to exercise
    the base-default path directly."""
    import propab.domain_modules.registry as reg
    from propab.domain_modules.base import DomainPlugin
    from services.orchestrator.campaign_loop import _is_ml_campaign

    class _NoneObjPlugin(DomainPlugin):
        domain_id = "none_obj_test"

        def available_features(self):
            return []

        def objective_spec(self):
            return None

    # _is_ml_campaign resolves the owning plugin from the registry at call time.
    monkeypatch.setattr(reg, "resolve_domain_plugin", lambda **_kw: _NoneObjPlugin())

    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000b4",
        question="A domain-owned question whose default metric carries the accuracy token",
        breakthrough_criteria=BreakthroughCriteria(metric_name="val_accuracy"),
        compute_budget_seconds=60,
    )
    assert _is_ml_campaign(campaign) is False


def test_genuine_plugin_less_ml_question_still_classified_ml() -> None:
    """The keyword heuristic is preserved for genuinely plugin-less ML questions."""
    from services.orchestrator.campaign_loop import _is_ml_campaign
    from propab.domain_modules.registry import resolve_domain_plugin

    question = "Find the optimal MLP architecture for MNIST classification"
    assert resolve_domain_plugin(question=question) is None  # precondition: no plugin owns it

    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000b5",
        question=question,
        breakthrough_criteria=BreakthroughCriteria(metric_name="val_accuracy"),
        compute_budget_seconds=60,
    )
    assert _is_ml_campaign(campaign) is True
