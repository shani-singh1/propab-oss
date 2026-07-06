"""DOM4: the plugin verification path must recheck on-topic before verifying.

``_plugin_verification_path`` used to call ``domain_plugin.run_verification``
directly, so an off-topic / misrouted hypothesis was verified against a domain's
fixed default feature pair and could produce a "confirmed" verdict decoupled from
the actual claim. These tests drive the path with fakes (no DB, no event bus) and
assert:

- off-topic hypothesis  → short-circuits to ``inconclusive`` with reason
  ``hypothesis_off_topic_for_domain`` and NEVER calls ``run_verification``;
- on-topic hypothesis   → proceeds to ``run_verification`` as before;
- a plugin whose ``hypothesis_on_topic`` is missing/broken is unaffected
  (fails open — still verified).
"""
from __future__ import annotations

import asyncio
from typing import Any

from propab.domain_modules.base import DomainPlugin
from services.worker.sub_agent_loop import _plugin_verification_path


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

    async def emit(self, **kwargs):
        self.events.append(kwargs)


class _CountingPlugin(DomainPlugin):
    """Plugin that records whether run_verification ran and honors an on-topic set."""

    domain_id = "counting"
    display_name = "counting"

    def __init__(self, *, on_topic: bool = True) -> None:
        self._on_topic = on_topic
        self.run_verification_calls = 0

    def matches(self, *, question: str = "", payload=None) -> bool:
        return False

    def available_features(self) -> list[str]:
        return ["f1", "f2"]

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        return self._on_topic

    def run_verification(self, hypothesis, evidence=None, features=None) -> dict[str, Any]:
        self.run_verification_calls += 1
        return {"metric_value": 0.9, "verified_true_steps": 1}

    def classify_verdict(self, hypothesis_text, result):
        return "confirmed", "counting verified", 0.9

    def confirmation_criteria(self) -> dict[str, Any]:
        return {"min_metric_steps_for_confirm": 1}

    def uses_synthetic_data(self) -> bool:
        return False


class _NoOnTopicPlugin(_CountingPlugin):
    """Plugin missing hypothesis_on_topic entirely (guard must fail open)."""

    domain_id = "noontopic"

    # Shadow the base method with a non-callable so getattr returns non-callable.
    hypothesis_on_topic = None  # type: ignore[assignment]


def _run(plugin: DomainPlugin) -> dict[str, Any]:
    return asyncio.run(
        _plugin_verification_path(
            payload={},
            hypothesis={"text": "some hypothesis", "test_methodology": "m"},
            hypothesis_id="h1",
            campaign_node_id=None,
            session_id="s1",
            question="q",
            session_factory=_fake_session_factory,
            emitter=_FakeEmitter(),
            registry=None,
            trace_pointer="tp",
            started=0.0,
            baseline={},
            domain_plugin=plugin,
        )
    )


def test_off_topic_short_circuits_inconclusive():
    plugin = _CountingPlugin(on_topic=False)
    result = _run(plugin)
    assert result["verdict"] == "inconclusive"
    assert result["confidence"] == 0.0
    assert result["failure_reason"] == "hypothesis_off_topic_for_domain"
    # The domain verification must NOT have run for an off-topic hypothesis.
    assert plugin.run_verification_calls == 0


def test_on_topic_still_verifies():
    plugin = _CountingPlugin(on_topic=True)
    result = _run(plugin)
    assert result["verdict"] == "confirmed"
    assert plugin.run_verification_calls == 1


def test_missing_on_topic_method_fails_open():
    plugin = _NoOnTopicPlugin(on_topic=True)
    result = _run(plugin)
    # No usable on-topic gate → verify exactly as before (never a false refusal).
    assert result["verdict"] == "confirmed"
    assert plugin.run_verification_calls == 1
