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
        # A counting/exact domain verifies deterministically; stamp the proof
        # shape so the F1 artifact gate (keyed on evidence SHAPE) passes it
        # through as a real deterministic confirm rather than treating it as
        # shapeless "unknown" evidence. Post-V4 the bare ``deterministic`` flag is
        # no longer a standalone gate bypass — it must co-occur with a recognized
        # proof ``verification_method``, exactly as a real exact/combinatorial
        # verifier emits.
        return {
            "metric_value": 0.9,
            "verified_true_steps": 1,
            "deterministic": True,
            "verification_method": "exact_check",
        }

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


class _LofoBadNullPlugin(_CountingPlugin):
    """On-topic plugin that 'confirms' on lofo evidence whose adversarial null FAILS."""

    domain_id = "lofo_bad"

    def run_verification(self, hypothesis, evidence=None, features=None) -> dict[str, Any]:
        self.run_verification_calls += 1
        # lofo shape (lofo_r2 + label_shuffle_null_p95) but the observed effect
        # sits BELOW the label-shuffle null p95 → an artifact, must not confirm.
        return {
            "lofo_r2": 0.05,
            "label_shuffle_null_p95": 0.20,
            "label_shuffle_permutation_p": 0.60,
            "metric_value": 0.05,
            "verified_true_steps": 1,
        }


class _UnknownEvidencePlugin(_CountingPlugin):
    """On-topic plugin that 'confirms' on shapeless (unknown) evidence."""

    domain_id = "unknown_ev"

    def run_verification(self, hypothesis, evidence=None, features=None) -> dict[str, Any]:
        self.run_verification_calls += 1
        # No proof method, no deterministic flag, no statistical null → "unknown".
        return {"note": "confirmed but nothing the honesty gate can verify"}


def test_f1_lofo_confirm_without_passing_null_is_gated():
    # F1: a plugin "confirmed" on lofo evidence whose adversarial null FAILS must be
    # downgraded by the artifact gate — not passed through raw. Verification still
    # runs (the gate acts AFTER classify_verdict), then the verdict is demoted.
    plugin = _LofoBadNullPlugin(on_topic=True)
    result = _run(plugin)
    assert plugin.run_verification_calls == 1
    assert result["verdict"] != "confirmed"


def test_f1_unknown_shaped_confirm_is_gated_to_inconclusive():
    # F1: a "confirmed" on shapeless (unknown) evidence — no proof method, no null —
    # cannot survive the gate and collapses to inconclusive.
    plugin = _UnknownEvidencePlugin(on_topic=True)
    result = _run(plugin)
    assert plugin.run_verification_calls == 1
    assert result["verdict"] == "inconclusive"


# ── Bug-fix regression tests: fail-closed gate, scope coverage, deterministic hole ──

def _run_full(plugin: DomainPlugin, *, hypothesis: dict | None = None, emitter=None) -> dict[str, Any]:
    """Drive the plugin path with a custom hypothesis and/or emitter."""
    return asyncio.run(
        _plugin_verification_path(
            payload={},
            hypothesis=hypothesis or {"text": "some hypothesis", "test_methodology": "m"},
            hypothesis_id="h1",
            campaign_node_id=None,
            session_id="s1",
            question="q",
            session_factory=_fake_session_factory,
            emitter=emitter or _FakeEmitter(),
            registry=None,
            trace_pointer="tp",
            started=0.0,
            baseline={},
            domain_plugin=plugin,
        )
    )


class _RaiseOnGateEmitter:
    """Emitter that fails ONLY on the artifact-gate downgrade notification.

    Reproduces a transient redis/DB failure at exactly the emit the old code fired
    (and swallowed with ``except: pass``) when a confirm was being rejected.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, **kwargs):
        if str(kwargs.get("step", "")).endswith(".artifact_gate"):
            raise RuntimeError("transient redis/db failure during downgrade emit")
        self.events.append(kwargs)


def test_bug1_plugin_gate_fails_closed_when_downgrade_emit_raises():
    # Bug 1 (false-confirm hole): the downgrade emit fires ONLY when a confirm is
    # being REJECTED. If it raises, the old ordering (emit THEN assign, both inside
    # one broad `try/except: pass`) swallowed the error and left the rejected result
    # standing as "confirmed". The fix applies the downgrade BEFORE the emit and
    # wraps the emit separately, so a failing emit can never resurrect the verdict.
    plugin = _UnknownEvidencePlugin(on_topic=True)  # confirm on shapeless evidence -> gated
    result = _run_full(plugin, emitter=_RaiseOnGateEmitter())
    assert plugin.run_verification_calls == 1
    # Before the fix this stayed "confirmed" (the assignment was skipped by the
    # swallowed emit exception); after the fix it is honestly inconclusive.
    assert result["verdict"] != "confirmed"
    assert result["verdict"] == "inconclusive"


class _ScopeInflatedLofoPlugin(_CountingPlugin):
    """On-topic plugin whose lofo confirm SURVIVES the null and the OOD gate but
    claims a broader (cross-class) scope than the experiment actually executed."""

    domain_id = "scope_inflated"

    def run_verification(self, hypothesis, evidence=None, features=None) -> dict[str, Any]:
        self.run_verification_calls += 1
        # lofo shape whose observed effect CLEARS the label-shuffle null p95 with a
        # strict shuffle p -> survives the artifact gate; lofo>0 and label_p<0.05 ->
        # passes the OOD gate. But the output records NO executed OOD split, so the
        # declared cross-class generalization cannot be substantiated.
        return {
            "lofo_r2": 0.6,
            "label_shuffle_null_p95": 0.2,
            "label_shuffle_permutation_p": 0.001,
            "n_families": 5,
            "verified_true_steps": 1,
        }


_SCOPE_INFLATED_HYP = {
    "text": (
        "The kinetic signal generalizes across all enzyme substrate classes.\n"
        "Population: 500 curated enzyme-substrate pairs\n"
        "Distribution: EC class 1 oxidoreductases only\n"
        "Claimed generalization: transfers to EC class 3 hydrolases held out entirely\n"
        "Expected failure modes: should weaken on membrane-bound enzymes\n"
        "OOD test: hold out EC class 3 hydrolases and evaluate cross-class transfer"
    ),
    "test_methodology": "leave-one-family-out cross validation",
}


def test_bug2_scope_inflated_lofo_confirm_is_caught_on_plugin_path():
    # Bug 2 (coverage gap): the plugin path used to apply ONLY the artifact gate,
    # skipping the OOD + scope-integrity stages the generic/mandrake/materials paths
    # run. This lofo confirm survives the adversarial null AND the OOD gate, so the
    # artifact gate alone passes it through as "confirmed". The declared cross-class
    # OOD test was never executed, so the newly-chained scope-integrity stage now
    # catches the inflation on the plugin path.
    plugin = _ScopeInflatedLofoPlugin(on_topic=True)
    result = _run_full(plugin, hypothesis=_SCOPE_INFLATED_HYP)
    assert plugin.run_verification_calls == 1
    assert result["verdict"] == "inconclusive"
    assert "scope" in str(result["learned"]).lower()


class _BareDeterministicPlugin(_CountingPlugin):
    """Confirms on a BARE ``deterministic: True`` flag with no proof method."""

    domain_id = "bare_det"

    def run_verification(self, hypothesis, evidence=None, features=None) -> dict[str, Any]:
        self.run_verification_calls += 1
        # No recognized proof verification_method -> post-V4 this is shapeless
        # "unknown" evidence and must be gated, not granted a free deterministic
        # bypass.
        return {"metric_value": 0.9, "verified_true_steps": 1, "deterministic": True}


class _RealProofPlugin(_CountingPlugin):
    """A genuine combinatorial proof: deterministic flag co-occurs with a real
    proof method, so it must STILL confirm through the gate untouched."""

    domain_id = "real_proof"

    def run_verification(self, hypothesis, evidence=None, features=None) -> dict[str, Any]:
        self.run_verification_calls += 1
        return {
            "metric_value": 0.98,
            "verified_true_steps": 1,
            "deterministic": True,
            "verification_method": "combinatorial_computation",
        }


def test_bug3_bare_deterministic_flag_confirm_is_gated_on_plugin_path():
    # Bug 3 (latent bypass): a bare ``deterministic: True`` without a recognized
    # proof method no longer bypasses the artifact gate — it is routed as "unknown"
    # and collapses to inconclusive.
    plugin = _BareDeterministicPlugin(on_topic=True)
    result = _run_full(plugin)
    assert plugin.run_verification_calls == 1
    assert result["verdict"] == "inconclusive"


def test_bug3_genuine_combinatorial_proof_still_confirms_on_plugin_path():
    # The other half of Bug 3: a real deterministic proof (flag + recognized proof
    # method) must STILL confirm — the fix must not break the deterministic path.
    plugin = _RealProofPlugin(on_topic=True)
    result = _run_full(plugin)
    assert plugin.run_verification_calls == 1
    assert result["verdict"] == "confirmed"
