"""Tests for the DomainPlugin interface and registry (fixes.md Checklist 2)."""
from __future__ import annotations

import pytest

from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.registry import (
    all_plugins,
    get_domain_plugin,
    register_plugin,
    resolve_domain_plugin,
)


def test_builtin_plugins_registered():
    ids = {p.domain_id for p in all_plugins()}
    assert {"materials", "mandrake", "enzyme_kinetics", "graph_invariants"} <= ids


def test_uses_synthetic_data_flag_per_domain():
    """DOM2: synthetic-data domains report True; real-data domains report False.

    The genomics / graph_invariants / enzyme_kinetics demo domains run on
    seed-generated frames (adapter meta ``synthetic: True``) presented under real
    dataset names; materials and mandrake use real data.
    """
    for did in ("genomics", "graph_invariants", "enzyme_kinetics"):
        plugin = get_domain_plugin(did)
        assert plugin is not None, f"{did} plugin not registered"
        assert plugin.uses_synthetic_data() is True, f"{did} should report synthetic data"
    for did in ("mandrake", "materials"):
        plugin = get_domain_plugin(did)
        assert plugin is not None, f"{did} plugin not registered"
        assert plugin.uses_synthetic_data() is False, f"{did} should report real data"
    # The base-class default is honest-by-omission: False.
    class _Bare(DomainPlugin):
        domain_id = "bare"

        def available_features(self):
            return []

    assert _Bare().uses_synthetic_data() is False


def test_get_domain_plugin_by_id():
    assert get_domain_plugin("materials").domain_id == "materials"
    assert get_domain_plugin("MANDRAKE").domain_id == "mandrake"
    assert get_domain_plugin("does_not_exist") is None
    assert get_domain_plugin(None) is None


def test_resolve_explicit_payload_domain():
    assert resolve_domain_plugin(payload={"domain": "materials"}).domain_id == "materials"
    assert resolve_domain_plugin(payload={"domain": "mandrake"}).domain_id == "mandrake"
    assert resolve_domain_plugin(payload={"domain_profile": "enzyme_kinetics"}).domain_id == "enzyme_kinetics"


def test_resolve_tag_in_question():
    assert resolve_domain_plugin(question="x [domain_profile:materials] y").domain_id == "materials"
    assert resolve_domain_plugin(question="[domain_profile:graph_invariants]").domain_id == "graph_invariants"


def test_resolve_mandrake_heuristic_requires_two_markers():
    # Two markers → mandrake
    q2 = "reverse transcriptase RT activity across an evolutionary family"
    assert resolve_domain_plugin(question=q2).domain_id == "mandrake"
    # One marker → not enough
    q1 = "reverse transcriptase structure only"
    assert resolve_domain_plugin(question=q1) is None


def test_resolve_none_for_generic_ml():
    assert resolve_domain_plugin(question="train a transformer on mnist and report accuracy") is None


def test_resolve_explicit_payload_beats_question():
    # Explicit payload domain wins over an unrelated question.
    p = resolve_domain_plugin(question="clustering coefficient graph invariant", payload={"domain": "materials"})
    assert p.domain_id == "materials"


def test_materials_criteria_and_features_from_profile():
    mp = get_domain_plugin("materials")
    crit = mp.confirmation_criteria()
    assert crit["requires_holdout"] is True
    assert crit["holdout_type"] == "LOFO"
    assert crit["min_groups"] == 5
    assert len(mp.available_features()) >= 10
    assert mp.domain_profile().profile_id == "materials"


def test_mandrake_criteria_default_when_no_profile():
    crit = get_domain_plugin("mandrake").confirmation_criteria()
    assert crit["requires_holdout"] is False
    assert crit["min_metric_steps_for_confirm"] == 2
    assert get_domain_plugin("mandrake").domain_profile() is None


def test_default_plugin_behaviour():
    class _Dummy(DomainPlugin):
        domain_id = "dummy_test_domain"
        display_name = "Dummy"

        def available_features(self):
            return ["x"]

    d = _Dummy()
    with pytest.raises(NotImplementedError):
        d.run_verification({"text": "h"})
    assert d.artifact_models(hypothesis={"text": "h"}) == []
    # D1 fix: a plugin with no verification capability (no run_verification
    # override, no domain profile) now FAILS the default preflight fail-closed.
    # The old assertion (`passed is True`) encoded the fail-open defect being fixed.
    pf = d.preflight()
    assert pf.passed is False
    assert "no verification capability" in pf.reason
    assert d.literature_prior("q") == {}
    # Default classify_verdict is neutral (domains override with their own classifier).
    verdict, rationale, conf = d.classify_verdict("h", {"mean_r2": 0.0})
    assert verdict == "inconclusive"
    assert 0.0 <= conf <= 1.0


def test_register_plugin_override():
    class _DummyOverride(DomainPlugin):
        domain_id = "materials"
        display_name = "Override"

        def available_features(self):
            return ["overridden"]

    original = get_domain_plugin("materials")
    try:
        register_plugin(_DummyOverride())
        assert get_domain_plugin("materials").available_features() == ["overridden"]
    finally:
        register_plugin(original)  # restore for other tests
    assert get_domain_plugin("materials").domain_id == "materials"
    assert len(get_domain_plugin("materials").available_features()) >= 10


def test_preflight_result_to_dict():
    r = PreflightResult(False, "nope", {"n": 1})
    assert r.to_dict() == {"passed": False, "reason": "nope", "details": {"n": 1}}


# ── D1: default preflight is fail-CLOSED without a verification capability ────

def test_default_preflight_fails_closed_without_verification():
    """A bare plugin (no run_verification override, no profile) fails preflight."""

    class _NoVerify(DomainPlugin):
        domain_id = "no_verify_domain"

        def available_features(self):
            return []

    p = _NoVerify()
    assert p.has_verification_capability() is False
    result = p.preflight()
    assert result.passed is False
    assert "no verification capability" in result.reason
    assert result.details["has_verification_capability"] is False


def test_default_preflight_passes_when_run_verification_overridden():
    """Overriding run_verification is a runnable verification path → preflight passes."""

    class _WithVerify(DomainPlugin):
        domain_id = "with_verify_domain"

        def available_features(self):
            return ["x"]

        def run_verification(self, hypothesis, evidence=None, features=None):
            return {"verification_method": "dummy", "lofo_r2": 0.1}

    p = _WithVerify()
    assert p.has_verification_capability() is True
    result = p.preflight()
    assert result.passed is True
    assert "verification path present" in result.reason


def test_default_preflight_passes_with_domain_profile_only():
    """A profile-backed plugin (generic verification path) also passes fail-closed default."""

    class _FakeProfile:
        profile_id = "fake"

    class _WithProfile(DomainPlugin):
        domain_id = "with_profile_domain"

        def available_features(self):
            return ["x"]

        def domain_profile(self):
            return _FakeProfile()

    p = _WithProfile()
    assert p.has_verification_capability() is True
    assert p.preflight().passed is True


def test_builtin_plugins_not_spuriously_failed_by_preflight():
    """
    Each real built-in plugin must NOT be spuriously failed by the fail-closed
    default. The 6 contract plugins either override preflight with a real power
    check OR expose a verification capability, so the default (when reached) still
    passes. We assert either an explicit passed=True or a real, non-default reason
    (a dataset-driven failure is a legitimate check, not the fail-closed default).
    """
    from propab.domain_modules.materials.plugin import MaterialsPlugin
    from propab.domain_modules.mandrake.plugin import MandrakePlugin
    from propab.domain_modules.enzyme_kinetics.plugin import EnzymeKineticsPlugin
    from propab.domain_modules.graph_invariants.plugin import GraphInvariantsPlugin
    from propab.domain_modules.network_diffusion.plugin import NetworkDiffusionPlugin
    from propab.domain_modules.math_combinatorics.plugin import MathCombinatoricsPlugin

    _DEFAULT_FAIL = "no verification capability"

    # network_diffusion is scope-only: no run_verification override, no profile,
    # and it does NOT override preflight — so it falls through to the base default.
    # It also never owns campaign routing (matches() == False), so core never
    # calls its preflight for launch. Assert the change does not break it in a way
    # that would matter: it must not be resolvable heuristically.
    assert NetworkDiffusionPlugin().matches(question="a contagion diffusion study") is False

    for plugin_cls in (
        MaterialsPlugin,
        MandrakePlugin,
        EnzymeKineticsPlugin,
        GraphInvariantsPlugin,
        MathCombinatoricsPlugin,
    ):
        plugin = plugin_cls()
        # All 5 override preflight with their own dataset/power check.
        assert type(plugin).preflight is not DomainPlugin.preflight, (
            f"{plugin.domain_id} unexpectedly uses the default preflight"
        )
        result = plugin.preflight()
        # A real power check may legitimately fail on a machine without the
        # dataset, but it must never fail with the fail-closed DEFAULT reason —
        # that would mean our change leaked into a plugin that has a real check.
        assert _DEFAULT_FAIL not in result.reason, (
            f"{plugin.domain_id} spuriously hit the fail-closed default: {result.reason}"
        )


def test_has_verification_capability_matches_builtin_owners():
    """The 5 verifying built-ins report a verification capability; scope-only does not."""
    from propab.domain_modules.materials.plugin import MaterialsPlugin
    from propab.domain_modules.enzyme_kinetics.plugin import EnzymeKineticsPlugin
    from propab.domain_modules.graph_invariants.plugin import GraphInvariantsPlugin
    from propab.domain_modules.math_combinatorics.plugin import MathCombinatoricsPlugin
    from propab.domain_modules.mandrake.plugin import MandrakePlugin
    from propab.domain_modules.network_diffusion.plugin import NetworkDiffusionPlugin

    for plugin_cls in (
        MaterialsPlugin,
        MandrakePlugin,
        EnzymeKineticsPlugin,
        GraphInvariantsPlugin,
        MathCombinatoricsPlugin,
    ):
        assert plugin_cls().has_verification_capability() is True

    # Scope-only contributor: no verification path (but never routed to as owner).
    assert NetworkDiffusionPlugin().has_verification_capability() is False


# ── D3: empty/None defaults are observable, not silently "passed" ────────────

def test_d3_scope_and_artifact_queries_are_observable():
    class _Bare(DomainPlugin):
        domain_id = "bare_d3_domain"

        def available_features(self):
            return []

    p = _Bare()
    assert p.has_scope_template() is False
    assert p.scope_template() is None
    assert p.has_artifact_models() is False
    assert p.artifact_models(hypothesis={"text": "h"}) == []

    from propab.domain_modules.network_diffusion.plugin import NetworkDiffusionPlugin

    nd = NetworkDiffusionPlugin()
    # network_diffusion supplies a scope template but no artifact models (no profile).
    assert nd.has_scope_template() is True
    assert nd.has_artifact_models() is False
