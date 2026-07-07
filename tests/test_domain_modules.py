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

    All domains now run on real data or real computation: graph_invariants -> real
    SNAP networks (ca-GrQc, email-Eu-core, migrated by D); genomics (GTEx v8 median
    TPM) and enzyme_kinetics (DLKcat / BRENDA + SABIO-RK) -> real public data. No
    domain is unconditionally synthetic anymore. The genomics/enzyme flag reads the
    cache meta, so it only reports True on the network-unavailable synthetic
    fallback; we assert not-True-under-real-cache without hard-failing an offline run.
    """
    for did in ("mandrake", "materials", "graph_invariants"):
        plugin = get_domain_plugin(did)
        assert plugin is not None, f"{did} plugin not registered"
        assert plugin.uses_synthetic_data() is False, f"{did} should report real data"
    # Real-data domains: assert False when real data was served; skip only if the
    # network-unavailable synthetic fallback was written (so offline CI is stable).
    from propab.domain_modules.enzyme_kinetics.adapter import (
        dataset_is_synthetic as enzyme_synth,
    )
    from propab.domain_modules.genomics.adapter import (
        dataset_is_synthetic as genomics_synth,
    )
    for did, is_synth in (("enzyme_kinetics", enzyme_synth), ("genomics", genomics_synth)):
        plugin = get_domain_plugin(did)
        assert plugin is not None, f"{did} plugin not registered"
        flag = plugin.uses_synthetic_data()  # ensures cache, then reads meta
        if is_synth():
            # Only the offline fallback should ever report synthetic.
            assert flag is True, f"{did} fallback flag must match cache meta"
        else:
            assert flag is False, f"{did} should report real data when cached"
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

    # network_diffusion is now a real verifying domain (SIS/SIR simulator + a real
    # cross-topology holdout null), so it self-routes on contagion questions and
    # joins the other built-ins below.
    assert NetworkDiffusionPlugin().matches(question="a contagion diffusion study") is True

    for plugin_cls in (
        MaterialsPlugin,
        MandrakePlugin,
        EnzymeKineticsPlugin,
        GraphInvariantsPlugin,
        MathCombinatoricsPlugin,
        NetworkDiffusionPlugin,
    ):
        plugin = plugin_cls()
        # All 6 override preflight with their own dataset/power check.
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
        NetworkDiffusionPlugin,
    ):
        assert plugin_cls().has_verification_capability() is True


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
    # network_diffusion now supplies BOTH a scope template and artifact models — it
    # gained a real domain profile when it became a verifying domain.
    assert nd.has_scope_template() is True
    assert nd.has_artifact_models() is True


# ── DOM3: keyword collisions route by MATCH SCORE, not registration order ─────

def test_dom3_collision_routes_to_higher_scoring_plugin():
    """
    A question that matches two domains routes to the HIGHER-scoring one, not the
    one registered first. graph_invariants is registered BEFORE math_combinatorics,
    so under the old first-match logic the graph plugin always won a collision. Here
    the question is combinatorics-dominant (5 math markers vs 2 graph markers), so
    the later-registered plugin must win.
    """
    q = (
        "sidon cap set sumset ap-free additive combinatorics "
        "with a spectral gap on a network family"
    )
    gi = get_domain_plugin("graph_invariants")
    mc = get_domain_plugin("math_combinatorics")
    # Both gate True — a genuine collision.
    assert gi.matches(question=q) is True
    assert mc.matches(question=q) is True
    # math_combinatorics scores strictly higher.
    assert mc.match_score(question=q) > gi.match_score(question=q)
    # Routing follows the score, not registration order.
    assert resolve_domain_plugin(question=q).domain_id == "math_combinatorics"


def test_dom3_registration_first_still_wins_when_it_scores_higher():
    """The graph plugin should still win when the question is graph-dominant."""
    q = "spectral gap clustering coefficient network family with a lone sidon set"
    gi = get_domain_plugin("graph_invariants")
    mc = get_domain_plugin("math_combinatorics")
    assert gi.matches(question=q) is True
    assert mc.matches(question=q) is True
    assert gi.match_score(question=q) > mc.match_score(question=q)
    assert resolve_domain_plugin(question=q).domain_id == "graph_invariants"


def test_dom3_single_domain_match_is_unchanged():
    """When exactly one plugin matches, routing is identical to before."""
    # Pure enzyme question (>=2 enzyme markers, no other domain vocabulary).
    q = "does kcat rise with enzyme catalytic turnover across brenda ec class families"
    matching = [p for p in all_plugins() if p.matches(question=q)]
    assert [p.domain_id for p in matching] == ["enzyme_kinetics"]
    assert resolve_domain_plugin(question=q).domain_id == "enzyme_kinetics"


def test_dom3_explicit_tag_fast_path_still_wins_over_score():
    """The [domain_profile:X] fast path must beat any heuristic score."""
    # Question body is graph-heavy, but the explicit tag names math_combinatorics.
    q = "[domain_profile:math_combinatorics] spectral gap clustering coefficient network family"
    assert resolve_domain_plugin(question=q).domain_id == "math_combinatorics"
    # Explicit payload likewise wins.
    p = resolve_domain_plugin(question=q, payload={"domain": "graph_invariants"})
    assert p.domain_id == "graph_invariants"


def test_dom3_near_tie_emits_routing_ambiguity_log(caplog):
    """A near-tie (top two scores within the margin) is surfaced as a warning."""
    import logging

    q = "sidon cap set with spectral gap clustering coefficient network family"
    with caplog.at_level(logging.WARNING, logger="propab.domain_modules.registry"):
        resolved = resolve_domain_plugin(question=q)
    # Higher score still wins deterministically (graph: 3 vs math: 2).
    assert resolved.domain_id == "graph_invariants"
    # ...but the near-tie is visible, not silently resolved.
    assert any("domain routing ambiguity" in r.message for r in caplog.records)


def test_dom3_no_ambiguity_log_on_clear_winner(caplog):
    """A clear winner (score gap beyond the margin) does NOT log an ambiguity."""
    import logging

    q = (
        "sidon cap set sumset ap-free additive combinatorics "
        "with a spectral gap on a network family"
    )
    with caplog.at_level(logging.WARNING, logger="propab.domain_modules.registry"):
        resolve_domain_plugin(question=q)
    assert not any("domain routing ambiguity" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# literature_profile completeness — every science domain must supply real
# tabulation + open-problem grounding so novelty_check can reject rediscovery
# and the /prior gap path has a real frontier to target. (Empty sources are
# what made novelty_check return "uncertain" for every non-combinatorics
# domain and left open_gaps unfillable.)
# --------------------------------------------------------------------------

_GROUNDED_DOMAINS = [
    "materials",
    "enzyme_kinetics",
    "genomics",
    "mandrake",
    "math_combinatorics",
]

# Substrings that betray a placeholder rather than a real, citeable source.
_PLACEHOLDER_MARKERS = ("todo", "tbd", "placeholder", "fixme", "xxx", "example.com")


@pytest.mark.parametrize("domain_id", _GROUNDED_DOMAINS)
def test_literature_profile_has_real_tabulation_and_open_problem_sources(domain_id):
    plugin = get_domain_plugin(domain_id)
    assert plugin is not None, f"{domain_id} plugin not registered"
    profile = plugin.literature_profile()

    tab = profile.get("tabulation_sources") or []
    ops = profile.get("open_problem_sources") or []
    crit = profile.get("novelty_criteria") or ""

    assert tab, f"{domain_id}: tabulation_sources must be non-empty (rediscovery rejection)"
    assert ops, f"{domain_id}: open_problem_sources must be non-empty (frontier gaps)"
    assert isinstance(crit, str) and len(crit) > 40, f"{domain_id}: novelty_criteria too thin"

    # Each tabulation source carries a name and at least one concrete
    # identifier/anchor (not an empty stub like the old {"name": ..., "identifiers": []}).
    for t in tab:
        assert t.get("name"), f"{domain_id}: tabulation source missing name"
        has_ids = bool(t.get("identifiers"))
        # OEIS-style id lists, or numeric best-known anchors, both count as real content.
        has_anchor = any(
            k not in ("name",) and v not in (None, "", [], {})
            for k, v in t.items()
        )
        assert has_ids or has_anchor, f"{domain_id}: tabulation source {t.get('name')!r} has no identifiers/anchors"

    # No placeholder text anywhere in the grounding fields.
    blob = repr(tab).lower() + repr(ops).lower() + crit.lower()
    for marker in _PLACEHOLDER_MARKERS:
        assert marker not in blob, f"{domain_id}: placeholder {marker!r} found in literature_profile grounding"


@pytest.mark.parametrize("domain_id", ["materials", "enzyme_kinetics", "genomics", "mandrake"])
def test_open_problem_sources_have_urls(domain_id):
    # gap_mapper / prior overlay scrapes open_problem_sources[].url — each entry
    # must carry a usable http(s) URL for that path to yield anything.
    plugin = get_domain_plugin(domain_id)
    ops = plugin.literature_profile().get("open_problem_sources") or []
    assert ops
    assert all(str(o.get("url", "")).startswith("http") for o in ops), (
        f"{domain_id}: every open_problem_sources entry needs an http(s) url"
    )
