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
    assert d.preflight().passed is True
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
