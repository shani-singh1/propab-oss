"""Quantitative-methods core skills cluster.

Extra coverage for the four DOMAIN-INDEPENDENT methodology skills authored in
``skills/core/`` (statistical-rigor, uncertainty-quantification,
exploratory-data-analysis, experimental-design-and-controls). Run with:

    PYTHONPATH="packages/propab-core;." python -m pytest tests/test_skills_quant.py -q

These tests are additive — they do not touch the framework or ``test_skills.py``.
"""
from __future__ import annotations

from propab.skills import (
    available_skill_index,
    load_skills,
    read_skills,
)

# The four skills this cluster adds, with the phase each is authored for.
_QUANT_SKILLS = {
    "statistical-rigor": "experiment",
    "exploratory-data-analysis": "experiment",
    "experimental-design-and-controls": "experiment",
    "uncertainty-quantification": "evidence",
}

# Same blocklist the framework test enforces; duplicated here so this file fails
# loudly on its own if a quant skill ever leaks domain vocabulary.
_DOMAIN_WORDS = (
    "cap set", "cap-set", "sidon", "enzyme", "kcat", "genom", "tissue", "gtex",
    "graph", "network", "dielectric", "crystal", "hamming", "code word",
)


def test_all_four_quant_skills_are_indexed_as_core():
    core = set(available_skill_index().get("core", []))
    for name in _QUANT_SKILLS:
        assert name in core, f"expected core skill {name!r} to be indexed"


def test_quant_skills_load_for_their_declared_phase():
    for name, phase in _QUANT_SKILLS.items():
        core = {s.name for s in load_skills(None, phase) if s.scope == "core"}
        assert name in core, f"{name!r} should load in the {phase!r} phase"


def test_quant_skills_are_core_scope_with_description():
    seen = {s.name: s for s in load_skills(None, "experiment") + load_skills(None, "evidence")}
    for name in _QUANT_SKILLS:
        s = seen.get(name)
        assert s is not None, f"{name!r} not loaded"
        assert s.scope == "core", f"{name!r} must be core-scoped, got {s.scope!r}"
        assert s.description.strip(), f"{name!r} must have a non-empty description"


def test_quant_skills_carry_no_domain_vocabulary():
    seen = {s.name: s for s in load_skills(None, "experiment") + load_skills(None, "evidence")}
    for name in _QUANT_SKILLS:
        s = seen[name]
        blob = (s.body + " " + s.description).lower()
        for w in _DOMAIN_WORDS:
            assert w not in blob, f"quant skill {name!r} leaks domain vocabulary: {w!r}"


def test_quant_skills_encode_their_load_bearing_methodology():
    """Each skill must actually carry the specific methodology it promises, not just a title."""
    seen = {s.name: s.body.lower() for s in load_skills(None, "experiment") + load_skills(None, "evidence")}

    sr = seen["statistical-rigor"]
    # decision-tree tests, assumption checks, multiplicity correction, effect size + CI, power.
    assert "decision tree" in sr
    assert "assumption" in sr and "normal" in sr and "independence" in sr
    assert "holm" in sr and "benjamini" in sr
    assert "effect size" in sr and "confidence interval" in sr
    assert "power" in sr and "p-hacking" in sr
    # Rejects the bare-p and observed-power anti-patterns.
    assert "observed power" in sr

    uq = seen["uncertainty-quantification"]
    assert "aleatoric" in uq and "epistemic" in uq
    assert "bootstrap" in uq and "propagat" in uq
    assert "interval" in uq and "overlap" in uq.replace("overlaps", "overlap")
    assert "point estimate" in uq

    eda = seen["exploratory-data-analysis"]
    assert "distribution" in eda and "missing" in eda and "outlier" in eda
    assert "balance" in eda
    assert "leak" in eda  # the load-bearing addition over the weak file-format version
    assert "before" in eda  # before any modeling

    ed = seen["experimental-design-and-controls"]
    assert "positive control" in ed and "negative control" in ed
    assert "randomiz" in ed and "block" in ed
    assert "confound" in ed and "factorial" in ed
    assert "pre-regist" in ed  # pre-registered decision rule


def test_quant_skills_are_readable_on_demand_and_rendered():
    block = read_skills(list(_QUANT_SKILLS))
    assert "RESEARCH METHODOLOGY" in block
    for name in _QUANT_SKILLS:
        assert name in block, f"{name!r} should be present in the on-demand read block"
