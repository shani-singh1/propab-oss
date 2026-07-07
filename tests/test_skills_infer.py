"""Inference-and-integrity CORE skill cluster.

These four skills extend the core methodology into causal identification,
measurement/operationalization, self-audit reproducibility, and rigorous
literature retrieval. They MUST stay domain-independent (no domain vocabulary),
load through the domain-general loader like every other core skill, and slot into
the intended phase + priority ordering. This file only ADDS coverage; it does not
touch the framework loader test in ``tests/test_skills.py``.
"""
from __future__ import annotations

from propab.skills import (
    available_skill_index,
    load_skills,
    read_skills,
    skills_catalog,
)

# The four skills authored for the inference-and-integrity cluster.
_INFER_SKILLS = {
    "causal-inference-and-identification": "evidence",
    "measurement-and-operationalization": "hypothesis",
    "reproducibility-and-integrity-audit": "evidence",
    "literature-lookup-and-provenance": "hypothesis",
}

# Same banned domain vocabulary the framework test enforces, re-checked here so a
# regression in one of THESE files fails loudly in this file too.
_DOMAIN_WORDS = (
    "cap set", "cap-set", "sidon", "enzyme", "kcat", "genom", "tissue", "gtex",
    "graph", "network", "dielectric", "crystal", "hamming", "code word",
)


def test_all_four_infer_skills_are_indexed_as_core():
    core = set(available_skill_index().get("core", []))
    for name in _INFER_SKILLS:
        assert name in core, f"{name!r} missing from the core skill index"


def test_infer_skills_declare_expected_phase_and_scope():
    by_name = {s.name: s for s in skills_catalog()}
    for name, phase in _INFER_SKILLS.items():
        s = by_name[name]
        assert s.scope == "core", f"{name!r} must be a core skill, got scope {s.scope!r}"
        assert s.phase == phase, f"{name!r} expected phase {phase!r}, got {s.phase!r}"
        assert s.description, f"{name!r} must carry a non-empty description"


def test_infer_skills_load_for_their_phase():
    for name, phase in _INFER_SKILLS.items():
        names = {s.name for s in load_skills(None, phase) if s.scope == "core"}
        assert name in names, f"{name!r} did not load for phase {phase!r}"


def test_infer_skills_carry_no_domain_vocabulary():
    for phase in ("hypothesis", "experiment", "evidence", "iteration"):
        for s in load_skills(None, phase):
            if s.scope != "core" or s.name not in _INFER_SKILLS:
                continue
            blob = (s.body + " " + s.description).lower()
            for w in _DOMAIN_WORDS:
                assert w not in blob, f"core skill {s.name!r} leaks domain vocabulary: {w!r}"


def test_infer_skill_bodies_are_readable_on_demand():
    block = read_skills(list(_INFER_SKILLS))
    assert "RESEARCH METHODOLOGY" in block
    for name in _INFER_SKILLS:
        assert name in block, f"{name!r} body not returned by read_skills"
    # Load-bearing content of each skill is actually present in the body.
    assert "identification" in block.lower()          # causal
    assert "proxy" in block.lower()                    # measurement
    assert "exploratory" in block.lower() and "confirmatory" in block.lower()  # audit
    assert "provenance" in block.lower()               # literature


def test_evidence_phase_orders_infer_skills_after_earlier_core():
    # Priorities were chosen so these slot into the intended place: the two
    # evidence-phase skills (28, 38) come after critical-evaluation (22).
    ev = [s.name for s in load_skills(None, "evidence") if s.scope == "core"]
    assert "critical-evaluation-of-claims-and-results" in ev
    assert "causal-inference-and-identification" in ev
    assert ev.index("critical-evaluation-of-claims-and-results") < ev.index(
        "causal-inference-and-identification"
    )
    assert ev.index("causal-inference-and-identification") < ev.index(
        "reproducibility-and-integrity-audit"
    )
