"""Extra guardrails for the DOMAIN-INDEPENDENT (core) research skills.

The framework test (tests/test_skills.py) checks the loader and scans the
hypothesis + experiment phases for domain vocabulary. Core skills also cover the
evidence and iteration phases, so this file extends the same invariants to EVERY
phase and asserts the core catalogue is well-formed. It only reads the public
loader API — it never touches core code or domain directories.
"""
from __future__ import annotations

from propab.skills import PHASES, Skill, load_skills

# Same list the framework test enforces, plus a few adjacent domain terms so a
# core skill cannot smuggle domain flavour past the two-phase framework scan.
_DOMAIN_WORDS = (
    "cap set", "cap-set", "sidon", "enzyme", "kcat", "genom", "tissue", "gtex",
    "graph", "network", "dielectric", "crystal", "hamming", "code word",
    "protein", "molecule", "qubit", "lattice", "catalys",
)

# Phases each authored core skill is expected to serve (an "any" skill would also
# be fine, but the current catalogue assigns concrete phases).
_EXPECTED_CORE = {
    "falsifiable-hypothesis-design": "hypothesis",
    "scope-and-generalization-discipline": "hypothesis",
    "adversarial-test-design": "experiment",
    "confound-and-leakage-control": "experiment",
    "evidence-honesty-and-calibration": "evidence",
    "anti-rediscovery-and-novelty-check": "evidence",
    "failure-analysis-and-inconclusive-reporting": "evidence",
    "iterative-refinement-and-cross-attempt-learning": "iteration",
}


def _all_core_skills() -> dict[str, Skill]:
    seen: dict[str, Skill] = {}
    for phase in PHASES:
        for s in load_skills(None, phase):
            if s.scope == "core":
                seen[s.name] = s
    return seen


def test_core_skills_carry_no_domain_vocabulary_in_every_phase():
    for s in _all_core_skills().values():
        blob = (s.body + " " + s.description).lower()
        for w in _DOMAIN_WORDS:
            assert w not in blob, (
                f"core skill {s.name!r} leaks domain vocabulary {w!r}"
            )


def test_expected_core_skills_present_with_right_phase():
    skills = _all_core_skills()
    for name, phase in _EXPECTED_CORE.items():
        assert name in skills, f"missing expected core skill {name!r}"
        assert skills[name].phase == phase, (
            f"core skill {name!r} has phase {skills[name].phase!r}, expected {phase!r}"
        )


def test_every_core_skill_is_well_formed():
    for s in _all_core_skills().values():
        assert s.description.strip(), f"core skill {s.name!r} has no description"
        assert len(s.body.strip()) > 200, f"core skill {s.name!r} body is too thin"
        assert s.phase in ("any",) + PHASES, (
            f"core skill {s.name!r} has invalid phase {s.phase!r}"
        )


def test_all_four_phases_have_core_coverage():
    for phase in PHASES:
        core = [s for s in load_skills(None, phase) if s.scope == "core"]
        assert core, f"no core skill covers the {phase!r} phase"
