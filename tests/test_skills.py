"""Propab research-skills framework: loader is domain-general; content is split
core (domain-independent) vs domains/<id> (domain-dependent), and gets injected
into the hypothesis-generation prompt."""
from __future__ import annotations

from propab.skills import (
    Skill,
    available_skill_index,
    load_skills,
    render_skills_block,
    skills_prompt_block,
)

# Domain vocabulary that must NEVER appear in a CORE (domain-independent) skill.
_DOMAIN_WORDS = (
    "cap set", "cap-set", "sidon", "enzyme", "kcat", "genom", "tissue", "gtex",
    "graph", "network", "dielectric", "crystal", "hamming", "code word",
)


def test_core_skills_load_for_hypothesis_phase():
    core = [s for s in load_skills(None, "hypothesis") if s.scope == "core"]
    assert core, "expected at least one core hypothesis skill"
    assert all(isinstance(s, Skill) for s in core)
    names = {s.name for s in core}
    assert "falsifiable-hypothesis-design" in names


def test_domain_skills_are_added_for_that_domain_only():
    generic = load_skills(None, "hypothesis")
    mathy = load_skills("math_combinatorics", "hypothesis")
    assert len(mathy) > len(generic)
    assert any(s.scope == "math_combinatorics" for s in mathy)
    # A domain's skills must not leak into a different domain's load.
    other = load_skills("genomics", "hypothesis")
    assert not any(s.scope == "math_combinatorics" for s in other)


def test_core_skills_carry_no_domain_vocabulary():
    for s in load_skills(None, "hypothesis") + load_skills(None, "experiment"):
        if s.scope != "core":
            continue
        blob = (s.body + " " + s.description).lower()
        for w in _DOMAIN_WORDS:
            assert w not in blob, f"core skill {s.name!r} leaks domain vocabulary: {w!r}"


def test_ordering_puts_core_before_domain():
    skills = load_skills("math_combinatorics", "hypothesis")
    scopes = [s.scope for s in skills]
    # first domain skill (if any) comes after the last core skill of equal-or-lower prio
    if "math_combinatorics" in scopes and "core" in scopes:
        assert scopes.index("core") < scopes.index("math_combinatorics")


def test_render_and_prompt_block_nonempty():
    block = skills_prompt_block("math_combinatorics", "hypothesis")
    assert "RESEARCH METHODOLOGY" in block
    assert "falsifiable-hypothesis-design" in block
    assert render_skills_block([]) == ""


def test_prompt_injects_only_agent_selected_skills():
    # Generation no longer auto-loads by domain: a skills_block the AGENT selected is
    # injected; without one, no methodology is auto-injected (agentic on-demand model).
    from services.orchestrator.hypotheses import _build_hypothesis_prompt
    from services.orchestrator.intake import ParsedQuestion
    from services.orchestrator.schemas import Prior
    from propab.skills import read_skills

    parsed = ParsedQuestion(text="improve the cap-set growth exponent",
                            domain="math_combinatorics", sub_questions=[])
    prior = Prior(established_facts=[], contested_claims=[], open_gaps=[], dead_ends=[], key_papers=[])

    block = read_skills(["falsifiable-hypothesis-design"])
    prompt = _build_hypothesis_prompt(parsed, prior, 6, skills_block=block)
    assert "RESEARCH METHODOLOGY" in prompt
    assert "falsifiable-hypothesis-design" in prompt

    # No agent selection -> no auto-injected methodology (not hardcoded by domain).
    prompt_none = _build_hypothesis_prompt(parsed, prior, 6)
    assert "RESEARCH METHODOLOGY" not in prompt_none


# ---- Agentic catalog + on-demand read -------------------------------------------

def test_catalog_lists_names_and_descriptions_not_bodies():
    from propab.skills import render_catalog

    cat = render_catalog()
    assert "AVAILABLE RESEARCH SKILLS" in cat
    assert "falsifiable-hypothesis-design" in cat          # names present
    assert "cap-set-and-extremal-constructions" in cat     # domain names present too
    # The catalog is the AWARENESS layer — compact, NOT the full bodies (no bloat).
    assert "Anchor on an open gap" not in cat              # body text of a core skill
    assert "Product / tensor constructions" not in cat     # body text of a domain skill


def test_read_skills_returns_bodies_on_demand():
    from propab.skills import read_skills

    block = read_skills(["falsifiable-hypothesis-design", "cap-set-and-extremal-constructions"])
    assert "RESEARCH METHODOLOGY" in block
    assert "falsifiable-hypothesis-design" in block and "cap-set-and-extremal-constructions" in block
    # The full body IS present when read on demand.
    assert "Anchor on an open gap" in block


def test_read_skills_ignores_unknown_and_empty():
    from propab.skills import read_skills

    assert read_skills(["does-not-exist-skill"]) == ""
    assert read_skills([]) == ""
    # a mix returns only the valid one's body
    block = read_skills(["falsifiable-hypothesis-design", "nonsense"])
    assert "falsifiable-hypothesis-design" in block


def test_catalog_covers_core_and_all_domains():
    from propab.skills import catalog_skill_names, skills_catalog

    names = catalog_skill_names()
    assert "falsifiable-hypothesis-design" in names
    assert "divergent-hypothesis-ideation" in names
    assert "critical-evaluation-of-claims-and-results" in names
    scopes = {s.scope for s in skills_catalog()}
    assert "core" in scopes and "math_combinatorics" in scopes


def test_available_index_has_core_and_a_domain():
    idx = available_skill_index()
    assert idx.get("core"), "core skills should be indexed"
    assert "math_combinatorics" in idx
