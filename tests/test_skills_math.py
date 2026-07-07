"""Domain-dependent research skills for math_combinatorics and coding_theory.

These lock in that the authored domain skills (a) load for the right domain +
phase, (b) never leak into a different domain, (c) come AFTER the core skills so
general methodology frames the domain technique, and (d) stay grounded in what the
verifier can actually compute (they name the real primitives, not just prose)."""
from __future__ import annotations

from propab.skills import available_skill_index, load_skills

_MATH_HYPOTHESIS = {
    "polynomial-slice-rank-method",
    "sidon-sets-and-additive-energy",
}
_MATH_EXPERIMENT = {"growth-exponent-open-bounds"}
_CODING_HYPOTHESIS = {
    "linear-code-construction-families",
    "distance-bounds-and-open-regime",
}
_CODING_EXPERIMENT = {"witnessed-code-vs-best-known"}


def _names(domain: str, phase: str) -> set[str]:
    return {s.name for s in load_skills(domain, phase)}


def test_math_combinatorics_domain_skills_present_by_phase():
    hyp = _names("math_combinatorics", "hypothesis")
    exp = _names("math_combinatorics", "experiment")
    assert _MATH_HYPOTHESIS <= hyp, f"missing math hypothesis skills: {_MATH_HYPOTHESIS - hyp}"
    assert _MATH_EXPERIMENT <= exp, f"missing math experiment skills: {_MATH_EXPERIMENT - exp}"


def test_coding_theory_domain_skills_present_by_phase():
    hyp = _names("coding_theory", "hypothesis")
    exp = _names("coding_theory", "experiment")
    assert _CODING_HYPOTHESIS <= hyp, f"missing coding hypothesis skills: {_CODING_HYPOTHESIS - hyp}"
    assert _CODING_EXPERIMENT <= exp, f"missing coding experiment skills: {_CODING_EXPERIMENT - exp}"


def test_domain_skills_do_not_cross_leak():
    coding = load_skills("coding_theory", "hypothesis") + load_skills("coding_theory", "experiment")
    assert not any(s.scope == "math_combinatorics" for s in coding)
    mathy = load_skills("math_combinatorics", "hypothesis") + load_skills("math_combinatorics", "experiment")
    assert not any(s.scope == "coding_theory" for s in mathy)


def test_core_precedes_domain_for_both_domains():
    for domain in ("math_combinatorics", "coding_theory"):
        for phase in ("hypothesis", "experiment"):
            scopes = [s.scope for s in load_skills(domain, phase)]
            if domain in scopes and "core" in scopes:
                assert scopes.index("core") < scopes.index(domain), (
                    f"core must precede {domain} skills in {phase}"
                )


def test_all_authored_skills_carry_a_description():
    for domain, phase in (
        ("math_combinatorics", "hypothesis"),
        ("math_combinatorics", "experiment"),
        ("coding_theory", "hypothesis"),
        ("coding_theory", "experiment"),
    ):
        for s in load_skills(domain, phase):
            if s.scope != domain:
                continue
            assert s.description.strip(), f"{s.name} needs a description"
            assert s.phase in ("hypothesis", "experiment"), f"{s.name} unexpected phase {s.phase}"


def test_skills_are_grounded_in_computable_primitives():
    """Each domain skill must reference a primitive the verifier actually computes,
    and every one must carry the anti-rediscovery guardrail — so the injected
    methodology matches the engine instead of inviting unfalsifiable ideation."""
    coding_blob = " ".join(
        s.body.lower()
        for s in load_skills("coding_theory", "hypothesis") + load_skills("coding_theory", "experiment")
        if s.scope == "coding_theory"
    )
    # Real coding_theory verifier primitives.
    for token in ("generator matrix", "minimum distance", "witness", "hamming", "gilbert"):
        assert token in coding_blob, f"coding skills should mention {token!r}"

    math_blob = " ".join(
        s.body.lower()
        for s in load_skills("math_combinatorics", "hypothesis") + load_skills("math_combinatorics", "experiment")
        if s.scope == "math_combinatorics"
    )
    for token in ("slice rank", "sidon", "growth exponent", "sumset", "2.756"):
        assert token in math_blob, f"math skills should mention {token!r}"

    # Anti-rediscovery guardrail present in every authored domain skill.
    for domain in ("math_combinatorics", "coding_theory"):
        for phase in ("hypothesis", "experiment"):
            for s in load_skills(domain, phase):
                if s.scope != domain or s.name in {
                    "cap-set-and-extremal-constructions",  # exemplar, tested elsewhere
                }:
                    continue
                assert "rediscovery" in s.body.lower(), (
                    f"{s.name} must carry an anti-rediscovery guardrail"
                )


def test_index_lists_both_domains():
    idx = available_skill_index()
    assert _MATH_HYPOTHESIS <= set(idx.get("math_combinatorics", []))
    assert _CODING_HYPOTHESIS <= set(idx.get("coding_theory", []))
