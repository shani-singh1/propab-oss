"""FamilyRegistry: is the search exploring different mathematical ideas, or one idea in ten outfits?"""
from __future__ import annotations

from propab.evolve.island import (
    FAMILY_KEY,
    REDISCOVERY_KEY,
    UNKNOWN_FAMILY,
    FamilyRegistry,
)
from propab.evolve.program import Program


def prog(tag: str, family: str, *, score: float = 5.0, valid: bool = True, redis: bool = False):
    detail: dict[str, object] = {FAMILY_KEY: family}
    if redis:
        detail[REDISCOVERY_KEY] = True
    return Program(code=f"# {tag}\ndef build():\n    return []\n", score=score, valid=valid, detail=detail)


def test_only_verified_programs_count_as_an_approach():
    """An idea that never produced a valid candidate is not an approach the search is 'using'."""
    reg = FamilyRegistry()
    reg.observe(prog("a", "algebraic"))
    reg.observe(prog("b", "algebraic", score=float("-inf"), valid=False))
    assert reg.counts() == {"algebraic": 1}
    assert reg.total() == 1


def test_observation_is_deduped_by_program_id():
    reg = FamilyRegistry()
    reg.observe(prog("a", "algebraic"))
    reg.observe(prog("a", "algebraic"))  # same code, re-derived
    assert reg.counts() == {"algebraic": 1}


def test_dominance_needs_evidence():
    """Two programs of one family is not a takeover, it is a coincidence."""
    reg = FamilyRegistry()
    reg.observe(prog("a", "algebraic"))
    reg.observe(prog("b", "algebraic"))
    assert reg.dominant() is None
    assert reg.exploration_hint() is None


def test_dominant_family_is_detected_and_warned_about():
    reg = FamilyRegistry()
    for i in range(4):
        reg.observe(prog(f"a{i}", "algebraic"))
    reg.observe(prog("c", "concatenation"))

    assert reg.dominant() == "algebraic"
    hint = reg.exploration_hint()
    assert "algebraic" in hint
    assert "80%" in hint
    assert "concatenation" in hint  # points at what to try instead


def test_untagged_programs_are_never_dominant():
    """'unknown' is not an approach — it is the absence of one. It must not trigger a redirect."""
    reg = FamilyRegistry()
    for i in range(6):
        reg.observe(prog(f"u{i}", UNKNOWN_FAMILY))
    assert reg.dominant() is None


def test_circular_family_is_flagged():
    """A family that keeps re-deriving the seed scores points but discovers nothing."""
    reg = FamilyRegistry()
    reg.observe(prog("a", "shortening", redis=True))
    reg.observe(prog("b", "shortening", redis=True))
    reg.observe(prog("c", "concatenation"))

    assert reg.circular() == ["shortening"]
    hint = reg.exploration_hint()
    assert "re-deriving the seed" in hint


def test_underexplored_lists_the_rarest_ideas_first():
    reg = FamilyRegistry()
    for i in range(5):
        reg.observe(prog(f"a{i}", "algebraic"))
    reg.observe(prog("c1", "concatenation"))
    reg.observe(prog("c2", "concatenation"))
    reg.observe(prog("s1", "shortening"))

    assert reg.underexplored(limit=2) == ["shortening", "concatenation"]
