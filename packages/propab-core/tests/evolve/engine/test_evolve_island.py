"""Island: bounded top-k population, score-weighted sampling, diversity pressure."""
from __future__ import annotations

import random
from collections import Counter

import pytest

from propab.evolve.island import (
    FAMILY_KEY,
    REDISCOVERY_KEY,
    NEG_INF,
    ProgramIsland,
    clone,
    score_of,
)
from propab.evolve.program import Program


def prog(tag: str, score: float, *, family: str = "unknown", rediscovery: bool = False) -> Program:
    """A program whose code — and therefore whose id — is determined by `tag` ALONE.

    The score must not leak into the code, or "the same program re-evaluated" would come out as a
    different program and the dedupe tests would be testing nothing.
    """
    detail: dict[str, object] = {FAMILY_KEY: family}
    if rediscovery:
        detail[REDISCOVERY_KEY] = True
    return Program(
        code=f"# {tag}\ndef build():\n    return [(1,)]\n",
        score=score,
        valid=score > NEG_INF,
        detail=detail,
    )


# --------------------------------------------------------------------------- insert / dedupe


def test_insert_dedupes_by_program_id():
    island = ProgramIsland()
    island.insert(prog("a", 1.0))
    island.insert(prog("a", 1.0))  # identical code => identical id => same program
    assert len(island) == 1


def test_dedupe_keeps_the_better_evaluation():
    """A flaky re-run (timeout, sandbox hiccup) must not demote a known winner."""
    island = ProgramIsland()
    island.insert(prog("a", 5.0))
    island.insert(prog("a", 1.0))  # same code, worse score
    assert island.best().score == 5.0

    island.insert(prog("a", 9.0))  # same code, better score => adopt it
    assert island.best().score == 9.0
    assert len(island) == 1


def test_capacity_keeps_the_top_k():
    island = ProgramIsland(capacity=5)
    for i in range(20):
        island.insert(prog(f"p{i}", float(i)))
    assert len(island) == 5
    assert sorted(p.score for p in island.programs()) == [15.0, 16.0, 17.0, 18.0, 19.0]


def test_insert_stamps_the_island_index():
    island = ProgramIsland(index=3)
    p = prog("a", 1.0)
    island.insert(p)
    assert p.island == 3


def test_best_and_empty_island():
    island = ProgramIsland()
    assert island.best() is None
    assert island.best_score() == NEG_INF
    assert island.sample(2) == []
    assert len(island) == 0


def test_reset_wipes_and_restocks():
    island = ProgramIsland()
    for i in range(5):
        island.insert(prog(f"p{i}", float(i)))
    island.reset([prog("seed", 2.0)])
    assert len(island) == 1
    assert island.best().score == 2.0


def test_nan_score_does_not_poison_ordering():
    """nan compares False against everything; untreated it silently corrupts max()/sort()."""
    island = ProgramIsland()
    nan_prog = prog("nan", 0.0)
    nan_prog.score = float("nan")
    island.insert(nan_prog)
    island.insert(prog("real", 3.0))
    assert score_of(nan_prog) == NEG_INF
    assert island.best().score == 3.0


# --------------------------------------------------------------------------- sampling


def test_sampling_favours_winners():
    island = ProgramIsland(rng=random.Random(7))
    for i in range(10):
        island.insert(prog(f"p{i}", float(i)))

    picks = Counter()
    for _ in range(2000):
        picks[island.sample(1)[0].score] += 1

    mean = sum(s * n for s, n in picks.items()) / 2000
    assert mean > 6.0, f"sampling is not favouring winners (mean {mean:.2f} vs uniform 4.5)"
    assert picks[9.0] > 3 * picks[0.0]


def test_sampling_returns_distinct_parents():
    island = ProgramIsland(rng=random.Random(7))
    for i in range(5):
        island.insert(prog(f"p{i}", float(i)))
    for _ in range(50):
        parents = island.sample(3)
        assert len({p.id for p in parents}) == 3


def test_sample_caps_at_population_size():
    island = ProgramIsland()
    island.insert(prog("only", 1.0))
    assert len(island.sample(5)) == 1


def test_invalid_programs_are_never_bred_from_while_a_viable_one_exists():
    island = ProgramIsland(rng=random.Random(7))
    island.insert(prog("good", 1.0))
    for i in range(5):
        island.insert(prog(f"junk{i}", NEG_INF))
    for _ in range(200):
        assert island.sample(1)[0].code.startswith("# good")


def test_all_invalid_population_still_samples():
    """Every program is junk (nothing has verified yet). Uniform fallback — no NaN, no crash, no
    empty return: the loop still needs a parent to mutate."""
    island = ProgramIsland(rng=random.Random(7))
    for i in range(4):
        island.insert(prog(f"junk{i}", NEG_INF))
    picks = {island.sample(1)[0].id for _ in range(200)}
    assert len(picks) > 1  # genuinely uniform, not stuck on one


# --------------------------------------------------------------------------- diversity pressure


def _family_share(island: ProgramIsland, family: str, n: int = 2000) -> float:
    hits = sum(1 for _ in range(n) if island.sample(1)[0].detail[FAMILY_KEY] == family)
    return hits / n


def test_family_bias_redirects_sampling_toward_the_underexplored_family():
    """Nine programs of one idea, one of another, all scoring identically. Without a family bias the
    rare idea is picked 10% of the time; with full bias the two ideas are drawn equally."""
    def build(bias: float) -> ProgramIsland:
        island = ProgramIsland(family_bias=bias, rng=random.Random(11))
        for i in range(9):
            island.insert(prog(f"crowd{i}", 5.0, family="algebraic"))
        island.insert(prog("rare", 5.0, family="concatenation"))
        return island

    unbiased = _family_share(build(0.0), "concatenation")
    biased = _family_share(build(1.0), "concatenation")

    assert 0.05 < unbiased < 0.15, unbiased
    assert biased > 0.35, f"crowded family was not redirected away from (rare share {biased:.2f})"


def test_family_bias_does_not_override_score():
    """Diversity pressure redirects; it must not promote a bad idea over a good one."""
    island = ProgramIsland(family_bias=1.0, rng=random.Random(11))
    island.insert(prog("winner", 100.0, family="algebraic"))
    for i in range(5):
        island.insert(prog(f"weak{i}", 1.0, family="random-search"))
    assert _family_share(island, "algebraic", n=500) > 0.5


def test_seed_rediscovery_is_penalised_as_a_parent():
    """An elegant-but-circular program that just re-derives the seed must not take over selection."""
    island = ProgramIsland(rediscovery_penalty=0.25, rng=random.Random(11))
    island.insert(prog("fresh", 5.0, family="algebraic"))
    island.insert(prog("circular", 5.0, family="algebraic", rediscovery=True))

    fresh = sum(1 for _ in range(1000) if island.sample(1)[0].code.startswith("# fresh"))
    assert fresh > 700, f"circular program was not penalised (fresh picked {fresh}/1000)"


def test_families_composition():
    island = ProgramIsland()
    island.insert(prog("a", 1.0, family="algebraic"))
    island.insert(prog("b", 2.0, family="algebraic"))
    island.insert(prog("c", 3.0, family="concatenation"))
    assert island.families() == {"algebraic": 2, "concatenation": 1}


# --------------------------------------------------------------------------- misc


def test_clone_does_not_share_mutable_state():
    """Islands must own their members: a shared detail dict would let one island corrupt another."""
    original = prog("a", 1.0, family="algebraic")
    original.parents = ["p1"]
    copy = clone(original)
    copy.detail["family"] = "other"
    copy.parents.append("p2")
    assert original.detail[FAMILY_KEY] == "algebraic"
    assert original.parents == ["p1"]
    assert copy.id == original.id  # same code => same identity


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capacity": 0}, "capacity"),
        ({"temperature": 0.0}, "temperature"),
        ({"family_bias": 1.5}, "family_bias"),
        ({"rediscovery_penalty": 0.0}, "rediscovery_penalty"),
    ],
)
def test_rejects_nonsense_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ProgramIsland(**kwargs)
