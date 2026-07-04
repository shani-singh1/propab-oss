"""Synthesis diversity and methodology filter (fixes.md B1, B2)."""
from __future__ import annotations

from propab.domain_modules.math_combinatorics.plugin import MathCombinatoricsPlugin
from propab.synthesis_diversity import (
    forced_problem_type,
    methodology_implementable,
    resolve_forced_problem_type,
)


def test_methodology_filter_rejects_simulated_annealing() -> None:
    kw = MathCombinatoricsPlugin().implementable_methodologies()
    assert not methodology_implementable(
        "Run simulated annealing on Sidon sets",
        "metaheuristic search",
        kw,
    )


def test_methodology_accepts_greedy() -> None:
    kw = MathCombinatoricsPlugin().implementable_methodologies()
    assert methodology_implementable(
        "Run greedy search for Sidon sets in {1,...,n}",
        "greedy construction",
        kw,
    )


def test_forced_from_active_cap_set_beliefs() -> None:
    beliefs = [
        "Cap-set CLP ratios decrease monotonically in F_3^n",
        "Cap-set plateau between n=8 and n=10",
    ]
    forced = resolve_forced_problem_type([], beliefs, streak=3)
    assert forced is not None
    assert forced != "cap_set"


def test_fallback_synthesis_seeds_sidon() -> None:
    from propab.synthesis_diversity import fallback_synthesis_seeds

    seeds = fallback_synthesis_seeds("sidon", generation=1)
    assert len(seeds) == 1
    assert "Sidon" in seeds[0]["text"]


def test_forced_problem_type_after_sidon_streak() -> None:
    history = [{"problem_type": "sidon"} for _ in range(5)]
    forced = forced_problem_type(history, streak=5)
    assert forced is not None
    assert forced != "sidon"
