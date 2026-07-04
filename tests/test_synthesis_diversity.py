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
    history = [{"problem_type": "cap_set"} for _ in range(5)]
    forced = resolve_forced_problem_type(history, beliefs, streak=3)
    assert forced is not None
    assert forced != "cap_set"


def test_history_streak_wins_over_sidon_beliefs() -> None:
    """Cap-set history should still force Sidon even when active beliefs are Sidon."""
    beliefs = ["Greedy Sidon F(n)/sqrt(n) is monotonically decreasing"]
    history = [{"problem_type": "cap_set"} for _ in range(5)]
    forced = resolve_forced_problem_type(history, beliefs, streak=3)
    assert forced == "sidon"


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


def test_tree_monoculture_forces_least_explored() -> None:
    from propab.synthesis_diversity import forced_from_tree_monoculture

    counts = {"cap_set": 166, "sidon": 22}
    forced = forced_from_tree_monoculture(counts)
    assert forced is not None
    assert forced != "cap_set"
    assert forced == "sidon"


def test_resolve_forced_prefers_tree_monoculture() -> None:
    tree = {"cap_set": 160, "sidon": 20, "ap_free": 2}
    forced = resolve_forced_problem_type([], [], tree_problem_counts=tree)
    assert forced == "ap_free"


def test_bootstrap_forced_sidon_on_q1_question() -> None:
    from propab.synthesis_diversity import bootstrap_forced_problem_type

    q = "where does it first fall below 0.60? Run threshold sweeps for n in [10000, 50000]"
    assert bootstrap_forced_problem_type(q) == "sidon"


def test_filter_seed_dicts_rejects_cap_set_when_forcing_sidon() -> None:
    from propab.synthesis_diversity import filter_seed_dicts_for_diversity

    caps = [{
        "text": "Cap-set CLP ratio in F_3^8",
        "test_methodology": "cap-set CLP table lookup",
    }]
    out = filter_seed_dicts_for_diversity(
        caps,
        tree_nodes={},
        question="fall below 0.60 n in [10000, 50000]",
        generation=1,
    )
    assert len(out) == 1
    assert "Sidon" in out[0]["text"] or "sidon" in out[0]["text"].lower()
