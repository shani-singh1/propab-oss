"""Synthesis diversity and methodology filter (fixes.md B1, B2)."""
from __future__ import annotations

from propab.domain_modules.math_combinatorics.plugin import MathCombinatoricsPlugin
from propab.synthesis_diversity import (
    forced_problem_type,
    methodology_implementable,
    resolve_forced_problem_type,
)


def test_dedup_allows_scope_narrowing_but_rejects_rephrasing() -> None:
    """Convergence works by narrowing: a child tests a stricter numeric region
    than its parent. Those texts are ~0.96 similar (only the numbers differ), so
    the pure-text dedup used to reject them — deleting the narrowing sequence and
    starving the frontier (investigation report §6b). Scope/parameter-aware dedup
    must let a DISTINCT-region candidate through while still rejecting a true
    rephrasing (same claim AND same numbers)."""
    from propab.belief_state import CampaignBeliefState
    from propab.campaign_synthesis import _is_duplicate_frontier_candidate
    from propab.hypothesis_tree import HypothesisTree

    tree = HypothesisTree()
    tree.add_seeds(
        [{"text": "Greedy F(n)/sqrt(n) crosses 0.60 within n in [34000,36000].\n"
                  "Population: greedy Sidon sequences\nDistribution: n in [34000,36000]\n"
                  "Claimed generalization: crossing in [34000,36000]\n"
                  "Expected failure modes: dip\nOOD test: dense grid",
          "test_methodology": "numeric_sweep"}],
        generation=0,
    )
    state = CampaignBeliefState()

    # Distinct region (a narrowing step) — must NOT be a duplicate.
    narrowing = ("Greedy F(n)/sqrt(n) crosses 0.60 within n in [34500,35500].\n"
                 "Population: greedy Sidon sequences\nDistribution: n in [34500,35500]\n"
                 "Claimed generalization: crossing in [34500,35500]\n"
                 "Expected failure modes: dip\nOOD test: dense grid")
    dup, _ = _is_duplicate_frontier_candidate(narrowing, tree, state, same_round_texts=[])
    assert dup is False

    # Same region, reworded — IS a duplicate (same experiment).
    rephrase = ("F(n)/sqrt(n) for the greedy sequence drops under 0.60 in n in [34000,36000].\n"
                "Population: greedy Sidon sequences\nDistribution: n in [34000,36000]\n"
                "Claimed generalization: crossing in [34000,36000]\n"
                "Expected failure modes: dip\nOOD test: dense grid")
    dup2, _ = _is_duplicate_frontier_candidate(rephrase, tree, state, same_round_texts=[])
    assert dup2 is True


def test_deepening_confirmed_bypasses_type_diversity_force() -> None:
    """The anti-monoculture type-diversity force must NOT reject the narrowing of
    a CONFIRMED finding — deepening one finding is inherently single-type and is
    exactly convergence. Without this the diversity force rejected 8/8 narrowing
    children and the search stalled at depth ~4 (investigation report §6e)."""
    from propab.campaign_synthesis import _is_deepening_confirmed
    from propab.hypothesis_tree import HypothesisNode

    confirmed = HypothesisNode(id="p", text="finding", parent_id=None, depth=1, verdict="confirmed")
    inconclusive = HypothesisNode(id="q", text="other", parent_id=None, depth=1, verdict="inconclusive")
    by_id = {"p": confirmed, "q": inconclusive}
    # narrowing (boundary) of a confirmed parent -> exempt from diversity force
    assert _is_deepening_confirmed({"expansion_type": "boundary", "parent_id": "p"}, by_id) is True
    # a lateral/alternative expansion is not a deepening move -> still subject to it
    assert _is_deepening_confirmed({"expansion_type": "alternative", "parent_id": "p"}, by_id) is False
    # deepening an inconclusive parent is not convergence of a finding -> subject to it
    assert _is_deepening_confirmed({"expansion_type": "boundary", "parent_id": "q"}, by_id) is False


def test_narrowing_child_survives_high_relevance_gate() -> None:
    """A narrowing child's lexical question-relevance drops as it narrows, so a
    child (refinement of an in-tree, on-topic parent) must bypass the lexical
    relevance THRESHOLD — else the search can't narrow past level 1 (report §6d).
    Even at a punishing threshold the child is kept."""
    from propab.belief_state import CampaignBeliefState
    from propab.campaign_synthesis import apply_synthesis_to_frontier
    from propab.hypothesis_tree import HypothesisTree

    tree = HypothesisTree()
    parent = tree.add_seeds([{"text": "crossing near n=35000", "test_methodology": "numeric_sweep"}], generation=0)[0]
    tree.update_node(parent.id, "confirmed", 0.9, 'evidence={"verdict_reason": "sweep"};')
    parsed = {"beliefs": [], "frontier_candidates": [{
        "id": "c1", "parent_id": parent.id, "expansion_type": "boundary",
        "text": "crossing within n in [34500,35500]", "test_methodology": "numeric_sweep",
    }], "direction_exhausted": False}
    added, metrics = apply_synthesis_to_frontier(
        tree, CampaignBeliefState(), parsed, question="Where does the ratio cross 0.60 for n in [10000,50000]?",
        generation=1, relevance_threshold=0.99,
    )
    assert len(added) == 1  # kept despite threshold 0.99, because it's a child refinement
    assert metrics.get("n_rejected_low_relevance", 0) == 0


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
    assert "decreasing" in seeds[0]["text"].lower()


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


def test_tree_monoculture_forces_at_8_nodes() -> None:
    tree = {"cap_set": 14, "sidon": 3}
    forced = resolve_forced_problem_type([], [], tree_problem_counts=tree)
    assert forced == "sidon"


def test_resolve_forced_prefers_tree_monoculture() -> None:
    tree = {"cap_set": 160, "sidon": 20, "ap_free": 2}
    forced = resolve_forced_problem_type([], [], tree_problem_counts=tree)
    assert forced == "ap_free"


def test_filter_seed_dicts_rejects_cap_set_when_forcing_sidon() -> None:
    from propab.synthesis_diversity import filter_seed_dicts_for_diversity

    caps = [{
        "text": "Cap-set CLP ratio in F_3^8",
        "test_methodology": "cap-set CLP table lookup",
    }]
    beliefs = ["Cap-set CLP ratios decrease monotonically in F_3^n"]
    out = filter_seed_dicts_for_diversity(
        caps,
        tree_nodes={},
        active_belief_statements=beliefs,
        generation=1,
    )
    assert len(out) == 1
    assert "Sidon" in out[0]["text"] or "sidon" in out[0]["text"].lower()
