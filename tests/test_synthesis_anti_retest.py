"""Anti-monoculture / anti-retest guard: CLAIM-SKELETON matching.

The generation overhaul added a CODE-level guard (not just a prompt instruction)
that rejects a proposed synthesis candidate which is a parametric RE-TEST — the
same claim skeleton with different numbers / parameters / identifiers — of an
already-tested tree node, while accepting a genuinely distinct hypothesis. The
guard is DOMAIN-GENERAL (structural normalization only; no field vocabulary) and
exempts genuine deepening of a confirmed lineage.
"""

from __future__ import annotations

from propab.belief_state import CampaignBeliefState
from propab.campaign_synthesis import (
    apply_synthesis_to_frontier,
    claim_skeleton,
    is_retest_of_tested_node,
    skeletons_match,
)
from propab.hypothesis_tree import HypothesisNode, HypothesisTree


# ---------------------------------------------------------------------------
# Pure skeleton matching (domain-general).
# ---------------------------------------------------------------------------

def test_skeleton_collapses_parametric_variants() -> None:
    a = "Greedy construction density F(n)/sqrt(n) is strictly decreasing for n in [1000, 5000]."
    b = "Greedy construction density F(n)/sqrt(n) is strictly decreasing for n in [20000, 40000] with lookahead k=3."
    assert skeletons_match(claim_skeleton(a), claim_skeleton(b))


def test_skeleton_distinguishes_distinct_claims() -> None:
    a = "Greedy construction density F(n)/sqrt(n) is strictly decreasing for n in [1000, 5000]."
    c = "A random baseline sumset grows faster than a structured construction at matched cardinality."
    assert not skeletons_match(claim_skeleton(a), claim_skeleton(c))


def test_skeleton_is_domain_general_biology() -> None:
    """Same guard works with non-math (biology) claims — no math-specific logic."""
    d = "Thermophilic RT enzymes show higher fidelity than mesophilic enzymes at 65C over 200 cycles."
    e = "Thermophilic RT enzymes show higher fidelity than mesophilic enzymes at 72C over 500 cycles."
    f = "Codon usage bias predicts translation efficiency in yeast ribosome profiling data."
    assert skeletons_match(claim_skeleton(d), claim_skeleton(e))   # retest
    assert not skeletons_match(claim_skeleton(d), claim_skeleton(f))  # distinct


# ---------------------------------------------------------------------------
# Guard against already-TESTED nodes.
# ---------------------------------------------------------------------------

def _tree_with_tested_claim(text: str, *, verdict: str = "confirmed") -> HypothesisTree:
    tree = HypothesisTree()
    tree.nodes["n0"] = HypothesisNode(
        id="n0", text=text, parent_id=None, depth=0, verdict=verdict
    )
    if verdict == "confirmed":
        tree.confirmed.append("n0")
    return tree


def test_is_retest_flags_parametric_variant_of_tested_node() -> None:
    tested = "Greedy construction density F(n)/sqrt(n) is strictly decreasing for n in [1000, 5000]."
    tree = _tree_with_tested_claim(tested)
    variant = "Greedy construction density F(n)/sqrt(n) is strictly decreasing for n in [30000, 60000] with dim k=4."
    retest, reason = is_retest_of_tested_node(variant, tree)
    assert retest
    assert reason.startswith("retest_skeleton:")


def test_is_retest_accepts_distinct_claim() -> None:
    tested = "Greedy construction density F(n)/sqrt(n) is strictly decreasing for n in [1000, 5000]."
    tree = _tree_with_tested_claim(tested)
    distinct = "A random baseline sumset grows faster than a structured construction at matched cardinality."
    retest, _ = is_retest_of_tested_node(distinct, tree)
    assert not retest


def test_is_retest_ignores_untested_pending_nodes() -> None:
    tree = _tree_with_tested_claim(
        "Greedy density F(n)/sqrt(n) decreasing for n in [1000, 5000].", verdict="pending"
    )
    variant = "Greedy density F(n)/sqrt(n) decreasing for n in [9000, 9000] with k=2."
    retest, _ = is_retest_of_tested_node(variant, tree)
    # A pending (not-yet-tested) node is not a settled result — not a retest.
    assert not retest


# ---------------------------------------------------------------------------
# Full frontier integration: retest rejected, distinct accepted.
# ---------------------------------------------------------------------------

_SCOPE = (
    "\nPopulation: instances in the stated family\n"
    "Distribution: the stated generating distribution\n"
    "Claimed generalization: transfers to a matched held-out family\n"
    "Expected failure modes: breaks outside the stated regime\n"
    "OOD test: evaluate on a held-out family before confirming"
)


def _candidate(text: str, *, expansion_type: str = "diagnostic", parent_id: str | None = None) -> dict:
    item = {"id": text[:8], "text": text + _SCOPE, "test_methodology": "sub_agent", "expansion_type": expansion_type}
    if parent_id:
        item["parent_id"] = parent_id
    return item


def test_frontier_rejects_retest_accepts_distinct() -> None:
    question = "Study a structured construction's density behaviour and alternatives."
    tested = (
        "Greedy construction density metric is strictly decreasing across the sweep for n in [1000, 5000]."
        + _SCOPE
    )
    tree = HypothesisTree()
    tree.nodes["n0"] = HypothesisNode(
        id="n0", text=tested, parent_id=None, depth=0, verdict="confirmed",
        test_methodology="sub_agent",
    )
    tree.confirmed.append("n0")

    belief_state = CampaignBeliefState()
    parsed = {
        "beliefs": [],
        "frontier_candidates": [
            # (1) A parametric RETEST of the tested node (new numbers/params) — reject.
            _candidate(
                "Greedy construction density metric is strictly decreasing across the sweep for n in [40000, 90000] with k=5.",
            ),
            # (2) A conceptually DISTINCT claim — accept.
            _candidate(
                "A random baseline construction yields a larger growth ratio than the structured construction at matched size.",
            ),
        ],
        "direction_exhausted": False,
    }

    added, metrics = apply_synthesis_to_frontier(
        tree, belief_state, parsed, question=question, generation=2,
    )

    assert metrics.get("n_rejected_retest", 0) >= 1, metrics
    added_texts = " ".join(n.text.lower() for n in added)
    assert "random baseline" in added_texts, [n.text[:70] for n in added]
    # The retest variant did not become a node.
    assert "40000" not in added_texts and "k=5" not in added_texts


def test_frontier_allows_deepening_confirmed_lineage() -> None:
    """A genuine narrowing (deepening) child of a CONFIRMED parent shares the
    parent's skeleton by design and must NOT be rejected as a retest."""
    question = "Study a structured construction's density behaviour."
    tested = (
        "Greedy construction density metric is strictly decreasing across the sweep for n in [1000, 5000]."
        + _SCOPE
    )
    tree = HypothesisTree()
    tree.nodes["n0"] = HypothesisNode(
        id="n0", text=tested, parent_id=None, depth=0, verdict="confirmed",
        test_methodology="sub_agent",
    )
    tree.confirmed.append("n0")

    belief_state = CampaignBeliefState()
    parsed = {
        "beliefs": [],
        "frontier_candidates": [
            _candidate(
                "Greedy construction density metric is strictly decreasing across the sweep for n in [2000, 3000].",
                expansion_type="boundary",
                parent_id="n0",
            ),
        ],
        "direction_exhausted": False,
    }

    added, metrics = apply_synthesis_to_frontier(
        tree, belief_state, parsed, question=question, generation=2,
    )
    # Deepening a confirmed finding is exempt from the retest guard.
    assert metrics.get("n_rejected_retest", 0) == 0, metrics
    assert len(added) == 1, [n.text[:70] for n in added]
