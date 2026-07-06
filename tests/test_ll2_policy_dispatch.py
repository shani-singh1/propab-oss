"""LL2: the learned search policy nudges live frontier scoring (bounded).

Verifies the loaded SearchPolicy's ``theme_weight`` moves the frontier score —
enough to reorder otherwise-equal candidates, never enough to dominate the
information-gain signal — and is exactly a no-op when no policy is loaded.
"""
from __future__ import annotations

from propab.hypothesis_tree import HypothesisTree


class _FakePolicy:
    """Minimal SearchPolicy-like stub: a uniform ``theme_weight`` + blocked sigs."""

    def __init__(self, weight_all: float = 1.0, blocked: tuple = ()) -> None:
        self._weight = float(weight_all)
        self.blocked_failure_signatures = tuple(blocked)

    def theme_weight(self, theme: str) -> float:  # noqa: ARG002 - uniform on purpose
        return self._weight


def _node(tree: HypothesisTree, theme: str = "alpha"):
    return tree.add_seeds(
        [{"text": f"claim about {theme}", "test_methodology": "sweep", "theme_id": theme}],
        generation=0,
    )[0]


def test_policy_multiplier_is_noop_without_policy() -> None:
    tree = HypothesisTree()
    node = _node(tree)
    tree.set_scoring_context("Q", [])  # no policy loaded
    assert tree._policy_score_multiplier(node, "alpha") == 1.0


def test_policy_multiplier_boosts_and_penalizes() -> None:
    tree = HypothesisTree()
    node = _node(tree)
    tree.set_scoring_context("Q", [], policy=_FakePolicy(2.0))
    assert tree._policy_score_multiplier(node, "alpha") > 1.0
    tree.set_scoring_context("Q", [], policy=_FakePolicy(0.5))
    assert tree._policy_score_multiplier(node, "alpha") < 1.0


def test_policy_nudge_is_bounded_never_dominates() -> None:
    tree = HypothesisTree()
    node = _node(tree)
    # An extreme boost cannot take over the information-gain signal.
    tree.set_scoring_context("Q", [], policy=_FakePolicy(1e6))
    assert 1.0 < tree._policy_score_multiplier(node, "alpha") <= 1.2
    # An extreme penalty cannot zero the score either.
    tree.set_scoring_context("Q", [], policy=_FakePolicy(0.0))
    assert 0.6 <= tree._policy_score_multiplier(node, "alpha") < 1.0


def test_boosting_policy_raises_information_gain_score() -> None:
    tree = HypothesisTree()
    node = _node(tree)
    tree.set_scoring_context("Q", [])
    base = tree._information_gain_score(node)
    tree.set_scoring_context("Q", [], policy=_FakePolicy(2.0))
    assert tree._information_gain_score(node) > base
