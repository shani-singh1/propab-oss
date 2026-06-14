"""Policy-aware frontier dispatch — deterministic, no LLM."""
from __future__ import annotations

from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.policy_record import PolicyRecord
from propab.search_policy import SearchPolicy


def policy_from_record(record: PolicyRecord) -> SearchPolicy:
    return record.to_search_policy()


def policy_adjusted_score(
    node: HypothesisNode,
    tree: HypothesisTree,
    policy: SearchPolicy,
) -> float:
    base = tree._information_gain_score(node)
    theme = node.primary_theme or node.theme_id or "general"
    return base * policy.theme_weight(theme)


def select_dispatch(
    tree: HypothesisTree,
    policy: SearchPolicy,
    *,
    exclude_ids: frozenset[str] | None = None,
) -> HypothesisNode | None:
    ex = exclude_ids or frozenset()
    candidates = [
        tree.nodes[nid]
        for nid in tree.frontier
        if nid in tree.nodes
        and tree.nodes[nid].verdict == "pending"
        and nid not in ex
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda n: policy_adjusted_score(n, tree, policy),
        reverse=True,
    )
    return candidates[0]


def select_dispatch_baseline(tree: HypothesisTree) -> HypothesisNode | None:
    return tree.next_dispatch_candidate()
