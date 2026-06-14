"""State vectors for operator statistics conditioning."""
from __future__ import annotations

from typing import Any

from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.layer05.replay_state import ReplayState


def state_from_node(node: HypothesisNode, tree: HypothesisTree) -> list[float]:
    tested = sum(1 for n in tree.nodes.values() if n.verdict != "pending")
    pending = sum(1 for n in tree.nodes.values() if n.verdict == "pending")
    confirmed = sum(1 for n in tree.nodes.values() if n.verdict == "confirmed")
    total = max(1, tested)
    closure = (confirmed + sum(1 for n in tree.nodes.values() if n.verdict == "refuted")) / total
    return [
        round(node.depth / 10.0, 4),
        round(closure, 4),
        round(pending / 20.0, 4),
        round(tested / 250.0, 4),
        round((node.confidence or 0), 4),
    ]


def state_from_snapshot(snap: dict[str, Any]) -> list[float]:
    rs = ReplayState.from_snapshot(snap)
    return [
        round(rs.entropy / 3.0, 4),
        round(rs.closure_ratio, 4),
        round(rs.pending_nodes / 20.0, 4),
        round(rs.tested_count / 250.0, 4),
        round(rs.frontier_size / 20.0, 4),
    ]


def state_bucket(state: list[float], *, precision: int = 2) -> str:
    return "|".join(str(round(v, precision)) for v in state[:4])
