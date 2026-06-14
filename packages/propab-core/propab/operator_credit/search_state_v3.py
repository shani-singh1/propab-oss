"""SearchStateV3 — richer search state representation (fixes.md #7)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.layer05.replay_state import ReplayState
from propab.operator_credit.operator_trace import NodeOperatorTrace


@dataclass
class SearchStateV3:
    """
    Captures uncertainty, saturation, diversity, operator history,
    frontier topology, and lineage — everything downstream depends on this.
    """

    uncertainty: float
    saturation: float
    diversity: float
    closure_ratio: float
    entropy: float
    tested_fraction: float
    pending_fraction: float
    frontier_size_norm: float
    operator_history: dict[str, float] = field(default_factory=dict)
    frontier_topology: dict[str, float] = field(default_factory=dict)
    lineage: dict[str, float] = field(default_factory=dict)
    theme_histogram: dict[str, int] = field(default_factory=dict)

    def to_vector(self) -> list[float]:
        hist = self.operator_history
        topo = self.frontier_topology
        lin = self.lineage
        return [
            round(self.uncertainty, 4),
            round(self.saturation, 4),
            round(self.diversity, 4),
            round(self.closure_ratio, 4),
            round(self.entropy / 3.0, 4),
            round(self.tested_fraction, 4),
            round(self.pending_fraction, 4),
            round(self.frontier_size_norm, 4),
            round(hist.get("branching", 0), 4),
            round(hist.get("retrieval", 0), 4),
            round(hist.get("verification", 0), 4),
            round(topo.get("depth_spread", 0), 4),
            round(topo.get("breadth", 0), 4),
            round(lin.get("mean_depth", 0), 4),
            round(lin.get("max_depth", 0), 4),
        ]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["vector"] = self.to_vector()
        d["dim"] = len(self.to_vector())
        return d

    @classmethod
    def from_snapshot(
        cls,
        snap: dict[str, Any],
        *,
        max_tested: float = 250.0,
        operator_history: dict[str, float] | None = None,
    ) -> SearchStateV3:
        rs = ReplayState.from_snapshot(snap)
        hist = rs.theme_histogram
        total = sum(hist.values()) or 1
        top = max(hist.values()) if hist else 0
        saturation = top / total
        diversity = 1.0 - saturation
        uncertainty = min(1.0, rs.entropy / 2.5) * (1.0 - rs.closure_ratio)

        return cls(
            uncertainty=round(uncertainty, 4),
            saturation=round(saturation, 4),
            diversity=round(diversity, 4),
            closure_ratio=round(rs.closure_ratio, 4),
            entropy=round(rs.entropy, 4),
            tested_fraction=round(rs.tested_count / max_tested, 4),
            pending_fraction=round(rs.pending_nodes / 20.0, 4),
            frontier_size_norm=round(rs.frontier_size / 20.0, 4),
            operator_history=operator_history or {},
            frontier_topology={
                "breadth": round(rs.frontier_size / max(1, rs.pending_nodes + 1), 4),
                "depth_spread": 0.5,
            },
            lineage={"mean_depth": 0.0, "max_depth": 0.0},
            theme_histogram=dict(hist),
        )

    @classmethod
    def from_node_and_tree(
        cls,
        node: HypothesisNode,
        tree: HypothesisTree,
        trace: NodeOperatorTrace | None = None,
    ) -> SearchStateV3:
        tested = sum(1 for n in tree.nodes.values() if n.verdict != "pending")
        pending = sum(1 for n in tree.nodes.values() if n.verdict == "pending")
        confirmed = sum(1 for n in tree.nodes.values() if n.verdict == "confirmed")
        total = max(1, tested)
        closure = (confirmed + sum(1 for n in tree.nodes.values() if n.verdict == "refuted")) / total
        depths = [n.depth for n in tree.nodes.values()]
        hist: dict[str, int] = {}
        for n in tree.nodes.values():
            t = n.primary_theme or "general"
            hist[t] = hist.get(t, 0) + 1
        th_total = sum(hist.values()) or 1
        top = max(hist.values()) if hist else 1

        op_hist: dict[str, float] = {}
        if trace:
            op_hist = {
                "branching": 1.0 if trace.branching == "closure_aware" else 0.5,
                "retrieval": 1.0 if trace.retrieval == "hybrid" else 0.3,
                "verification": 1.0 if trace.verification == "symbolic" else 0.5,
            }

        entropy = float(node.theme_confidence or 0.5) * 2.0
        uncertainty = min(1.0, entropy / 2.5) * (1.0 - closure)

        return cls(
            uncertainty=round(uncertainty, 4),
            saturation=round(top / th_total, 4),
            diversity=round(1.0 - top / th_total, 4),
            closure_ratio=round(closure, 4),
            entropy=round(entropy, 4),
            tested_fraction=round(tested / 250.0, 4),
            pending_fraction=round(pending / 20.0, 4),
            frontier_size_norm=round(len(tree.frontier) / 20.0, 4),
            operator_history=op_hist,
            frontier_topology={
                "breadth": round(len(tree.frontier) / max(1, pending + 1), 4),
                "depth_spread": round((max(depths) - min(depths)) / max(1, max(depths)), 4) if depths else 0,
            },
            lineage={
                "mean_depth": round(sum(depths) / max(1, len(depths)), 4),
                "max_depth": float(max(depths) if depths else 0),
            },
            theme_histogram=hist,
        )
