"""Extract NodeOperatorTrace from hypothesis trees and campaign artifacts."""
from __future__ import annotations

from typing import Any

from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.operator_credit.operator_registry import DEFAULT_OPERATORS, OperatorFamily
from propab.operator_credit.operator_state import state_from_node
from propab.operator_credit.operator_trace import NodeOperatorTrace, OperatorStep, OperatorTraceLedger

_EXPANSION_TO_MUTATION = {
    "boundary": "boundary",
    "mechanistic": "local_refinement",
    "generalization": "local_refinement",
    "alternative": "contradiction",
    "retest": "local_refinement",
}

_VERIFICATION_MAP = {
    "symbolic_identity": "symbolic",
    "finite_scan": "symbolic",
    "counterexample": "symbolic",
    "statistical": "numerical",
    "unknown": "numerical",
}


def _infer_branching(tree: HypothesisTree, node: HypothesisNode) -> str:
    if node.frontier_score is not None and node.frontier_score > 0.5:
        return "closure_aware"
    if node.depth > 3:
        return "depth_first"
    return "breadth_first"


def _infer_decomposition(node: HypothesisNode) -> str:
    if node.verdict == "confirmed":
        return "confirmed_expand"
    if node.verdict == "refuted":
        return "refuted_expand"
    if node.verdict == "inconclusive":
        return "inconclusive_retest"
    return DEFAULT_OPERATORS[OperatorFamily.DECOMPOSITION]


def _infer_retrieval(node: HypothesisNode) -> str:
    finding = node.finding or {}
    tools = finding.get("tools") or finding.get("tool_trace") or []
    if isinstance(tools, list) and len(tools) > 2:
        return "hybrid"
    if node.evidence_summary and len(node.evidence_summary) > 200:
        return "semantic"
    return "bm25"


def _node_cost(node: HypothesisNode) -> float:
  base = 1.0
  if node.depth > 4:
      base += 0.5
  if node.verification_method in ("finite_scan", "symbolic_identity"):
      base += 0.3
  return round(base, 2)


def extract_trace_for_node(
    *,
    campaign_id: str,
    node: HypothesisNode,
    tree: HypothesisTree,
    order: int,
) -> NodeOperatorTrace:
    branching = _infer_branching(tree, node)
    mutation = _EXPANSION_TO_MUTATION.get(node.expansion_type or "", "local_refinement")
    verification = _VERIFICATION_MAP.get(node.verification_method or "unknown", "numerical")
    retrieval = _infer_retrieval(node)
    decomposition = _infer_decomposition(node)
    if node.verdict == "refuted":
        decomposition = "refuted_expand"
    elif node.verdict == "inconclusive":
        decomposition = "inconclusive_retest"
    elif node.verdict == "confirmed":
        decomposition = "confirmed_expand"

    steps = [
        OperatorStep(family="retrieval", operator=retrieval, order=0),
        OperatorStep(family="branching", operator=branching, order=1),
        OperatorStep(family="mutation", operator=mutation, order=2),
        OperatorStep(family="verification", operator=verification, order=3),
        OperatorStep(family="model", operator="default_llm", order=4),
        OperatorStep(family="decomposition", operator=decomposition, order=5),
    ]
    return NodeOperatorTrace(
        campaign_id=campaign_id,
        node_id=node.id,
        operators_used=steps,
        order=order,
        cost=_node_cost(node),
        outcome=node.verdict,
        retrieval=retrieval,
        branching=branching,
        mutation=mutation,
        verification=verification,
        model="default_llm",
        decomposition=decomposition,
        state_vector=state_from_node(node, tree),
        source="tree",
    )


def extract_traces_from_tree(
    *,
    campaign_id: str,
    tree: HypothesisTree,
) -> OperatorTraceLedger:
    ledger = OperatorTraceLedger()
    tested = [
        n for n in tree.nodes.values()
        if n.verdict in ("confirmed", "refuted", "inconclusive")
    ]
    tested.sort(key=lambda n: n.generation)
    for i, node in enumerate(tested):
        ledger.add(extract_trace_for_node(
            campaign_id=campaign_id,
            node=node,
            tree=tree,
            order=i,
        ))
    return ledger


def extract_traces_from_snapshots(
    *,
    campaign_id: str,
    snapshots: list[dict[str, Any]],
) -> OperatorTraceLedger:
    """Approximate traces when hypothesis tree is unavailable."""
    from propab.operator_credit.operator_state import state_from_snapshot

    ledger = OperatorTraceLedger()
    for i, snap in enumerate(snapshots[1:], start=1):
        prev = snapshots[i - 1]
        tested_delta = int(snap.get("tested") or 0) - int(prev.get("tested") or 0)
        branching = "closure_aware" if float(snap.get("closure_ratio") or 0) > 0.3 else "breadth_first"
        ledger.add(NodeOperatorTrace(
            campaign_id=campaign_id,
            node_id=f"snap-{i}",
            operators_used=[
                OperatorStep("retrieval", "hybrid", 0),
                OperatorStep("branching", branching, 1),
                OperatorStep("mutation", "local_refinement", 2),
                OperatorStep("verification", "numerical", 3),
                OperatorStep("model", "default_llm", 4),
                OperatorStep("decomposition", "confirmed_expand", 5),
            ],
            order=i,
            cost=round(max(1, tested_delta) * 0.5, 2),
            outcome="inconclusive",
            branching=branching,
            state_vector=state_from_snapshot(snap),
            source="snapshot",
        ))
    return ledger
