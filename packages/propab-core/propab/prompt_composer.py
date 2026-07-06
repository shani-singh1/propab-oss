"""Prompt assembly and template-based node-history compression (fixes.md §5)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from propab.belief_state import CampaignBeliefState
from propab.hypothesis_tree import HypothesisNode, HypothesisTree

_PROMPTS_ROOT = Path(__file__).resolve().parents[3] / "prompts"


def prompts_root() -> Path:
    return _PROMPTS_ROOT


def load_prompt(name: str) -> str:
    path = prompts_root() / name
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _parse_evidence_blob(evidence_summary: str | None) -> dict[str, Any]:
    return HypothesisTree._parse_evidence_blob(evidence_summary)


def _artifact_gate_line(ev: dict[str, Any]) -> str:
    ag = ev.get("artifact_gate")
    if not isinstance(ag, dict):
        return "n/a"
    ranked = ag.get("ranked_artifacts") or []
    top = ranked[0] if ranked and isinstance(ranked[0], dict) else {}
    top_id = top.get("artifact_id") or "?"
    return f"{ag.get('verdict', '?')} — {top_id}: {str(ag.get('verdict_reason') or '')[:120]}"


def format_node_compact(node: HypothesisNode, *, include_prose: bool = True) -> str:
    """Template-based compact node line — no LLM paraphrase."""
    ev = _parse_evidence_blob(node.evidence_summary)
    lines = [
        f"node_id={node.id} verdict={node.verdict} conf={node.confidence:.2f}",
        f"  verdict_reason: {ev.get('verdict_reason') or node.mechanism or 'n/a'}",
        f"  inconclusive_reason: {node.inconclusive_reason or 'n/a'}",
        f"  failure_signature: {node.failure_signature or 'n/a'}",
        f"  artifact_gate: {_artifact_gate_line(ev)}",
    ]
    if ev.get("lofo_r2") is not None:
        lines.append(f"  lofo_r2={ev.get('lofo_r2')} lofo_gap={ev.get('lofo_gap', 'n/a')}")
    if ev.get("p_value") is not None:
        lines.append(f"  p_value={ev.get('p_value')} metric={ev.get('metric_value', 'n/a')}")
    if include_prose:
        text = (node.text or "").replace("\n", " ")[:280]
        lines.append(f"  hypothesis: {text}")
    return "\n".join(lines)


def compress_node_history(
    tree: HypothesisTree,
    *,
    since_node_ids: set[str] | None = None,
    max_prose_nodes: int = 40,
    char_budget: int = 120_000,
) -> str:
    """
    Template compression: structured fields always full; drop oldest hypothesis prose first.
    """
    nodes = list(tree.nodes.values())
    if since_node_ids:
        nodes = [n for n in nodes if n.id in since_node_ids or n.verdict != "pending"]
    else:
        nodes = [n for n in nodes if n.verdict != "pending"]

    nodes.sort(key=lambda n: (n.generation, n.depth, n.id))
    blocks: list[str] = []
    total = 0
    n = len(nodes)
    for i, node in enumerate(nodes):
        # Keep prose for the most recent max_prose_nodes completed nodes
        include_prose = i >= n - max_prose_nodes
        block = format_node_compact(node, include_prose=include_prose)
        if total + len(block) > char_budget:
            block = format_node_compact(node, include_prose=False)
        if total + len(block) > char_budget:
            break
        blocks.append(block)
        total += len(block) + 1
    return "\n\n".join(blocks) if blocks else "(no completed nodes yet)"


def _lineage_path(tree: HypothesisTree, node: HypothesisNode) -> list[str]:
    path: list[str] = []
    cur: str | None = node.id
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        path.append(cur)
        parent = tree.nodes.get(cur)
        if parent is None:
            break
        cur = parent.parent_id
    path.reverse()
    return path


def _sibling_summary(tree: HypothesisTree, node: HypothesisNode, *, max_siblings: int = 4) -> str:
    if not node.parent_id:
        return "n/a"
    parent = tree.nodes.get(node.parent_id)
    if parent is None:
        return "n/a"
    siblings: list[str] = []
    for sid in parent.children:
        if sid == node.id:
            continue
        sibling = tree.nodes.get(sid)
        if sibling is None:
            continue
        text = (sibling.text or "").replace("\n", " ")[:120]
        siblings.append(f"{sibling.id}:{sibling.verdict}:{text}")
        if len(siblings) >= max_siblings:
            break
    return " | ".join(siblings) if siblings else "none"


def _suggested_expansion(node: HypothesisNode) -> str:
    if node.verdict == "confirmed":
        return "boundary, mechanism, generalization, or downstream utility child"
    if node.verdict == "refuted":
        return "alternative child that changes the variable implicated by refutation"
    if node.verdict == "inconclusive":
        return "better-powered retest or diagnostic child that resolves the failure"
    return "do not expand pending nodes"


def format_expansion_targets(tree: HypothesisTree, *, max_targets: int = 16) -> str:
    """Branch-aware target list for synthesis: every new candidate should choose one."""
    candidates = [
        n for n in tree.nodes.values()
        if n.verdict in ("confirmed", "refuted", "inconclusive")
        and n.depth < 8
        and n.id not in tree.exhausted
        and len(n.children) < 5
    ]
    if not candidates:
        return "(no completed expansion targets yet; bootstrap root hypotheses only if frontier is empty)"

    verdict_rank = {"confirmed": 0, "inconclusive": 1, "refuted": 2}
    candidates.sort(
        key=lambda n: (
            len(n.children),
            verdict_rank.get(n.verdict, 9),
            -int(n.generation or 0),
            -int(n.depth or 0),
        )
    )
    blocks: list[str] = []
    for node in candidates[:max_targets]:
        ev = _parse_evidence_blob(node.evidence_summary)
        path = " > ".join(_lineage_path(tree, node))
        text = (node.text or "").replace("\n", " ")[:260]
        blocks.append(
            "\n".join([
                (
                    f"target_id={node.id} verdict={node.verdict} conf={node.confidence:.2f} "
                    f"depth={node.depth} children={len(node.children)} generation={node.generation}"
                ),
                f"  parent_id: {node.parent_id or 'ROOT'}",
                f"  branch_path: {path}",
                f"  suggested_expansion: {_suggested_expansion(node)}",
                f"  verdict_reason: {ev.get('verdict_reason') or node.mechanism or 'n/a'}",
                f"  inconclusive_reason: {node.inconclusive_reason or 'n/a'}",
                f"  failure_signature: {node.failure_signature or 'n/a'}",
                f"  siblings: {_sibling_summary(tree, node)}",
                f"  hypothesis: {text}",
            ])
        )
    return "\n\n".join(blocks)


def format_pinned_beliefs(belief_state: CampaignBeliefState) -> str:
    parts: list[str] = []
    if belief_state.active_beliefs:
        parts.append("## Active beliefs (≤3, never summarized)")
        for b in belief_state.active_beliefs:
            parts.append(
                f"- [{b.confidence}/{b.status}] {b.statement}\n"
                f"  supporting: {b.supporting_nodes}\n"
                f"  contradicting: {b.contradicting_nodes}"
            )
    if belief_state.closed_beliefs:
        parts.append("## Closed beliefs (do not re-propose)")
        for c in belief_state.closed_beliefs:
            parts.append(f"- {c.statement} — {c.reason}")
    return "\n".join(parts) if parts else "(no beliefs yet)"


def format_confirmed_artifacts(tree: HypothesisTree) -> str:
    lines: list[str] = []
    for nid in tree.confirmed:
        node = tree.nodes.get(nid)
        if node is None:
            continue
        ev = _parse_evidence_blob(node.evidence_summary)
        ag = ev.get("artifact_gate") if isinstance(ev.get("artifact_gate"), dict) else {}
        if ag.get("verdict") == "survived" or node.verdict == "confirmed":
            lines.append(format_node_compact(node, include_prose=True))
    return "\n\n".join(lines) if lines else "(none yet)"



def compose_synthesis_prompt(
    *,
    question: str,
    belief_state: CampaignBeliefState,
    tree: HypothesisTree,
    since_node_ids: set[str] | None = None,
    role_addendum: str | None = None,
    role_text: str | None = None,
    lifetime_context: str = "",
) -> str:
    """Assemble 4-part synthesis prompt (fixes.md §5.2)."""
    role = role_text if role_text is not None else load_prompt("orchestrator_role.md")
    if role_addendum:
        role = f"{role.strip()}\n\n{role_addendum.strip()}"
    task = load_prompt("synthesis_task.md")

    human_block = ""
    if belief_state.human_messages:
        human_block = "## Recent human guidance (verbatim)\n" + "\n".join(
            f"- {m}" for m in belief_state.human_messages[-10:]
        )

    recent = belief_state.recent_activity_summary.strip()
    if recent:
        recent_block = f"## Recent activity / branch goal\n{recent}"
    else:
        recent_block = "## Recent activity / branch goal\n(initial synthesis — explore the research question)"

    pinned = "\n\n".join([
        f"## Research question (verbatim)\n{question}",
        human_block,
        format_pinned_beliefs(belief_state),
        f"## Artifact-verified confirmed findings\n{format_confirmed_artifacts(tree)}",
        f"## Open expansion targets (choose parent_id from target_id)\n{format_expansion_targets(tree)}",
        f"## Completed nodes (structured fields pinned; prose compressed by recency)\n"
        f"{compress_node_history(tree, since_node_ids=since_node_ids)}",
        recent_block,
    ])
    if lifetime_context and lifetime_context.strip():
        pinned = f"{pinned}\n\n## Cross-campaign numerical baseline (use for Q1 threshold sweeps)\n{lifetime_context.strip()}"

    return f"{role.strip()}\n\n{pinned.strip()}\n\n{task.strip()}\n"
