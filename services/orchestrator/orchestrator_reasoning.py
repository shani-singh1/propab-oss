"""Orchestrator reasoning step (C3b) — strategic action selection over the tree.

This is the *strategy* brain of the orchestrator-as-agent redesign. Given a node
whose honesty VERDICT was ALREADY decided deterministically (C2 —
``verdict_pipeline.compute_authoritative_verdict``), an LLM reasons over the full
tree context and picks ONE strategic ACTION for that node:

    deepen         — write a NARROWER child hypothesis (convergence).
    retune         — re-run the SAME hypothesis with changed params/data
                     (a bounded node *attempt*, §3.6; NOT a new child).
    spawn_related  — write a LATERAL, related hypothesis (breadth).
    drop           — close the node; no further exploration from here.

CRITICAL SAFETY INVARIANTS
--------------------------
* The reasoning step NEVER decides confirmed / refuted / inconclusive and never
  touches the honesty gates. It only chooses what to TEST NEXT for an
  already-judged node. The deterministic verdict remains authoritative.
* Everything here is inert unless ``settings.orchestrator_reasoning_enabled`` is
  True. The wiring in ``campaign_loop`` is flag-gated; with the flag OFF (the
  default) none of this module runs and the mechanical frontier/synthesis path is
  byte-for-byte unchanged.
* Domain-independence: the prompt carries only the campaign question and the
  tree-derived context (node text, verdict, evidence, lineage). No domain
  specifics are hard-coded here — anything domain-specific arrives via the
  hypothesis text / plugin-produced evidence.

The pure pieces (``tree_context_for_reasoning``, ``build_reasoning_prompt``,
``parse_reasoning_decision``, ``apply_reasoning_decision``) are unit-tested; the
single async LLM call (``orchestrator_reason_next``) is failure-isolated.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from propab.hypothesis_tree import HypothesisNode, HypothesisTree

logger = logging.getLogger(__name__)

# The four strategic actions the reasoning step may return. Anything else parses
# to ``drop`` (the safe close-the-node fallback).
ACTION_DEEPEN = "deepen"
ACTION_RETUNE = "retune"
ACTION_SPAWN_RELATED = "spawn_related"
ACTION_DROP = "drop"
_ALLOWED_ACTIONS = frozenset({ACTION_DEEPEN, ACTION_RETUNE, ACTION_SPAWN_RELATED, ACTION_DROP})

# Free-form LLM synonyms → canonical action. Keeps parsing robust to phrasing.
_ACTION_ALIASES: dict[str, str] = {
    "deepen": ACTION_DEEPEN,
    "narrow": ACTION_DEEPEN,
    "refine": ACTION_DEEPEN,
    "boundary": ACTION_DEEPEN,
    "mechanistic": ACTION_DEEPEN,
    "child": ACTION_DEEPEN,
    "retune": ACTION_RETUNE,
    "retry": ACTION_RETUNE,
    "rerun": ACTION_RETUNE,
    "re-run": ACTION_RETUNE,
    "retest": ACTION_RETUNE,
    "spawn_related": ACTION_SPAWN_RELATED,
    "spawn": ACTION_SPAWN_RELATED,
    "spawn-related": ACTION_SPAWN_RELATED,
    "lateral": ACTION_SPAWN_RELATED,
    "related": ACTION_SPAWN_RELATED,
    "alternative": ACTION_SPAWN_RELATED,
    "generalize": ACTION_SPAWN_RELATED,
    "drop": ACTION_DROP,
    "close": ACTION_DROP,
    "abandon": ACTION_DROP,
    "stop": ACTION_DROP,
    "exhaust": ACTION_DROP,
    "prune": ACTION_DROP,
}


# ── Decision / outcome contracts ──────────────────────────────────────────────

@dataclass
class ReasoningDecision:
    """One structured strategic decision returned by the reasoning LLM.

    ``action`` is always one of the four canonical actions. ``child_hypothesis_text``
    is the new node text for deepen/spawn_related; ``retune_changes`` describes the
    param/data change for a retune. ``parse_error`` is True when the LLM output was
    missing/malformed and this is a safe fallback — the wiring must NOT mutate the
    tree on a parse error.
    """

    action: str
    rationale: str = ""
    child_hypothesis_text: str | None = None
    retune_changes: str | None = None
    parse_error: bool = False
    raw: str | None = None


@dataclass
class ReasoningOutcome:
    """What ``apply_reasoning_decision`` actually did to the tree.

    ``action`` is the ACTUAL action taken, which can differ from the requested one
    (e.g. a deepen at max depth or a retune past its budget becomes a ``drop``).
    ``new_nodes`` are the freshly-created child/lateral nodes (empty for retune/drop).
    """

    action: str
    requested_action: str
    new_nodes: list[HypothesisNode] = field(default_factory=list)
    retuned: bool = False
    dropped: bool = False
    note: str = ""


# ── Defensive JSON extraction (mirrors think_act._extract_json) ────────────────

def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object from raw LLM text.

    Tries a direct parse, then the first ``{...}`` span. Returns None when nothing
    parseable is present.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _norm_action(raw_action: Any) -> str:
    """Map a free-form action string to a canonical action, defaulting to drop."""
    s = str(raw_action or "").strip().lower().replace(" ", "_")
    if s in _ALLOWED_ACTIONS:
        return s
    return _ACTION_ALIASES.get(s, _ACTION_ALIASES.get(s.replace("_", "-"), ACTION_DROP))


def parse_reasoning_decision(raw: str) -> ReasoningDecision:
    """Parse raw LLM output into a ``ReasoningDecision`` — defensively.

    A missing/unparseable object yields ``action=drop`` with ``parse_error=True`` so
    the caller can treat it as "no strategy available" and leave the tree untouched.
    Any recognizable but off-vocabulary action normalizes to ``drop`` (parse_error
    stays False — the model spoke, it just said "close this line").
    """
    data = _extract_json(raw)
    if data is None:
        return ReasoningDecision(
            action=ACTION_DROP,
            rationale="could not parse reasoning output",
            parse_error=True,
            raw=raw,
        )
    action = _norm_action(data.get("action"))
    rationale = str(data.get("rationale") or data.get("reason") or "").strip()
    child = data.get("child_hypothesis_text") or data.get("child_hypothesis") or data.get("child")
    child_text = str(child).strip() if child not in (None, "") else None
    changes = data.get("retune_changes") or data.get("changes") or data.get("retune")
    changes_text = str(changes).strip() if changes not in (None, "") else None
    return ReasoningDecision(
        action=action,
        rationale=rationale,
        child_hypothesis_text=child_text,
        retune_changes=changes_text,
        parse_error=False,
        raw=raw,
    )


# ── Pure tree-context builder ──────────────────────────────────────────────────

def _node_brief(node: HypothesisNode | None, *, limit: int = 240) -> dict[str, Any] | None:
    """Compact, JSON-safe view of a node for the reasoning prompt."""
    if node is None:
        return None
    text = " ".join(str(node.text or "").split())
    return {
        "node_id": str(node.id),
        "text": text[:limit],
        "verdict": str(node.verdict or "pending"),
        "confidence": round(float(node.confidence or 0.0), 3),
        "depth": int(node.depth or 0),
        "expansion_type": node.expansion_type,
    }


def tree_context_for_reasoning(
    tree: HypothesisTree,
    node: HypothesisNode,
    *,
    question: str = "",
    max_siblings: int = 5,
    max_confirmed: int = 8,
) -> dict[str, Any]:
    """Pure: assemble the tree context the reasoning step reasons over.

    Returns a JSON-serializable dict with this node (verdict + evidence summary +
    attempts), its parent, its siblings, a summary of the confirmed set, the
    question, and a compact "what's working" summary (dominant confirmed themes).
    No LLM, no I/O — trivially unit-testable.
    """
    from propab.research_quality import is_discovery_node

    this_node = _node_brief(node) or {"node_id": str(getattr(node, "id", "")), "text": ""}
    this_node["evidence_summary"] = " ".join(str(node.evidence_summary or "").split())[:400]
    this_node["attempts"] = len(node.attempts or [])
    this_node["node_role"] = getattr(node, "node_role", None)

    parent = tree.nodes.get(node.parent_id) if node.parent_id else None
    parent_brief = _node_brief(parent)

    siblings: list[dict[str, Any]] = []
    if parent is not None:
        for cid in parent.children:
            if cid == node.id or cid not in tree.nodes:
                continue
            sb = _node_brief(tree.nodes[cid])
            if sb is not None:
                siblings.append(sb)
            if len(siblings) >= max_siblings:
                break

    confirmed_nodes = [
        tree.nodes[nid]
        for nid in tree.confirmed
        if nid in tree.nodes and is_discovery_node(tree.nodes[nid])
    ]
    confirmed_claims = [" ".join(str(n.text or "").split())[:200] for n in confirmed_nodes[:max_confirmed]]

    # "What's working" = the dominant confirmed themes (breadth of real signal).
    theme_hits: dict[str, int] = {}
    for n in confirmed_nodes:
        tid = n.primary_theme or n.theme_id or "general"
        theme_hits[tid] = theme_hits.get(tid, 0) + 1
    top_themes = sorted(theme_hits.items(), key=lambda kv: kv[1], reverse=True)[:5]
    if top_themes:
        whats_working = "confirmed themes so far: " + ", ".join(f"{t} (x{c})" for t, c in top_themes)
    else:
        whats_working = "no confirmed findings yet"

    return {
        "question": " ".join(str(question or "").split())[:400],
        "this_node": this_node,
        "parent": parent_brief,
        "siblings": siblings,
        "confirmed": {"count": len(confirmed_nodes), "claims": confirmed_claims},
        "whats_working": whats_working,
        "frontier_size": len(tree.frontier),
    }


# ── Pure prompt builder ────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are the ORCHESTRATOR of a scientific discovery campaign. A worker just finished \
testing ONE hypothesis. Its honesty VERDICT has ALREADY been decided by a deterministic \
pipeline and is FINAL — you do NOT re-judge it. Your job is to decide the single best \
STRATEGIC next move for this line of inquiry, given the whole tree so far.

Research question:
{question}

This hypothesis (verdict is FINAL — do not change it):
- text: {node_text}
- deterministic verdict: {verdict} (confidence {confidence:.2f})
- evidence summary: {evidence_summary}
- retune attempts already used: {attempts_used} of {max_retune}
- depth in tree: {depth}

Parent hypothesis: {parent_text}
Sibling hypotheses under the same parent:
{siblings_block}

What is working across the campaign: {whats_working}
Confirmed findings so far ({confirmed_count}):
{confirmed_block}

Choose ONE action:
- "deepen": write a NARROWER child hypothesis that converges on WHY / boundary / \
mechanism of this result. Best when the verdict is confirmed and there is more to pin down.
- "retune": re-run the SAME hypothesis with changed params/data (more power, tighter \
controls, a better null). Best when the result was inconclusive and a fixable setup issue \
plausibly caused it. Only if attempts remain.
- "spawn_related": write a LATERAL, related-but-different hypothesis. Best when this line \
is refuted or saturated but an adjacent idea is promising.
- "drop": close this line. Best when it is a dead end and nothing nearby is worth testing.

Return ONLY a JSON object, no prose:
{{
  "action": "deepen|retune|spawn_related|drop",
  "rationale": "one sentence, plain language, why this move",
  "child_hypothesis_text": "required for deepen/spawn_related — the new hypothesis, one sentence",
  "retune_changes": "required for retune — what to change in params/data"
}}
"""


def _siblings_block(siblings: list[dict[str, Any]]) -> str:
    if not siblings:
        return "  (none)"
    return "\n".join(
        f"  - [{s.get('verdict', 'pending')}] {s.get('text', '')}" for s in siblings
    )


def _confirmed_block(claims: list[str]) -> str:
    if not claims:
        return "  (none)"
    return "\n".join(f"  - {c}" for c in claims)


def build_reasoning_prompt(
    context: dict[str, Any],
    *,
    verdict: str,
    confidence: float,
    max_retune_rounds: int,
) -> str:
    """Pure: render the reasoning prompt from a tree context + the FINAL verdict."""
    this_node = context.get("this_node") or {}
    parent = context.get("parent") or None
    confirmed = context.get("confirmed") or {}
    return _PROMPT_TEMPLATE.format(
        question=context.get("question") or "(unspecified)",
        node_text=this_node.get("text") or "(no text)",
        verdict=str(verdict or "pending"),
        confidence=float(confidence or 0.0),
        evidence_summary=this_node.get("evidence_summary") or "(no evidence summary)",
        attempts_used=int(this_node.get("attempts") or 0),
        max_retune=int(max_retune_rounds),
        depth=int(this_node.get("depth") or 0),
        parent_text=(parent.get("text") if parent else "(this is a seed hypothesis)"),
        siblings_block=_siblings_block(context.get("siblings") or []),
        whats_working=context.get("whats_working") or "n/a",
        confirmed_count=int(confirmed.get("count") or 0),
        confirmed_block=_confirmed_block(confirmed.get("claims") or []),
    )


# ── Pure tree mutation ─────────────────────────────────────────────────────────

def _build_child_node(
    tree: HypothesisTree,
    parent: HypothesisNode,
    text: str,
    *,
    generation: int,
    expansion_type: str,
) -> HypothesisNode:
    """Create a child ``HypothesisNode`` under ``parent`` (same construction path as
    ``add_seeds`` / ``parse_expanded_nodes`` — tree-owned UUID, theme + role inferred)."""
    from propab.research_quality import extract_theme_vector, infer_node_role

    primary, secondary, theme_conf = extract_theme_vector(text)
    return HypothesisNode(
        id=str(uuid4()),
        text=text,
        parent_id=parent.id,
        depth=parent.depth + 1,
        verdict="pending",
        generation=generation,
        expansion_type=expansion_type,
        node_role=infer_node_role(text),
        primary_theme=primary,
        secondary_themes=secondary,
        theme_id=primary,
        theme_confidence=theme_conf,
        lineage_length=tree.lineage_length(parent.id) + 1,
    )


def _exhaust(tree: HypothesisTree, node: HypothesisNode) -> None:
    """Close a node: drop it from the frontier and mark it exhausted (no re-expansion)."""
    if node.id in tree.frontier:
        tree.frontier.remove(node.id)
    if node.id not in tree.exhausted:
        tree.exhausted.append(node.id)


def apply_reasoning_decision(
    tree: HypothesisTree,
    node: HypothesisNode,
    decision: ReasoningDecision,
    *,
    generation: int,
    max_retune_rounds: int,
    max_depth: int = 8,
) -> ReasoningOutcome:
    """Pure: mutate the tree to enact a reasoning ``decision``. Never calls an LLM.

    - deepen        → new NARROWER child of ``node`` (expansion_type ``boundary``).
    - spawn_related → new LATERAL node (sibling of ``node`` when it has a parent, else
                      a child of ``node``), expansion_type ``alternative``.
    - retune        → record a bounded node ATTEMPT and requeue ``node`` as pending; at
                      the attempt budget it degrades to ``drop``.
    - drop          → close ``node`` (exhaust).

    Depth-exceeding deepen/spawn also degrade to ``drop``. The deterministic verdict on
    ``node`` is only reset for a retune (the node is genuinely re-run); it is never
    otherwise altered.
    """
    requested = decision.action

    if requested == ACTION_DEEPEN:
        text = (decision.child_hypothesis_text or "").strip()
        if not text:
            return ReasoningOutcome(action="noop", requested_action=requested, note="no_child_text")
        if node.depth + 1 > max_depth:
            _exhaust(tree, node)
            return ReasoningOutcome(action=ACTION_DROP, requested_action=requested,
                                    dropped=True, note="max_depth_reached")
        child = _build_child_node(tree, node, text, generation=generation, expansion_type="boundary")
        tree.add_to_frontier([child])
        return ReasoningOutcome(action=ACTION_DEEPEN, requested_action=requested, new_nodes=[child])

    if requested == ACTION_SPAWN_RELATED:
        text = (decision.child_hypothesis_text or "").strip()
        if not text:
            return ReasoningOutcome(action="noop", requested_action=requested, note="no_child_text")
        # A lateral attaches to the node's PARENT (a true sibling) when one exists,
        # so breadth does not artificially deepen the tree; otherwise it hangs off
        # the seed node itself.
        anchor = tree.nodes.get(node.parent_id) if node.parent_id else None
        anchor = anchor if anchor is not None else node
        if anchor.depth + 1 > max_depth:
            _exhaust(tree, node)
            return ReasoningOutcome(action=ACTION_DROP, requested_action=requested,
                                    dropped=True, note="max_depth_reached")
        child = _build_child_node(tree, anchor, text, generation=generation, expansion_type="alternative")
        tree.add_to_frontier([child])
        return ReasoningOutcome(action=ACTION_SPAWN_RELATED, requested_action=requested, new_nodes=[child])

    if requested == ACTION_RETUNE:
        if len(node.attempts or []) >= max_retune_rounds:
            _exhaust(tree, node)
            return ReasoningOutcome(action=ACTION_DROP, requested_action=requested,
                                    dropped=True, note="retune_budget_exhausted")
        round_no = len(node.attempts or []) + 1
        node.attempts.append({
            "round": round_no,
            "changes": decision.retune_changes or "",
            "rationale": decision.rationale or "",
        })
        # Re-run the SAME hypothesis: reset to pending and requeue on the frontier.
        node.verdict = "pending"
        node.expansion_type = "retest"
        if node.id in tree.confirmed:
            tree.confirmed.remove(node.id)
        if node.id in tree.exhausted:
            tree.exhausted.remove(node.id)
        if node.id not in tree.frontier:
            tree.frontier.append(node.id)
        # Carry the change hint into the methodology so the re-dispatch actually differs.
        if decision.retune_changes:
            base = (node.test_methodology or "").strip()
            note = f"[retune {round_no}: {decision.retune_changes}]"
            node.test_methodology = f"{base} {note}".strip()
        return ReasoningOutcome(action=ACTION_RETUNE, requested_action=requested, retuned=True)

    # drop (and any unrecognized action normalized to drop upstream).
    _exhaust(tree, node)
    return ReasoningOutcome(action=ACTION_DROP, requested_action=requested, dropped=True,
                            note=decision.rationale or "")


# ── Async LLM reasoning step ───────────────────────────────────────────────────

async def orchestrator_reason_next(
    *,
    campaign: Any,
    node: HypothesisNode,
    evidence: dict[str, Any] | None,
    verdict: str,
    confidence: float,
    llm: Any,
    max_retune_rounds: int,
    hypothesis_id: str | None = None,
) -> ReasoningDecision:
    """One LLM call: given tree context + the FINAL deterministic verdict, decide the
    strategic next action for ``node``. Failure-isolated — any exception returns a
    safe ``drop`` decision with ``parse_error=True`` so the campaign loop never crashes
    and the wiring leaves the tree untouched.
    """
    try:
        context = tree_context_for_reasoning(
            campaign.hypothesis_tree, node, question=getattr(campaign, "question", "") or "",
        )
        # Surface the parsed evidence as the node's evidence summary if the tree
        # copy is thin (the loop has the richer parsed object).
        if evidence and not context["this_node"].get("evidence_summary"):
            context["this_node"]["evidence_summary"] = " ".join(str(evidence).split())[:400]
        prompt = build_reasoning_prompt(
            context, verdict=verdict, confidence=confidence, max_retune_rounds=max_retune_rounds,
        )
        raw = await llm.call(
            prompt=prompt,
            purpose="orchestrator.reason_next",
            session_id=campaign.id,
            hypothesis_id=hypothesis_id,
        )
        return parse_reasoning_decision(raw)
    except Exception as exc:  # noqa: BLE001 — reasoning must never break the campaign loop
        logger.warning("[campaign %s] orchestrator reasoning step failed: %s",
                       getattr(campaign, "id", "?"), exc)
        return ReasoningDecision(
            action=ACTION_DROP,
            rationale=f"reasoning step failed: {exc}",
            parse_error=True,
        )
