"""Post-hoc fabrication audit for AstaBench × Propab runs."""
from __future__ import annotations

from typing import Any

from propab.evidence_binding import binding_check_statement_to_node


def audit_campaign_answer(
    campaign_payload: dict[str, Any],
    answer: dict[str, str],
    *,
    tree_nodes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check whether submitted hypothesis has binding-supported citations in-tree."""
    campaign = campaign_payload.get("campaign") or campaign_payload
    beliefs = (campaign.get("belief_state") or {}).get("active_beliefs") or []
    hypothesis = answer.get("hypothesis") or ""
    nodes = tree_nodes or (campaign.get("hypothesis_tree") or {}).get("nodes") or {}

    matched_belief = None
    for b in beliefs:
        if not isinstance(b, dict):
            continue
        stmt = str(b.get("statement") or "")
        if stmt and (stmt[:80] in hypothesis or hypothesis[:80] in stmt):
            matched_belief = b
            break

    citations_audited = 0
    mismatched = 0
    examples: list[dict[str, str]] = []

    if matched_belief:
        for nid in matched_belief.get("supporting_nodes") or []:
            node = nodes.get(nid) if isinstance(nodes, dict) else None
            if not isinstance(node, dict):
                continue
            citations_audited += 1
            result = binding_check_statement_to_node(hypothesis, node)
            if not result.match:
                mismatched += 1
                if len(examples) < 5:
                    examples.append({"node_id": str(nid), "reason": result.reason or "mismatch"})

    return {
        "hypothesis": hypothesis[:300],
        "matched_belief": bool(matched_belief),
        "citations_audited": citations_audited,
        "citations_mismatched": mismatched,
        "mismatch_fraction": (mismatched / citations_audited) if citations_audited else None,
        "clean": citations_audited == 0 or mismatched == 0,
        "examples": examples,
    }
