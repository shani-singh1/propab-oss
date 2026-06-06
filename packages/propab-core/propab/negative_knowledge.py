"""
Phase D — negative knowledge: counterexamples, dead directions, failure signatures.
"""
from __future__ import annotations

from typing import Any

from propab.knowledge_graph import Claim, FailureRecord, KnowledgeGraph, MechanismRecord, new_id
from propab.research_quality import NODE_ROLE_CONTROL, is_discovery_node


def extract_failures_from_campaign(
    campaign_id: str,
    tree_nodes: dict[str, Any],
    *,
    ledger: list[dict[str, Any]] | None = None,
) -> list[FailureRecord]:
    """Build failure records from refuted and wasteful inconclusive nodes."""
    failures: list[FailureRecord] = []
    waste_reasons = frozenset({
        "code_timeout", "sample_budget_exhausted", "metric_missing", "duplicate_evidence",
    })

    for nid, node in tree_nodes.items():
        if isinstance(node, dict):
            role = node.get("node_role") or "DISCOVERY"
            verdict = node.get("verdict")
            text = node.get("text") or ""
            theme = node.get("primary_theme") or node.get("theme_id") or "general"
            reason = node.get("inconclusive_reason") or ""
            sig = node.get("failure_signature")
        else:
            role = getattr(node, "node_role", "DISCOVERY")
            verdict = getattr(node, "verdict", None)
            text = getattr(node, "text", "")
            theme = getattr(node, "primary_theme", None) or getattr(node, "theme_id", None) or "general"
            reason = getattr(node, "inconclusive_reason", None) or ""
            sig = getattr(node, "failure_signature", None)

        if role == NODE_ROLE_CONTROL:
            continue
        if verdict == "refuted":
            failures.append(FailureRecord(
                id=new_id("fail"),
                text=text[:500],
                reason="refuted",
                failure_signature=sig,
                theme=theme,
                verdict="refuted",
                campaign_id=campaign_id,
            ))
        elif verdict == "inconclusive" and reason in waste_reasons:
            failures.append(FailureRecord(
                id=new_id("fail"),
                text=text[:500],
                reason=reason,
                failure_signature=sig,
                theme=theme,
                verdict="inconclusive",
                campaign_id=campaign_id,
            ))
    return failures


def extract_confirmed_claims(
    campaign_id: str,
    ledger: list[dict[str, Any]],
) -> list[Claim]:
    from propab.knowledge_graph import Claim

    claims: list[Claim] = []
    for entry in ledger:
        if entry.get("node_role") == NODE_ROLE_CONTROL:
            continue
        if entry.get("verdict") != "confirmed":
            continue
        claims.append(Claim(
            id=str(entry.get("claim_id") or new_id("claim")),
            text=str(entry.get("claim") or "")[:600],
            verdict="confirmed",
            theme=str(entry.get("primary_theme") or "general"),
            confidence=float(entry.get("confidence") or 0.0),
            replication_level=str(entry.get("replication_level") or "T1"),
            campaign_id=campaign_id,
            claim_type=entry.get("claim_type"),
        ))
    return claims


def merge_failures_into_graph(graph: KnowledgeGraph, failures: list[FailureRecord]) -> int:
    added = 0
    existing_texts = {f.text[:200] for f in graph.failures.values()}
    for f in failures:
        key = f.text[:200]
        if key in existing_texts:
            continue
        graph.add_failure(f)
        existing_texts.add(key)
        added += 1
    return added
