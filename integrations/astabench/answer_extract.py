"""Map Propab campaign state → DiscoveryBench JSON answer."""
from __future__ import annotations

import json
from typing import Any


ABSTAIN_HYPOTHESIS = (
    "Insufficient evidence within the allocated time budget to state a confident "
    "hypothesis supported by verified experiments."
)


def _confirmed_findings(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    tree = campaign.get("hypothesis_tree") or {}
    nodes = tree.get("nodes") or {}
    out: list[dict[str, Any]] = []
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if node.get("verdict") != "confirmed":
            continue
        out.append(
            {
                "id": nid,
                "text": str(node.get("text") or ""),
                "confidence": float(node.get("confidence") or 0.0),
                "key_finding": str(node.get("key_finding") or ""),
            }
        )
    out.sort(key=lambda x: x["confidence"], reverse=True)
    return out


def _belief_state(campaign: dict[str, Any]) -> dict[str, Any]:
    return campaign.get("belief_state") or {}


def _artifact_gate_survivors(campaign: dict[str, Any]) -> list[str]:
    """Node ids whose evidence survived artifact gate (if recorded)."""
    tree = campaign.get("hypothesis_tree") or {}
    nodes = tree.get("nodes") or {}
    survivors: list[str] = []
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if node.get("verdict") != "confirmed":
            continue
        ev = node.get("evidence") or {}
        if isinstance(ev, str):
            try:
                ev = json.loads(ev)
            except json.JSONDecodeError:
                ev = {}
        if not isinstance(ev, dict):
            ev = {}
        gate = ev.get("artifact_gate") or {}
        if isinstance(gate, dict) and gate.get("verdict") == "confirmed":
            survivors.append(str(nid))
    return survivors


def extract_discoverybench_answer(
    campaign_payload: dict[str, Any],
    *,
    abstain_if_not_strong: bool = True,
    require_confirmed: bool = True,
) -> tuple[dict[str, str], dict[str, Any]]:
    """
    Return (answer_dict, audit_meta).

    answer_dict has keys hypothesis, workflow for DiscoveryBench scorer.

    Policy (fixes.md Step 2 / architecture spec):
    - Prefer confirmed hypothesis-tree findings.
    - If ``require_confirmed`` (default), abstain when there are zero confirmed
      nodes — do not submit strong beliefs without verified experiments.
    - Legacy ``abstain_if_not_strong=False`` allows weak beliefs only when there
      are confirmed nodes but none ranked first (rare); still never submits
      ungrounded strong beliefs when ``require_confirmed`` is True.
    """
    campaign = campaign_payload.get("campaign") or campaign_payload
    beliefs = _belief_state(campaign)
    active = beliefs.get("active_beliefs") or []
    strong = [b for b in active if isinstance(b, dict) and b.get("confidence") == "strong"]
    weak = [b for b in active if isinstance(b, dict) and b.get("confidence") == "weak"]
    confirmed = _confirmed_findings(campaign)
    gate_survivors = _artifact_gate_survivors(campaign)
    summary = campaign_payload.get("summary") or {}
    activity = str(beliefs.get("recent_activity_summary") or "")

    audit: dict[str, Any] = {
        "strong_beliefs": len(strong),
        "weak_beliefs": len(weak),
        "confirmed_nodes": len(confirmed),
        "stop_reason": campaign.get("stop_reason"),
        "status": campaign.get("status"),
        "source": None,
        "artifact_gate_survivors": gate_survivors,
        "wait_timeout": bool(campaign_payload.get("_wait_timeout")),
        "non_terminal_at_timeout": bool(campaign_payload.get("_non_terminal_at_timeout")),
    }

    hypothesis = ""
    workflow_parts: list[str] = []

    if confirmed:
        best = confirmed[0]
        hypothesis = (best.get("key_finding") or best.get("text") or "").strip()
        audit["source"] = "confirmed_finding"
        audit["belief_confidence"] = None
        audit["supporting_nodes"] = [best.get("id")]
        workflow_parts.append(f"Confirmed experiment: {best.get('text', '')[:500]}")
    elif require_confirmed:
        hypothesis = ABSTAIN_HYPOTHESIS
        audit["source"] = "abstain"
        audit["belief_confidence"] = None
        audit["supporting_nodes"] = []
        if strong:
            audit["abstain_reason"] = "confirmed_nodes_zero_despite_strong_beliefs"
            workflow_parts.append(
                f"Strong belief not submitted (no confirmed experiments): "
                f"{str(strong[0].get('statement') or '')[:300]}"
            )
    elif not abstain_if_not_strong and weak:
        hypothesis = str(weak[0].get("statement") or "").strip()
        audit["source"] = "weak_belief"
        audit["belief_confidence"] = "weak"
        audit["supporting_nodes"] = list(weak[0].get("supporting_nodes") or [])
        workflow_parts.append(f"Belief synthesis (weak): {hypothesis}")
    else:
        hypothesis = ABSTAIN_HYPOTHESIS
        audit["source"] = "abstain"
        audit["belief_confidence"] = None
        audit["supporting_nodes"] = []

    if activity:
        workflow_parts.append(f"Campaign activity: {activity[:1500]}")
    workflow_parts.append(
        f"Tested {summary.get('total_hypotheses', campaign.get('total_hypotheses', '?'))} hypotheses; "
        f"confirmed {summary.get('total_confirmed', campaign.get('total_confirmed', 0))}; "
        f"stop={campaign.get('stop_reason') or summary.get('stop_reason')}."
    )
    if confirmed:
        workflow_parts.append(
            "Key confirmed results: "
            + "; ".join(
                (c.get("key_finding") or c.get("text", ""))[:200]
                for c in confirmed[:3]
            )
        )

    answer = {
        "hypothesis": hypothesis,
        "workflow": "\n".join(workflow_parts).strip(),
    }
    return answer, audit


def format_completion(answer: dict[str, str]) -> str:
    return json.dumps(answer, ensure_ascii=False)
