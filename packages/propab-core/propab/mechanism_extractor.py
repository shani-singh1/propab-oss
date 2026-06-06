"""
Phase B — extract causal mechanisms from findings (not verdict metadata).
"""
from __future__ import annotations

from typing import Any

from propab.knowledge_graph import MechanismRecord, new_id
from propab.research_quality import build_mechanism_object, build_refutation_mechanism


def extract_mechanism_from_finding(
    finding: dict[str, Any],
    *,
    campaign_id: str | None = None,
) -> MechanismRecord | None:
    claim = str(finding.get("claim") or "")
    verdict = str(finding.get("verdict") or "confirmed")
    mechs = finding.get("mechanisms") or []
    evidence_obj: dict[str, Any] = {}
    if mechs and isinstance(mechs[0], dict):
        m0 = mechs[0]
        if verdict == "refuted":
            obj = build_refutation_mechanism(
                claim=claim,
                evidence=evidence_obj,
                verdict_reason=str(m0.get("effect") or m0.get("mechanism") or ""),
            )
        else:
            obj = m0 if "cause" in m0 else build_mechanism_object(
                claim=claim,
                mechanism=m0.get("mechanism"),
                evidence=evidence_obj,
                verdict=verdict,
            )
    else:
        obj = build_mechanism_object(claim=claim, mechanism=None, evidence=evidence_obj, verdict=verdict)
    if not obj:
        return None
    return MechanismRecord(
        id=new_id("mech"),
        claim_id=str(finding.get("claim_id") or ""),
        cause=str(obj.get("cause") or claim)[:500],
        effect=str(obj.get("effect") or "")[:600],
        conditions=str(obj.get("conditions") or "")[:400],
        failure_modes=list(obj.get("failure_modes") or [])[:6],
        evidence=list(obj.get("evidence") or [])[:8],
        campaign_id=campaign_id,
    )
