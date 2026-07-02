#!/usr/bin/env python3
"""
Artifact-verification audit for final synthesis rival beliefs (fixes.md step 3).

Beliefs are not confirmed findings — this audits the evidence behind each belief
(supporting tree nodes + the belief statement as a meta-claim).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.artifact_verification import (  # noqa: E402
    audit_confirmed_row,
    evidence_context_from_hypothesis,
    parse_evidence_summary,
    run_artifact_gate,
)
from propab.campaign_resume import belief_state_from_synthesis_events  # noqa: E402

DEFAULT_CID = "faaf394b-7f95-4778-9136-e922f2401e7f"
DEFAULT_OUT = ROOT / "artifacts" / "mandrake_final_beliefs_artifact_audit.json"


def _get(url: str) -> Any:
    with urlopen(url, timeout=120) as r:
        return json.load(r)


def _node_row(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": node.get("id"),
        "text": node.get("text"),
        "verdict": node.get("verdict"),
        "confidence": node.get("confidence"),
        "evidence_summary": node.get("evidence_summary"),
        "failure_signature": node.get("failure_signature"),
        "primary_theme": node.get("primary_theme"),
    }


def _resolve_node_id(prefix: str, nodes: dict[str, dict]) -> str | None:
    p = (prefix or "").strip().lower()
    if not p:
        return None
    if p in nodes:
        return p
    for nid in nodes:
        if nid.lower().startswith(p) or p in nid.lower():
            return nid
    return None


def _beliefs_from_analysis(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    evo = blob.get("belief_evolution") or {}
    beliefs = evo.get("final_beliefs")
    return beliefs if isinstance(beliefs, list) and beliefs else None


def _resolve_belief_nodes(belief: dict[str, Any], nodes: dict[str, dict]) -> dict[str, Any]:
    out = dict(belief)
    out["supporting_nodes"] = [
        rid for p in (belief.get("supporting_nodes") or [])
        if (rid := _resolve_node_id(str(p), nodes))
    ]
    out["contradicting_nodes"] = [
        rid for p in (belief.get("contradicting_nodes") or [])
        if (rid := _resolve_node_id(str(p), nodes))
    ]
    return out


def audit_belief(
    belief: dict[str, Any],
    nodes: dict[str, dict],
    *,
    question: str,
) -> dict[str, Any]:
    stmt = str(belief.get("statement") or "")
    supporting = [str(x) for x in (belief.get("supporting_nodes") or [])]
    contradicting = [str(x) for x in (belief.get("contradicting_nodes") or [])]

    support_rows = [_node_row(nodes[nid]) for nid in supporting if nid in nodes]
    contra_rows = [_node_row(nodes[nid]) for nid in contradicting if nid in nodes]

    support_gates = [audit_confirmed_row(r) for r in support_rows]
    contra_gates = [audit_confirmed_row(r) for r in contra_rows]

    # Meta-claim: belief statement with pooled LOFO stats from supporting evidence
    pooled: dict[str, Any] = {}
    lofo_vals: list[float] = []
    gaps: list[float] = []
    for r in support_rows:
        ev = parse_evidence_summary(str(r.get("evidence_summary") or ""))
        if ev.get("lofo_r2") is not None:
            lofo_vals.append(float(ev["lofo_r2"]))
        if ev.get("lofo_gap") is not None:
            gaps.append(float(ev["lofo_gap"]))
        if ev.get("mean_r2") is not None and not lofo_vals:
            lofo_vals.append(float(ev["mean_r2"]))
    if lofo_vals:
        pooled["lofo_r2"] = sum(lofo_vals) / len(lofo_vals)
    if gaps:
        pooled["lofo_gap"] = max(gaps)
    pooled["n_families"] = 7
    pooled["methodology"] = "LOFO"

    ctx = evidence_context_from_hypothesis(
        stmt,
        pooled,
        methodology="mandrake_verification",
        domain_bucket="mandrake",
    )
    belief_gate = run_artifact_gate(ctx)

    n_support = len(support_gates)
    n_survived = sum(1 for g in support_gates if g.top_artifact_survived)
    n_refuted_support = sum(1 for g in support_gates if g.verdict == "refuted")

    if n_support == 0:
        trust = "NO_EVIDENCE_LINKED"
    elif n_refuted_support == n_support and belief_gate.verdict == "refuted":
        trust = "INTERNAL_ONLY"
    elif belief_gate.verdict == "refuted" and n_survived == 0:
        trust = "META_AND_SUPPORT_REFUTED"
    elif n_refuted_support == n_support:
        trust = "INTERNAL_ONLY"
    elif n_survived / n_support < 0.5:
        trust = "WEAK_EVIDENCE"
    elif belief_gate.verdict == "confirmed":
        trust = "ARTIFACT_SURVIVING"
    else:
        trust = "MIXED"

    return {
        "statement": stmt,
        "confidence": belief.get("confidence"),
        "status": belief.get("status"),
        "supporting_node_count": n_support,
        "contradicting_node_count": len(contra_rows),
        "support_artifact_survival_rate": round(n_survived / n_support, 3) if n_support else None,
        "supporting_nodes": [
            {
                "node_id": support_rows[i].get("id"),
                "verdict": support_rows[i].get("verdict"),
                "text_snippet": str(support_rows[i].get("text") or "")[:120],
                "gate_verdict": support_gates[i].verdict,
                "top_artifact": (
                    support_gates[i].ranked_artifacts[0].artifact_id
                    if support_gates[i].ranked_artifacts else None
                ),
                "top_artifact_survived": support_gates[i].top_artifact_survived,
            }
            for i in range(n_support)
        ],
        "belief_meta_gate": belief_gate.to_dict(),
        "trust_verdict": trust,
        "interpretation": (
            "Belief surviving synthesis rounds ≠ artifact-verified truth. "
            "Compare trust_verdict to belief confidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CID)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=ROOT / "artifacts" / "mandrake_redesign_campaign_analysis.json",
        help="Post-run analysis with final_beliefs.supporting_nodes (short id prefixes)",
    )
    args = parser.parse_args()

    base = args.api.rstrip("/")
    camp_resp = _get(f"{base}/campaigns/{args.campaign_id}")
    camp = camp_resp.get("campaign") or camp_resp
    question = str(camp.get("question") or "")
    tree = camp.get("hypothesis_tree") or {}
    nodes = tree.get("nodes") or {}

    events = _get(f"{base}/sessions/{args.campaign_id}/events?limit=2000")
    if not isinstance(events, list):
        events = events.get("events", [])

    state = belief_state_from_synthesis_events(events)
    raw_beliefs: list[dict[str, Any]] = []
    if state and state.active_beliefs:
        raw_beliefs = [b.to_dict() for b in state.active_beliefs]
    analysis_beliefs = _beliefs_from_analysis(args.analysis)
    if analysis_beliefs:
        raw_beliefs = analysis_beliefs

    if not raw_beliefs:
        print("No synthesis beliefs found", file=sys.stderr)
        return 1

    belief_audits = [
        audit_belief(_resolve_belief_nodes(b, nodes), nodes, question=question)
        for b in raw_beliefs
    ]

    verdict_counts = {}
    for a in belief_audits:
        verdict_counts[a["trust_verdict"]] = verdict_counts.get(a["trust_verdict"], 0) + 1

    report = {
        "campaign_id": args.campaign_id,
        "question": question,
        "total_hypotheses_tested": camp.get("total_hypotheses"),
        "note_calibration": (
            "292 tested nodes overstates independent evidence: 16 near-duplicate seed pairs "
            "were redundant volume. Beliefs audit whether rivals are artifact-trustworthy."
        ),
        "synthesis_rounds": sum(1 for e in events if (e.get("step") or "") == "campaign.synthesis"),
        "final_beliefs_audited": len(belief_audits),
        "trust_distribution": verdict_counts,
        "beliefs": belief_audits,
        "overall": (
            "PASS redesign search quality does not imply PASS belief truth — "
            "see trust_verdict per belief."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "beliefs": len(belief_audits),
        "trust_distribution": verdict_counts,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
