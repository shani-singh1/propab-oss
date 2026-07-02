#!/usr/bin/env python3
"""
Post-campaign deliverables (fixes.md P1–P5):
  artifacts/campaign_scope_metrics.json
  artifacts/scope_integrity_audit.json
  artifacts/human_review.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.scoped_claim import (
    classify_manual_audit,
    extract_executed_ood_from_experiment,
    parse_scope_from_methodology,
    parse_scope_from_text,
)


def _parse_evidence(summary: str) -> dict[str, Any]:
    if not summary:
        return {}
    m = re.search(r"evidence=(\{.*?\});", summary)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


def _fetch_campaign(conn, campaign_id: str) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, question, status, total_hypotheses, total_confirmed,
                   hypothesis_tree_json
            FROM research_campaigns WHERE id = %s
            """,
            (campaign_id,),
        )
        row = cur.fetchone()
        if not row:
            raise SystemExit(f"Campaign {campaign_id} not found")
        cols = [d[0] for d in cur.description]
        camp = dict(zip(cols, row, strict=True))
        tree = camp.get("hypothesis_tree_json")
        if isinstance(tree, str):
            camp["hypothesis_tree_json"] = json.loads(tree)

        cur.execute(
            """
            SELECT id::text, text, verdict, confidence, evidence_summary, key_finding
            FROM hypotheses WHERE session_id = %s ORDER BY created_at
            """,
            (campaign_id,),
        )
        hyps = [
            dict(zip([d[0] for d in cur.description], r, strict=True))
            for r in cur.fetchall()
        ]

        cur.execute(
            """
            SELECT payload_json FROM events
            WHERE session_id = %s AND step = 'hypothesis.scope_gate'
            ORDER BY created_at
            """,
            (campaign_id,),
        )
        scope_rows = cur.fetchall()
        scope_gate_event: dict[str, Any] = {}
        if scope_rows:
            from collections import Counter

            total_gen = 0
            total_rej = 0
            total_pass = 0
            reasons: Counter[str] = Counter()
            for row in scope_rows:
                p = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
                g = int(p.get("n_generated") or p.get("n_children_generated") or 0)
                r = int(p.get("n_scope_rejected") or p.get("n_children_rejected") or 0)
                total_gen += g
                total_rej += r
                total_pass += int(p.get("n_scope_passed") or p.get("n_children_passed") or max(0, g - r))
                for k, v in (p.get("rejection_reasons") or {}).items():
                    reasons[k] += int(v)
            scope_gate_event = {
                "n_generated": total_gen,
                "n_scope_rejected": total_rej,
                "n_scope_passed": total_pass,
                "scope_rejection_rate": round(total_rej / max(1, total_gen), 4),
                "rejection_reasons": dict(reasons),
                "n_events": len(scope_rows),
            }

        cur.execute(
            """
            SELECT h.id::text, e.output_json, e.input_json, e.step_type
            FROM experiment_steps e
            JOIN hypotheses h ON h.id = e.hypothesis_id
            WHERE h.session_id = %s AND e.step_type IN ('tool_call', 'code_exec')
            ORDER BY e.created_at
            """,
            (campaign_id,),
        )
        steps = []
        for r in cur.fetchall():
            inp = r[2] if isinstance(r[2], dict) else (json.loads(r[2]) if r[2] else {})
            out = r[1] if isinstance(r[1], dict) else (json.loads(r[1]) if r[1] else {})
            code = inp.get("code", "") if isinstance(inp, dict) else ""
            parsed = out.get("parsed", out) if isinstance(out, dict) else out
            steps.append({
                "hypothesis_id": r[0],
                "result_output": parsed,
                "source_code": code,
                "step_type": r[3],
            })
    return {"campaign": camp, "hypotheses": hyps, "scope_gate_event": scope_gate_event, "steps": steps}


def _map_audit_class(audit_class: str) -> str:
    if audit_class in ("fake_ood", "boilerplate_scope"):
        return "fake_scope"
    if audit_class == "mismatched_scope":
        return "mismatched_scope"
    return "valid_scope"


def _gate_failures(hyps: list[dict]) -> dict[str, int]:
    n_ood_failed = 0
    n_artifact_failed = 0
    for h in hyps:
        ev = _parse_evidence(str(h.get("evidence_summary") or ""))
        if not ev:
            continue
        if ev.get("ood_passed") is False:
            n_ood_failed += 1
        elif "OOD gate:" in str(ev.get("verdict_reason") or ""):
            n_ood_failed += 1
        ag = ev.get("artifact_gate")
        if isinstance(ag, dict) and ag.get("verdict") not in (None, "confirmed"):
            n_artifact_failed += 1
        elif ev.get("top_artifact_survived") is False:
            n_artifact_failed += 1
    return {"n_ood_failed": n_ood_failed, "n_artifact_failed": n_artifact_failed}


def _interpretation(
    scope_rejection_rate: float,
    n_survivors: int,
    human_believable: int,
    n_artifact_failed: int,
    n_tested: int,
) -> str:
    if scope_rejection_rate == 0:
        return "Case C — scope rejection = 0; deployment or gate problem."
    if n_survivors == 0 and n_tested > 0:
        return "Case D — everything dies before confirmation; move upstream."
    if human_believable > 0 and n_artifact_failed > 0:
        return "Case A — scope rejection > 0, artifact failures > 0, human believable > 0; architecture improving."
    if n_survivors > 0 and human_believable == 0:
        return "Case B — scope rejection > 0, human kills all survivors; study artifact gate gaps."
    if n_survivors > 0:
        return "Case B — survivors exist but none personally believable."
    return "Case D — no confirmed survivors."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import psycopg
    except ImportError:
        print("psycopg required", file=sys.stderr)
        return 1

    dsn = "postgresql://propab:propab@localhost:5432/propab"
    with psycopg.connect(dsn) as conn:
        data = _fetch_campaign(conn, args.campaign_id)

    camp = data["campaign"]
    hyps = data["hypotheses"]
    question = str(camp.get("question") or "")

    verdict_counts = {"confirmed": 0, "refuted": 0, "inconclusive": 0, "pending": 0}
    for h in hyps:
        v = str(h.get("verdict") or "pending")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    scope_event = data.get("scope_gate_event") or {}
    gate_fail = _gate_failures(hyps)
    n_tested = sum(1 for h in hyps if h.get("verdict") not in (None, "pending"))
    n_gen = int(scope_event.get("n_generated") or camp.get("total_hypotheses") or len(hyps) or 1)
    n_scope_rej = int(scope_event.get("n_scope_rejected") or 0)
    metrics = {
        "campaign_id": args.campaign_id,
        "question": question,
        "status": camp.get("status"),
        "n_generated": n_gen,
        "n_scope_rejected": n_scope_rej,
        "n_scope_passed": scope_event.get("n_scope_passed", 0),
        "scope_rejection_rate": scope_event.get("scope_rejection_rate", round(n_scope_rej / max(1, n_gen), 4)),
        "rejection_reasons": scope_event.get("rejection_reasons", {}),
        "n_ood_failed": gate_fail["n_ood_failed"],
        "n_artifact_failed": gate_fail["n_artifact_failed"],
        "ood_failure_rate": round(gate_fail["n_ood_failed"] / max(1, n_tested), 4),
        "artifact_failure_rate": round(gate_fail["n_artifact_failed"] / max(1, n_tested), 4),
        "n_confirmed": verdict_counts.get("confirmed", 0),
        "n_refuted": verdict_counts.get("refuted", 0),
        "n_inconclusive": verdict_counts.get("inconclusive", 0),
        "n_pending": verdict_counts.get("pending", 0),
        "total_hypotheses": len(hyps),
    }

    steps_by_hyp = {}
    for s in data["steps"]:
        steps_by_hyp.setdefault(s["hypothesis_id"], []).append(s)

    tested = [h for h in hyps if h.get("verdict") not in (None, "pending")]
    rng = random.Random(args.seed)
    sample = rng.sample(tested, min(5, len(tested))) if tested else []

    integrity_rows: list[dict] = []
    for h in sample:
        ev = _parse_evidence(str(h.get("evidence_summary") or ""))
        scope = parse_scope_from_text(str(h.get("text") or "")) or parse_scope_from_methodology(
            str(h.get("text") or ""), None,
        )
        step = (steps_by_hyp.get(h["id"]) or [{}])[-1]
        out = step.get("result_output") if isinstance(step.get("result_output"), dict) else {}
        if not out and ev.get("executed_ood"):
            out = ev
        executed = extract_executed_ood_from_experiment(out or ev, code=str(step.get("source_code") or ""))
        audit_class = _map_audit_class(classify_manual_audit(
            scope, executed, question=question, experiment_code=str(step.get("source_code") or "")[:500],
        ))
        integrity_rows.append({
            "hypothesis_id": h["id"],
            "verdict": h.get("verdict"),
            "declared_scope": scope.to_dict() if scope else None,
            "declared_ood": scope.ood_test if scope else None,
            "executed_ood": executed.to_dict() if executed else None,
            "scope_integrity_from_evidence": ev.get("scope_integrity"),
            "audit_class": audit_class,
            "text_snippet": str(h.get("text") or "")[:240],
            "experiment_code_snippet": str(step.get("source_code") or "")[:400],
        })

    survivors = [h for h in hyps if h.get("verdict") == "confirmed"]
    human_reviews: list[dict] = []
    for h in survivors:
        ev = _parse_evidence(str(h.get("evidence_summary") or ""))
        scope = parse_scope_from_text(str(h.get("text") or ""))
        human_reviews.append({
            "finding": str(h.get("key_finding") or h.get("text") or "")[:400],
            "evidence": {
                "metric_value": ev.get("metric_value"),
                "p_value": ev.get("p_value"),
                "ood_passed": ev.get("ood_passed"),
                "scope_gate_result": ev.get("scope_gate_result"),
            },
            "top_artifact": (ev.get("artifact_gate") or {}).get("ranked_artifacts", [{}])[0]
            if isinstance(ev.get("artifact_gate"), dict) else ev.get("top_artifact"),
            "ood_test": scope.ood_test if scope else ev.get("executed_ood"),
            "human_classification": None,
            "strongest_counterargument": None,
            "reviewer_prompt": (
                "What is the strongest argument that this finding is false? "
                "Classify: obviously_wrong | artifact_explains_it | unclear | interesting | personally_believable"
            ),
        })

    n_believable = sum(1 for r in human_reviews if r.get("human_classification") == "personally_believable")
    interpretation = _interpretation(
        float(metrics.get("scope_rejection_rate") or 0),
        len(survivors),
        n_believable,
        int(metrics.get("n_artifact_failed") or 0),
        n_tested,
    )

    art = ROOT / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / "campaign_scope_metrics.json").write_text(json.dumps({**metrics, "interpretation": interpretation}, indent=2), encoding="utf-8")
    (art / "scope_integrity_audit.json").write_text(json.dumps({
        "campaign_id": args.campaign_id,
        "n_sampled": len(integrity_rows),
        "samples": integrity_rows,
        "audit_class_counts": {
            c: sum(1 for r in integrity_rows if r["audit_class"] == c)
            for c in {r["audit_class"] for r in integrity_rows}
        },
    }, indent=2), encoding="utf-8")
    (art / "human_review.json").write_text(json.dumps({
        "campaign_id": args.campaign_id,
        "primary_metric": "Would I personally believe this finding if another lab showed it to me?",
        "n_survivors": len(survivors),
        "findings": human_reviews,
        "interpretation": interpretation,
        "note": "Fill human_classification and strongest_counterargument manually after review.",
    }, indent=2), encoding="utf-8")

    print(json.dumps({
        "campaign_scope_metrics": str(art / "campaign_scope_metrics.json"),
        "scope_integrity_audit": str(art / "scope_integrity_audit.json"),
        "human_review": str(art / "human_review.json"),
        "interpretation": interpretation,
        "scope_rejection_rate": metrics["scope_rejection_rate"],
        "n_confirmed": metrics["n_confirmed"],
        "n_survivors_for_review": len(human_reviews),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
