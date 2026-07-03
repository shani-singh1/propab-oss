#!/usr/bin/env python3
"""Snapshot combinatorics campaign state for wake review (fixes.md Step 4)."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CID = "4b2292c8-5dbd-4772-85a2-18f02d494364"
OUT = ROOT / "artifacts" / "campaign_combinatorics_results.json"
STATE_FILE = ROOT / "artifacts" / "v1_frontier_campaign_latest.json"


def main() -> int:
    cid = DEFAULT_CID
    if STATE_FILE.is_file():
        try:
            cid = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("campaign_id") or cid
        except json.JSONDecodeError:
            pass
    if len(sys.argv) > 1:
        cid = sys.argv[1]

    api = "http://localhost:8000"
    st = json.loads(urllib.request.urlopen(f"{api}/campaigns/{cid}", timeout=60).read())
    camp = st.get("campaign") or st
    tree = camp.get("hypothesis_tree") or {}
    nodes = tree.get("nodes") or {}

    verdicts: dict[str, int] = {}
    confirmed_findings: list[dict] = []
    for nid, node in nodes.items():
        v = node.get("verdict") or "pending"
        verdicts[v] = verdicts.get(v, 0) + 1
        if v == "confirmed":
            confirmed_findings.append({
                "node_id": nid,
                "text": (node.get("text") or "")[:300],
                "evidence_summary": (node.get("evidence_summary") or "")[:500],
                "confidence": node.get("confidence"),
            })

    checklist = {}
    for name in ("combinatorics", "genomics"):
        p = ROOT / "artifacts" / f"domain_checklist_{name}.json"
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            checklist[f"{name}_result"] = "PASS" if all(v.get("pass") for v in data.values()) else "FAIL"

    tested = sum(verdicts.get(k, 0) for k in ("confirmed", "refuted", "inconclusive"))
    confirmed_n = verdicts.get("confirmed", 0)

    if confirmed_n >= 1 and camp.get("status") != "active":
        verdict = "FINDING"
    elif confirmed_n >= 1:
        verdict = "FINDING_IN_PROGRESS"
    elif camp.get("status") == "active" and tested > 0:
        verdict = "NO_FINDING_YET"
    elif tested == 0 and camp.get("status") == "active":
        verdict = "INFRASTRUCTURE_OK"
    else:
        verdict = "NO_FINDING"

    record = {
        "campaign_id": cid,
        "domain": "math_combinatorics",
        "domain_checklist": checklist,
        "campaign_result": {
            "status": camp.get("status"),
            "tested": tested,
            "confirmed": confirmed_n,
            "refuted": verdicts.get("refuted", 0),
            "inconclusive": verdicts.get("inconclusive", 0),
            "pending": verdicts.get("pending", 0),
            "stop_reason": camp.get("stop_reason"),
            "best_finding": confirmed_findings[0] if confirmed_findings else None,
        },
        "confirmed_findings": confirmed_findings[:20],
        "health_metrics": {
            "worker_utilization": camp.get("worker_utilization"),
            "experiment_success_rate": camp.get("worker_experiment_success_rate"),
        },
        "verdict": verdict,
        "next_step_recommendation": (
            "Review confirmed Sidon/cap-set/AP-free findings; compare greedy densities to literature bounds."
            if confirmed_n
            else "Check worker/orchestrator logs if inconclusive rate stays high after evidence-parse fix."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
