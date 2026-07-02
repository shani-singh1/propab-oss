#!/usr/bin/env python3
"""Extract Mandrake campaign run data for analysis."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

CID = "1a61b453-38e0-4023-9135-cf36caeed505"
API = "http://localhost:8000"
ROOT = Path(__file__).resolve().parents[1]


def get_json(url: str):
    with urlopen(url, timeout=120) as resp:
        return json.loads(resp.read())


def payload(ev: dict) -> dict:
    raw = ev.get("payload_json") or ev.get("payload") or {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def main() -> None:
    camp_resp = get_json(f"{API}/campaigns/{CID}")
    camp = camp_resp.get("campaign") or camp_resp
    summary = camp_resp.get("summary") or {}
    tree = camp.get("hypothesis_tree") or {}
    nodes = tree.get("nodes") or {}

    events = get_json(f"{API}/sessions/{CID}/events?limit=2000")
    if not isinstance(events, list):
        events = events.get("events", [])

    out = {
        "campaign_id": CID,
        "summary": summary,
        "campaign_fields": {
            k: camp.get(k)
            for k in (
                "question", "status", "seed_source", "anomaly_artifacts_dir",
                "max_hypotheses_cap", "compute_budget_seconds", "compute_seconds_used",
                "baseline_metric", "best_metric", "breakthrough_criteria",
            )
        },
        "event_counts": dict(Counter(e.get("event_type") for e in events)),
        "nodes": [],
        "anomaly_seeds": [],
        "paper_ready": None,
        "verification_diagnostics": [],
    }

    for n in sorted(nodes.values(), key=lambda x: (x.get("generation", 0), x.get("rank", 0))):
        out["nodes"].append({
            "id": n.get("id"),
            "generation": n.get("generation"),
            "rank": n.get("rank"),
            "verdict": n.get("verdict"),
            "text": (n.get("text") or "")[:500],
            "test_methodology": (n.get("test_methodology") or "")[:300],
            "theme_id": n.get("theme_id"),
            "claim_type": n.get("claim_type"),
            "metrics": n.get("metrics"),
        })

    for ev in events:
        et = ev.get("event_type")
        if et == "hypothesis.generated" and (ev.get("step") or "").endswith("anomaly_seed"):
            p = payload(ev)
            out["anomaly_seeds"].append({
                "step": ev.get("step"),
                "n_mechanisms": p.get("n_mechanisms"),
                "hypotheses": p.get("hypotheses"),
            })
        if et == "paper.ready":
            out["paper_ready"] = payload(ev)
        if et == "campaign.progress" and payload(ev).get("phase") == "verification_diagnostic":
            out["verification_diagnostics"].append(payload(ev))

    mech_path = ROOT / "artifacts" / "mechanism_objects.json"
    anom_path = ROOT / "artifacts" / "anomaly_objects.json"
    if mech_path.is_file():
        out["upstream_mechanisms"] = json.loads(mech_path.read_text(encoding="utf-8"))
    if anom_path.is_file():
        anoms = json.loads(anom_path.read_text(encoding="utf-8"))
        out["upstream_anomaly_summary"] = {
            "count": len(anoms),
            "types": dict(Counter(a.get("anomaly_type") for a in anoms)),
            "top_features": [a.get("feature_subset") for a in anoms[:8]],
        }

    dest = ROOT / "artifacts" / "mandrake_run_analysis.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({
        "written": str(dest),
        "hypotheses": len(out["nodes"]),
        "verdicts": dict(Counter(n["verdict"] for n in out["nodes"])),
        "anomaly_seed_batches": len(out["anomaly_seeds"]),
        "paper_ready": bool(out["paper_ready"]),
        "event_types_top": dict(Counter(e.get("event_type") for e in events).most_common(15)),
    }, indent=2))


if __name__ == "__main__":
    main()
