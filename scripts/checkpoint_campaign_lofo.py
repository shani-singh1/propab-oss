#!/usr/bin/env python3
"""30-minute checkpoint: lofo_r2 evidence + pipeline verdicts (fixes.md)."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.verdict_pipeline import run_verdict_pipeline  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-id", default=None)
    p.add_argument("--state-file", default=str(ROOT / "artifacts" / "v1_frontier_campaign_latest.json"))
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--limit", type=int, default=500)
    args = p.parse_args()

    cid = args.campaign_id
    if not cid:
        cid = json.loads(Path(args.state_file).read_text(encoding="utf-8"))["campaign_id"]

    api = args.api.rstrip("/")
    camp = json.load(urllib.request.urlopen(f"{api}/campaigns/{cid}", timeout=60))
    summary = camp.get("summary") or {}
    ev_data = json.load(urllib.request.urlopen(f"{api}/sessions/{cid}/events?limit={args.limit}", timeout=60))
    events = ev_data.get("events") or []

    mat_calls = 0
    mat_errors = 0
    lofo_events: list[dict] = []
    tool_verdicts: Counter[str] = Counter()

    for e in events:
        pload = e.get("payload_json") or {}
        et = e.get("event_type") or ""
        if et == "tool.called" and pload.get("tool") == "materials_verification":
            mat_calls += 1
        if et == "tool.error" and pload.get("tool") == "materials_verification":
            mat_errors += 1
        blob = json.dumps(pload)
        if "lofo_r2" in blob:
            lofo_events.append({
                "event_type": et,
                "step": e.get("step"),
                "tool": pload.get("tool"),
                "lofo_r2": pload.get("lofo_r2") or (pload.get("output") or {}).get("lofo_r2"),
                "label_shuffle_null_p95": pload.get("label_shuffle_null_p95")
                or (pload.get("output") or {}).get("label_shuffle_null_p95"),
                "verdict": pload.get("verdict"),
                "verdict_reason": (pload.get("verdict_reason") or "")[:120],
            })
        if et == "agent.completed" and pload.get("materials"):
            tool_verdicts[str(pload.get("verdict"))] += 1

    # Replay sample LOFO tool outputs through fixed pipeline
    pipeline_samples: list[dict] = []
    for e in events:
        pload = e.get("payload_json") or {}
        if e.get("event_type") != "tool.result" or pload.get("tool") != "materials_verification":
            continue
        out = pload.get("output") or {}
        if out.get("lofo_r2") is None:
            continue
        ev_copy = dict(out)
        v, conf, reason = run_verdict_pipeline(ev_copy)
        pipeline_samples.append({
            "lofo_r2": out.get("lofo_r2"),
            "label_shuffle_null_p95": out.get("label_shuffle_null_p95"),
            "pipeline_verdict": v,
            "confidence": conf,
            "reason": reason[:160],
        })
        if len(pipeline_samples) >= 5:
            break

    report = {
        "campaign_id": cid,
        "elapsed_min": round(float(summary.get("elapsed_sec") or 0) / 60, 1),
        "hypotheses_tested": summary.get("total_hypotheses"),
        "confirmed": summary.get("total_confirmed"),
        "materials_verification_calls": mat_calls,
        "materials_verification_errors": mat_errors,
        "events_with_lofo_r2": len(lofo_events),
        "materials_agent_verdicts": dict(tool_verdicts),
        "lofo_event_samples": lofo_events[:5],
        "pipeline_replay_samples": pipeline_samples,
    }

    out_path = ROOT / "artifacts" / f"checkpoint_lofo_{cid[:8]}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
