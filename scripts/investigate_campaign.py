#!/usr/bin/env python3
"""Diagnose a campaign that looks stuck or has status/event mismatch (fixes.md)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def _get(url: str) -> Any:
    with urlopen(url, timeout=120) as r:
        return json.load(r)


def _payload(ev: dict) -> dict:
    raw = ev.get("payload_json") or ev.get("payload") or {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def investigate(campaign_id: str, api: str) -> dict[str, Any]:
    base = api.rstrip("/")
    camp_resp = _get(f"{base}/campaigns/{campaign_id}")
    camp = camp_resp.get("campaign") or camp_resp
    sess = camp_resp.get("research_session") or {}
    summary = camp_resp.get("summary") or {}

    events = _get(f"{base}/sessions/{campaign_id}/events?limit=500")
    evs = events.get("events") if isinstance(events, dict) else events
    if not isinstance(evs, list):
        evs = []

    steps = Counter(e.get("step") for e in evs)
    salvage = next((e for e in evs if e.get("step") == "campaign.complete_salvaged"), None)
    failed = next((e for e in evs if e.get("step") == "campaign.failed"), None)
    last_campaign = None
    for e in reversed(evs):
        if (e.get("step") or "").startswith("campaign."):
            last_campaign = e
            break

    diagnosis: list[str] = []
    campaign_status = camp.get("status")
    session_status = sess.get("status")

    if session_status == "completed" and campaign_status == "active":
        diagnosis.append(
            "STATUS_DESYNC: research_sessions=completed but research_campaigns=active "
            "(monitor will poll forever; salvage path omitted db_save before fix)"
        )
    if summary.get("total_hypotheses", 0) == 0 and steps.get("campaign.verification_diagnostic", 0) > 0:
        diagnosis.append(
            "COUNTER_DESYNC: verification events exist but total_hypotheses=0 in DB"
        )
    if salvage:
        sp = _payload(salvage)
        err = sp.get("salvaged_after_error", "unknown")
        diagnosis.append(f"SALVAGED_EARLY: fatal {err} after ~{summary.get('elapsed_sec', '?')}s")
    if failed:
        fp = _payload(failed)
        diagnosis.append(f"FAILED: {fp.get('error_type')}: {str(fp.get('error_message', ''))[:200]}")

    tree = summary.get("tree") or {}
    pending = (tree.get("verdict_counts") or {}).get("pending", 0)
    if pending and campaign_status == "active" and session_status == "completed":
        diagnosis.append(f"ZOMBIE_ROW: {pending} pending nodes in stale active campaign row")

    return {
        "campaign_id": campaign_id,
        "question": camp.get("question", "")[:120],
        "campaign_status": campaign_status,
        "session_status": session_status,
        "stop_reason": camp.get("stop_reason"),
        "summary": summary,
        "event_count": len(evs),
        "top_steps": dict(steps.most_common(15)),
        "salvage": _payload(salvage) if salvage else None,
        "last_campaign_event": {
            "step": last_campaign.get("step") if last_campaign else None,
            "at": str(last_campaign.get("created_at") if last_campaign else ""),
        },
        "diagnosis": diagnosis,
        "likely_cause": diagnosis[0] if diagnosis else "no_obvious_issue",
        "not_synthesis_trigger_regression": bool(
            salvage and _payload(salvage).get("salvaged_after_error") == "FileNotFoundError"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_id")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = investigate(args.campaign_id, args.api)
    out = args.out or ROOT / "artifacts" / f"campaign_investigation_{args.campaign_id[:8]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "diagnosis": report["diagnosis"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
