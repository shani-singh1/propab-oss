#!/usr/bin/env python3
"""Progress snapshot for network resilience campaign."""
from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

STATE = Path("artifacts/network_resilience_campaign_latest.json")
state = json.loads(STATE.read_text(encoding="utf-8"))
cid = state["campaign_id"]
api = state["api"].rstrip("/")


def get(url: str):
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.loads(r.read())


c = get(f"{api}/campaigns/{cid}")
camp = c.get("campaign") or c
s = c.get("summary") or {}
tree = camp.get("hypothesis_tree") or {}
nodes = tree.get("nodes") or {}
rs = c.get("research_session") or {}
used = float(s.get("compute_seconds_used") or camp.get("compute_seconds_used") or 0)
budget = float(s.get("compute_budget_seconds") or camp.get("compute_budget_seconds") or 14400)

events = get(f"{api}/sessions/{cid}/events?limit=2000")
if not isinstance(events, list):
    events = events.get("events", [])
t = Counter(e.get("event_type") for e in events)
steps = Counter(e.get("step") or "" for e in events)

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
print(f"=== NETWORK RESILIENCE @ {now} ===")
print(f"campaign: {cid}")
print(f"status: {camp.get('status')} | stage: {rs.get('stage')}")
print(f"budget: {used:.0f}/{budget:.0f}s ({100*used/max(budget,1):.1f}%) | remaining {max(0,budget-used)/3600:.2f}h")
print(
    f"tested: {s.get('total_hypotheses')} confirmed: {s.get('total_confirmed')} "
    f"refuted: {s.get('total_refuted')} inconclusive: {s.get('total_inconclusive')}"
)
print(f"tree: {len(nodes)} nodes | frontier: {tree.get('frontier_size')}")
print(
    f"agents: completed={t.get('agent.completed',0)} failed={t.get('agent.failed',0)} | "
    f"dispatched={t.get('hypothesis.dispatched',0)}"
)
print(
    f"sandbox: sub={t.get('code.submitted',0)} ok={t.get('code.result',0)} "
    f"timeout={t.get('code.timeout',0)} err={t.get('code.error',0)}"
)
print(
    f"diagnostics: rejected={t.get('hypothesis.rejected',0)} snapshots={steps.get('campaign.frontier_snapshot',0)} "
    f"expanded={t.get('hypothesis.tree_expanded',0)}"
)
verdicts = Counter(n.get("verdict") for n in nodes.values())
print(f"verdicts: {dict(verdicts)}")
print(f"claim_typed: {sum(1 for n in nodes.values() if n.get('claim_type'))}")

failed = [e for e in events if e.get("event_type") == "session.failed"]
if failed:
    p = failed[-1].get("payload_json") or failed[-1].get("payload") or {}
    if isinstance(p, str):
        p = json.loads(p)
    print(f"ALERT session.failed: {str(p.get('traceback') or p)[:400]}")

complete = camp.get("status") in ("budget_exhausted", "breakthrough") or rs.get("stage") == "completed"
print(f"COMPLETE: {complete}")
