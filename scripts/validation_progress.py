#!/usr/bin/env python3
"""Quick validation campaign progress (5-min review)."""
from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

path = Path("artifacts/validation_campaign_latest.json")
state = json.loads(path.read_text(encoding="utf-8"))
cid = state["campaign_id"]
api = state["api"].rstrip("/")

def get(u):
    with urllib.request.urlopen(u, timeout=90) as r:
        return json.loads(r.read())

c = get(f"{api}/campaigns/{cid}")
camp = c.get("campaign") or c
s = c.get("summary") or {}
tree = camp.get("hypothesis_tree") or {}
nodes = tree.get("nodes") or {}
rs = c.get("research_session") or {}
used = float(s.get("compute_seconds_used") or camp.get("compute_seconds_used") or 0)
budget = float(s.get("compute_budget_seconds") or camp.get("compute_budget_seconds") or 1800)

events = get(f"{api}/sessions/{cid}/events?limit=2000")
if not isinstance(events, list):
    events = events.get("events", [])
t = Counter(e.get("event_type") for e in events)
steps = Counter(e.get("step") or "" for e in events)

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
print(f"=== VALIDATION PROGRESS @ {now} ===")
print(f"campaign: {cid}")
print(f"status: {camp.get('status')} | stage: {rs.get('stage')}")
print(f"budget: {used:.0f}/{budget:.0f}s ({100*used/max(budget,1):.1f}%)")
print(f"tested: {s.get('total_hypotheses')} confirmed: {s.get('total_confirmed')} refuted: {s.get('total_refuted')} inconclusive: {s.get('total_inconclusive')}")
print(f"tree: {len(nodes)} nodes | frontier: {tree.get('frontier_size')}")
print(f"events: rejected={t.get('hypothesis.rejected',0)} snapshots={steps.get('campaign.frontier_snapshot',0)} merit_pruned={steps.get('campaign.expansion_merit_gate',0)}")
print(f"sandbox: submitted={t.get('code.submitted',0)} result={t.get('code.result',0)} timeout={t.get('code.timeout',0)} error={t.get('code.error',0)}")
verdicts = Counter(n.get("verdict") for n in nodes.values())
print(f"verdicts: {dict(verdicts)}")
typed = sum(1 for n in nodes.values() if n.get("claim_type"))
print(f"claim_typed_nodes: {typed} | theme_ids: {sum(1 for n in nodes.values() if n.get('theme_id'))}")
complete = camp.get("status") in ("budget_exhausted", "breakthrough") or rs.get("stage") == "completed"
print(f"COMPLETE: {complete}")
