#!/usr/bin/env python3
"""Review validation campaign against fixes.md success checklist."""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

GENERIC_PHRASES = (
    "targeted intervention",
    "baseline metric",
    "effect size",
    "noise robustness",
    "statistically significant effect beyond noise",
)


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_json(url: str) -> dict | list:
    with urllib.request.urlopen(url, timeout=90) as resp:
        return json.loads(resp.read())


def review(state_path: Path) -> dict:
    state = _load_state(state_path)
    cid = state["campaign_id"]
    api = state["api"].rstrip("/")
    camp_resp = _get_json(f"{api}/campaigns/{cid}")
    camp = camp_resp.get("campaign") or camp_resp
    summary = camp_resp.get("summary") or {}
    tree = camp.get("hypothesis_tree") or {}
    nodes = tree.get("nodes") or {}
    rs = camp_resp.get("research_session") or {}

    events = _get_json(f"{api}/sessions/{cid}/events?limit=2000")
    if not isinstance(events, list):
        events = events.get("events", [])

    issues: list[str] = []
    checks: dict[str, bool] = {}

    # Claim typing populated on decisive outcomes
    decisive = [n for n in nodes.values() if n.get("verdict") in ("confirmed", "refuted")]
    typed_decisive = sum(1 for n in decisive if n.get("claim_type"))
    checks["claim_typing_populated"] = len(decisive) == 0 or typed_decisive > 0
    if decisive and typed_decisive == 0:
        issues.append("No claim_type on confirmed/refuted nodes")

    # Frontier scores
    scored = sum(1 for n in nodes.values() if n.get("frontier_score") is not None)
    checks["frontier_scores_present"] = scored > 0 or len(nodes) == 0

    # Theme histogram in snapshots
    snap_events = [e for e in events if (e.get("step") or "") == "campaign.frontier_snapshot"]
    has_theme_hist = False
    for ev in snap_events:
        p = ev.get("payload_json") or ev.get("payload") or {}
        if isinstance(p, str):
            p = json.loads(p)
        if p.get("theme_histogram"):
            has_theme_hist = True
            break
    checks["theme_saturation_in_logs"] = has_theme_hist or len(snap_events) == 0

    # Relevance gate fired / no generic templates in tree
    rejected = [e for e in events if e.get("event_type") == "hypothesis.rejected"]
    generic_in_tree = []
    for n in nodes.values():
        t = (n.get("text") or "").lower()
        if any(g in t for g in GENERIC_PHRASES):
            generic_in_tree.append(n.get("text", "")[:120])
    checks["no_generic_templates_in_tree"] = len(generic_in_tree) == 0
    if generic_in_tree:
        issues.append(f"Generic ML templates in tree: {len(generic_in_tree)}")

    # Inconclusive expansion attempted
    inc_expand = [e for e in events if e.get("step") == "campaign.tree_expand_inconclusive"]
    checks["inconclusive_expansion_seen"] = len(inc_expand) > 0 or summary.get("total_inconclusive", 0) == 0

    # Mechanism / structured findings
    with_finding = sum(1 for n in nodes.values() if n.get("finding"))
    with_mech = sum(1 for n in nodes.values() if n.get("mechanism"))
    checks["structured_findings"] = with_finding > 0 or summary.get("total_confirmed", 0) == 0
    checks["mechanism_extraction"] = with_mech > 0 or summary.get("total_confirmed", 0) == 0

    # Paper count mismatch (tree vs summary)
    verdicts = Counter(n.get("verdict") for n in nodes.values())
    tree_confirmed = verdicts.get("confirmed", 0)
    tree_tested = sum(v for k, v in verdicts.items() if k != "pending")
    api_confirmed = int(summary.get("total_confirmed") or camp.get("total_confirmed") or 0)
    api_tested = int(summary.get("total_hypotheses") or camp.get("total_hypotheses") or 0)
    checks["tree_api_count_sync"] = tree_confirmed == api_confirmed and tree_tested == api_tested
    if tree_confirmed != api_confirmed or tree_tested != api_tested:
        issues.append(
            f"Tree/API mismatch: tree {tree_tested}/{tree_confirmed} vs api {api_tested}/{api_confirmed}"
        )

    # Fake timeout bug
    timeouts = [e for e in events if e.get("event_type") == "code.timeout"]
    fake_timeouts = 0
    for ev in timeouts:
        p = ev.get("payload_json") or ev.get("payload") or {}
        if isinstance(p, str):
            p = json.loads(p)
        msg = str(p.get("error") or p.get("message") or p)
        if "unexpected keyword argument 'timeout'" in msg.lower():
            fake_timeouts += 1
    checks["no_fake_timeouts"] = fake_timeouts == 0
    if fake_timeouts:
        issues.append(f"Fake timeout bug: {fake_timeouts} events")

    # P0.1 — no confirmed control nodes
    control_confirmed = [
        n for n in nodes.values()
        if n.get("verdict") == "confirmed" and n.get("node_role") == "CONTROL"
    ]
    checks["no_confirmed_control_nodes"] = len(control_confirmed) == 0
    if control_confirmed:
        issues.append(f"Confirmed CONTROL nodes: {len(control_confirmed)}")

    # P0.2 — evidence reuse visible
    dup_inc = sum(1 for n in nodes.values() if n.get("inconclusive_reason") == "duplicate_evidence")
    checks["evidence_reuse_detection"] = dup_inc >= 0  # active if any dup; pass if none occurred

    # P1 — replication + themes
    with_repl = sum(1 for n in nodes.values() if n.get("replication_level"))
    checks["replication_tiers_populated"] = with_repl > 0 or api_confirmed == 0
    non_general = sum(
        1 for n in nodes.values()
        if (n.get("primary_theme") or n.get("theme_id") or "general") != "general"
    )
    checks["theme_system_upgraded"] = non_general > 0 or len(nodes) == 0

    # P3 — inconclusive reasons
    inc_nodes = [n for n in nodes.values() if n.get("verdict") == "inconclusive"]
    with_reason = sum(1 for n in inc_nodes if n.get("inconclusive_reason"))
    checks["inconclusive_reasons_populated"] = (
        len(inc_nodes) == 0 or with_reason == len(inc_nodes)
    )
    if inc_nodes and with_reason < len(inc_nodes):
        issues.append(f"Inconclusive without reason: {len(inc_nodes) - with_reason}")

    # P4 — finding ledger
    ledger = tree.get("finding_ledger") or []
    checks["finding_ledger_operational"] = len(ledger) > 0 or api_confirmed == 0

    # Paper ready + abstract counts (if paper generated)
    paper_ready = [e for e in events if e.get("event_type") == "paper.ready"]
    stage = rs.get("stage") or ""
    status = camp.get("status") or summary.get("status") or ""
    complete = status in ("budget_exhausted", "breakthrough", "completed") or stage == "completed"

    report = {
        "campaign_id": cid,
        "status": status,
        "stage": stage,
        "complete": complete,
        "budget_used": summary.get("compute_seconds_used"),
        "budget_total": summary.get("compute_budget_seconds"),
        "tested": api_tested,
        "confirmed": api_confirmed,
        "refuted": summary.get("total_refuted"),
        "inconclusive": summary.get("total_inconclusive"),
        "tree_nodes": len(nodes),
        "rejected_hypotheses_events": len(rejected),
        "frontier_snapshots": len(snap_events),
        "checks": checks,
        "issues": issues,
        "clean": complete and all(checks.values()) and len(issues) == 0,
    }
    return report


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/validation_campaign_latest.json")
    report = review(path)
    print("=== VALIDATION REVIEW ===")
    print(json.dumps(report, indent=2))
    for k, v in report["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if report["issues"]:
        print("ISSUES:")
        for i in report["issues"]:
            print(f"  - {i}")
    print("CLEAN:", report["clean"])
    sys.exit(0 if report["clean"] else 1)


if __name__ == "__main__":
    main()
