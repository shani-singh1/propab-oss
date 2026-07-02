#!/usr/bin/env python3
"""Post-run analysis for Mandrake redesign campaigns (belief state, synthesis, frontier)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "mandrake_redesign_campaign_analysis.json"


def _get_json(url: str) -> Any:
    with urlopen(url, timeout=120) as resp:
        return json.loads(resp.read())


def _payload(ev: dict) -> dict:
    raw = ev.get("payload_json") or ev.get("payload") or {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower().strip())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio()


def _belief_timeline(synthesis_events: list[dict]) -> list[dict]:
    timeline: list[dict] = []
    for i, ev in enumerate(synthesis_events, 1):
        p = _payload(ev)
        active = p.get("active_beliefs") or []
        timeline.append({
            "round": i,
            "timestamp": ev.get("created_at") or ev.get("timestamp"),
            "n_candidates_added": p.get("n_candidates_added"),
            "branch_exhausted": p.get("branch_exhausted"),
            "exhaustion_rounds": p.get("exhaustion_rounds"),
            "critical_experiment": p.get("critical_experiment"),
            "beliefs": [
                {
                    "statement": b.get("statement"),
                    "confidence": b.get("confidence"),
                    "status": b.get("status"),
                    "supporting_nodes": b.get("supporting_nodes"),
                    "contradicting_nodes": b.get("contradicting_nodes"),
                }
                for b in active if isinstance(b, dict)
            ],
        })
    return timeline


def _belief_evolution_score(timeline: list[dict]) -> dict[str, Any]:
    if not timeline:
        return {"verdict": "NO_DATA", "notes": "No campaign.synthesis events found"}
    rounds = len(timeline)
    conf_changes = 0
    status_changes = 0
    new_beliefs = 0
    abandoned = 0
    for i in range(1, len(timeline)):
        prev = {b["statement"]: b for b in timeline[i - 1]["beliefs"] if b.get("statement")}
        curr = {b["statement"]: b for b in timeline[i]["beliefs"] if b.get("statement")}
        for stmt, b in curr.items():
            if stmt not in prev:
                new_beliefs += 1
            else:
                if b.get("confidence") != prev[stmt].get("confidence"):
                    conf_changes += 1
                if b.get("status") != prev[stmt].get("status"):
                    status_changes += 1
        for stmt, b in prev.items():
            if b.get("status") == "abandoned" or stmt not in curr:
                abandoned += 1
    sensible = (conf_changes + status_changes + new_beliefs) > 0 or rounds == 1
    return {
        "verdict": "PASS" if sensible else "WEAK",
        "synthesis_rounds": rounds,
        "confidence_changes": conf_changes,
        "status_changes": status_changes,
        "new_beliefs_introduced": new_beliefs,
        "beliefs_abandoned_or_dropped": abandoned,
        "final_beliefs": timeline[-1]["beliefs"] if timeline else [],
    }


def _critical_experiment_score(timeline: list[dict]) -> dict[str, Any]:
    exps = [t.get("critical_experiment") for t in timeline if t.get("critical_experiment")]
    if not exps:
        return {"verdict": "NO_DATA", "n_rounds_with_crit": 0}
    titles = [str(e.get("title") or "") for e in exps if isinstance(e, dict)]
    unique_titles = len(set(_norm_text(t) for t in titles if t))
    repeats = len(titles) - unique_titles
    discriminate = sum(
        1 for e in exps
        if isinstance(e, dict) and len(e.get("discriminates_between") or []) >= 2
    )
    generic = sum(
        1 for t in titles
        if re.search(r"systematic|broad|diverse set|investigate further|more combinations", t, re.I)
    )
    improved = unique_titles >= max(1, len(exps) // 2) and generic < len(exps)
    return {
        "verdict": "PASS" if improved and discriminate >= 1 else ("PARTIAL" if exps else "NO_DATA"),
        "n_rounds_with_crit": len(exps),
        "unique_titles": unique_titles,
        "repeated_titles": repeats,
        "rounds_with_2plus_rivals": discriminate,
        "generic_title_count": generic,
        "titles": titles,
    }


def _frontier_recycling(tree: dict, events: list[dict]) -> dict[str, Any]:
    nodes = tree.get("nodes") or {}
    texts = [_norm_text(n.get("text") or "") for n in nodes.values() if n.get("text")]
    dup_pairs = []
    for i, a in enumerate(texts):
        for j, b in enumerate(texts):
            if j <= i:
                continue
            sim = _similarity(a, b)
            if sim >= 0.85:
                dup_pairs.append({"sim": round(sim, 3), "i": i, "j": j})

    snap_events = [e for e in events if (e.get("step") or "") == "campaign.frontier_snapshot"]
    theme_entropy = [(_payload(e).get("theme_entropy")) for e in snap_events]
    theme_entropy = [x for x in theme_entropy if isinstance(x, (int, float))]

    frontier_ids = tree.get("frontier") or []
    pending_frontier = [
        nodes[fid]["text"][:120]
        for fid in frontier_ids
        if fid in nodes and nodes[fid].get("verdict") == "pending"
    ]

    recycled = len(dup_pairs) >= 3
    return {
        "verdict": "PASS" if not recycled and len(dup_pairs) <= 1 else "FAIL",
        "n_nodes": len(nodes),
        "near_duplicate_pairs": dup_pairs[:10],
        "frontier_snapshot_count": len(snap_events),
        "theme_entropy_trend": theme_entropy[-5:] if theme_entropy else [],
        "pending_frontier_sample": pending_frontier[:5],
    }


def _branch_exhaustion(timeline: list[dict], camp: dict) -> dict[str, Any]:
    bs = camp.get("belief_state") or {}
    exhausted_events = [t for t in timeline if t.get("branch_exhausted")]
    final_exhausted = bool(bs.get("branch_exhausted")) or bool(exhausted_events)
    exhaustion_rounds = bs.get("exhaustion_rounds") or (
        exhausted_events[-1].get("exhaustion_rounds") if exhausted_events else 0
    )
    direction_exhausted_in_synthesis = any(
        (_payload(e).get("direction_exhausted")) for e in []
    )
    return {
        "verdict": "PASS" if final_exhausted or exhaustion_rounds >= 1 else "NOT_TRIGGERED",
        "final_branch_exhausted": final_exhausted,
        "exhaustion_rounds": exhaustion_rounds,
        "closed_beliefs": bs.get("closed_beliefs") or [],
        "synthesis_rounds_reporting_exhausted": len(exhausted_events),
        "note": "NOT_TRIGGERED is OK if campaign ended on budget with live rivals",
    }


def _belief_discrimination(timeline: list[dict]) -> dict[str, Any]:
    if not timeline:
        return {"verdict": "NO_DATA"}
    last = timeline[-1]["beliefs"]
    stmts = [b.get("statement") or "" for b in last if b.get("statement")]
    if len(stmts) < 2:
        return {"verdict": "WEAK", "n_active": len(stmts), "note": "<2 active beliefs at end"}
    pairs = []
    for i, a in enumerate(stmts):
        for j, b in enumerate(stmts):
            if j <= i:
                continue
            pairs.append({"i": i, "j": j, "similarity": round(_similarity(a, b), 3)})
    max_sim = max((p["similarity"] for p in pairs), default=0)
    has_contradictions = any(
        b.get("contradicting_nodes") for b in last if isinstance(b, dict)
    )
    distinct = max_sim < 0.55
    return {
        "verdict": "PASS" if distinct else "FAIL",
        "n_active": len(stmts),
        "max_pairwise_similarity": max_sim,
        "pairwise": pairs,
        "any_contradicting_nodes_recorded": has_contradictions,
        "statements": stmts,
    }


def analyze(campaign_id: str, api: str) -> dict[str, Any]:
    base = api.rstrip("/")
    camp_resp = _get_json(f"{base}/campaigns/{campaign_id}")
    camp = camp_resp.get("campaign") or camp_resp
    summary = camp_resp.get("summary") or {}
    tree = camp.get("hypothesis_tree") or {}

    events = _get_json(f"{base}/sessions/{campaign_id}/events?limit=2000")
    if not isinstance(events, list):
        events = events.get("events", [])

    synthesis_events = [
        e for e in events
        if (e.get("step") or "") == "campaign.synthesis"
        or (_payload(e).get("phase") == "synthesis")
    ]
    timeline = _belief_timeline(synthesis_events)

    verdicts = Counter(n.get("verdict") for n in (tree.get("nodes") or {}).values())

    report = {
        "campaign_id": campaign_id,
        "question": camp.get("question"),
        "status": camp.get("status"),
        "architecture": "campaign_synthesis_redesign",
        "summary": summary,
        "verdict_counts": dict(verdicts),
        "event_counts": dict(Counter(e.get("event_type") for e in events)),
        "synthesis_event_count": len(synthesis_events),
        "belief_evolution": _belief_evolution_score(timeline),
        "critical_experiment": _critical_experiment_score(timeline),
        "frontier_recycling": _frontier_recycling(tree, events),
        "branch_exhaustion": _branch_exhaustion(timeline, camp),
        "belief_discrimination": _belief_discrimination(timeline),
        "belief_timeline": timeline,
        "final_belief_state": camp.get("belief_state"),
    }

    checks = [
        report["belief_evolution"]["verdict"] in ("PASS", "WEAK"),
        report["critical_experiment"]["verdict"] in ("PASS", "PARTIAL"),
        report["frontier_recycling"]["verdict"] == "PASS",
        report["branch_exhaustion"]["verdict"] in ("PASS", "NOT_TRIGGERED"),
        report["belief_discrimination"]["verdict"] in ("PASS", "WEAK"),
    ]
    report["overall"] = {
        "checks_passed": sum(checks),
        "checks_total": len(checks),
        "verdict": "PASS" if sum(checks) >= 4 else "PARTIAL" if sum(checks) >= 2 else "FAIL",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Mandrake redesign campaign")
    parser.add_argument("campaign_id", nargs="?", help="Campaign UUID (default: latest json)")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    cid = args.campaign_id
    if not cid:
        latest = ROOT / "artifacts" / "mandrake_campaign_latest.json"
        if not latest.is_file():
            print("No campaign_id and no mandrake_campaign_latest.json", file=sys.stderr)
            return 1
        cid = json.loads(latest.read_text(encoding="utf-8"))["campaign_id"]

    report = analyze(cid, args.api)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "campaign_id": cid,
        "status": report["status"],
        "synthesis_rounds": report["synthesis_event_count"],
        "verdict_counts": report["verdict_counts"],
        "belief_evolution": report["belief_evolution"]["verdict"],
        "critical_experiment": report["critical_experiment"]["verdict"],
        "frontier_recycling": report["frontier_recycling"]["verdict"],
        "branch_exhaustion": report["branch_exhaustion"]["verdict"],
        "belief_discrimination": report["belief_discrimination"]["verdict"],
        "overall": report["overall"]["verdict"],
        "written": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
