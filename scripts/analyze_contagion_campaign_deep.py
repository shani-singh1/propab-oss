#!/usr/bin/env python3
"""Deep post-mortem for Phase 2 contagion campaign — trajectory, themes, mechanisms, ledger, replication, IG."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import psycopg
except ImportError:
    print("psycopg required", file=sys.stderr)
    sys.exit(1)

CAMPAIGN_ID = "1e419640-7726-4cb3-8b92-303d5af99105"
DSN = "postgresql://propab:propab@localhost:5432/propab"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "contagion_campaign_deep_analysis.json"


def _parse_evidence(summary: str | None) -> dict[str, Any]:
    if not summary:
        return {}
    out: dict[str, Any] = {}
    m = re.search(r"evidence=(\{.*?\});", summary)
    if m:
        try:
            out["evidence"] = json.loads(m.group(1).replace("'", '"'))
        except json.JSONDecodeError:
            out["evidence_raw"] = m.group(1)[:400]
    for key in ("relevance_score", "delta_pct", "p_value", "effect_size", "claim_type"):
        mm = re.search(rf'"{key}":\s*([^,}}]+)', summary)
        if mm:
            out[key] = mm.group(1).strip().strip('"')
    mm = re.search(r"info_gain=([\d.]+)", summary)
    if mm:
        out["info_gain"] = float(mm.group(1))
    mm = re.search(r"relevance=([\d.]+)", summary)
    if mm:
        out["relevance_inline"] = float(mm.group(1))
    return out


def _info_gain_from_expansion(reason: str | None) -> float | None:
    if not reason:
        return None
    m = re.search(r"info_gain=([\d.]+)", reason)
    return float(m.group(1)) if m else None


def _relevance_from_expansion(reason: str | None) -> float | None:
    if not reason:
        return None
    m = re.search(r"relevance=([\d.]+)", reason)
    return float(m.group(1)) if m else None


def main() -> None:
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, status, baseline_metric, best_metric, improvement_pct,
                       total_hypotheses, total_confirmed, compute_budget_seconds,
                       compute_seconds_used, started_at, completed_at, hypothesis_tree_json,
                       breakthrough_criteria_json, best_finding_json
                FROM research_campaigns WHERE id = %s
                """,
                (CAMPAIGN_ID,),
            )
            row = cur.fetchone()
            if not row:
                sys.exit("campaign not found")
            cols = [d[0] for d in cur.description]
            campaign = dict(zip(cols, row, strict=True))
            for k in ("hypothesis_tree_json", "breakthrough_criteria_json", "best_finding_json"):
                v = campaign.get(k)
                if isinstance(v, str):
                    campaign[k] = json.loads(v)

            cur.execute(
                """
                SELECT id, text, verdict, confidence, evidence_summary, key_finding,
                       status, rank, created_at
                FROM hypotheses WHERE session_id = %s ORDER BY created_at
                """,
                (CAMPAIGN_ID,),
            )
            hyps = [dict(zip([d[0] for d in cur.description], r, strict=True)) for r in cur.fetchall()]

            cur.execute(
                """
                SELECT event_type, step, payload_json, created_at
                FROM events WHERE session_id = %s
                ORDER BY created_at
                """,
                (CAMPAIGN_ID,),
            )
            events = [
                {
                    "event_type": r[0],
                    "step": r[1],
                    "payload": r[2] if isinstance(r[2], dict) else (json.loads(r[2]) if r[2] else {}),
                    "created_at": r[3].isoformat() if r[3] else None,
                }
                for r in cur.fetchall()
            ]

            cur.execute(
                """
                SELECT MIN(created_at), MAX(created_at), COUNT(*)
                FROM events WHERE session_id = %s
                """,
                (CAMPAIGN_ID,),
            )
            t0, t1, n_events = cur.fetchone()

            cur.execute(
                "SELECT status, stage, completed_at FROM research_sessions WHERE id = %s",
                (CAMPAIGN_ID,),
            )
            sess = cur.fetchone()

            cur.execute(
                """
                SELECT COALESCE(SUM(duration_ms),0) FROM llm_calls WHERE session_id = %s
                """,
                (CAMPAIGN_ID,),
            )
            llm_ms = int(cur.fetchone()[0] or 0)

    tree = campaign.get("hypothesis_tree_json") or {}
    nodes: dict[str, dict] = tree.get("nodes") or {}
    ledger: list[dict] = tree.get("finding_ledger") or []
    frontier_ids: list[str] = tree.get("frontier") or []
    exhausted_ids: list[str] = tree.get("exhausted") or []

    # --- Node-level enrichments ---
    node_records: list[dict] = []
    for nid, n in nodes.items():
        ev = _parse_evidence(n.get("evidence_summary"))
        node_records.append({
            "id": nid,
            "generation": n.get("generation"),
            "depth": n.get("depth"),
            "parent_id": n.get("parent_id"),
            "verdict": n.get("verdict"),
            "confidence": n.get("confidence"),
            "node_role": n.get("node_role"),
            "claim_type": n.get("claim_type"),
            "verification_method": n.get("verification_method"),
            "theme_id": n.get("theme_id"),
            "primary_theme": n.get("primary_theme"),
            "secondary_themes": n.get("secondary_themes"),
            "question_relevance_score": n.get("question_relevance_score"),
            "frontier_score": n.get("frontier_score"),
            "expansion_type": n.get("expansion_type"),
            "expansion_reason": n.get("expansion_reason"),
            "expansion_info_gain": _info_gain_from_expansion(n.get("expansion_reason")),
            "expansion_relevance": _relevance_from_expansion(n.get("expansion_reason")),
            "inconclusive_reason": n.get("inconclusive_reason"),
            "inconclusive_expansions": n.get("inconclusive_expansions"),
            "replication_level": n.get("replication_level"),
            "evidence_hash": n.get("evidence_hash"),
            "verification_hash": n.get("verification_hash"),
            "mechanism": n.get("mechanism"),
            "finding": n.get("finding"),
            "text_snippet": (n.get("text") or "")[:220],
            "parsed_evidence": ev,
        })
    node_records.sort(key=lambda x: (x.get("generation") or 0, x.get("id") or ""))

    # --- Search trajectory (by generation wave) ---
    by_gen: dict[int, list[dict]] = defaultdict(list)
    for rec in node_records:
        by_gen[int(rec.get("generation") or 0)].append(rec)

    trajectory_waves: list[dict] = []
    for g in sorted(by_gen.keys()):
        wave = by_gen[g]
        trajectory_waves.append({
            "generation": g,
            "count": len(wave),
            "verdicts": dict(Counter(x["verdict"] for x in wave)),
            "roles": dict(Counter(x.get("node_role") or "?" for x in wave)),
            "themes": dict(Counter(x.get("primary_theme") or x.get("theme_id") or "general" for x in wave)),
            "claim_types": dict(Counter(x.get("claim_type") or "?" for x in wave)),
            "replication_levels": dict(Counter(x.get("replication_level") or "?" for x in wave)),
            "avg_frontier_score": round(
                sum(x["frontier_score"] or 0 for x in wave) / max(len(wave), 1), 4
            ),
            "avg_relevance": round(
                sum(x.get("question_relevance_score") or 0 for x in wave) / max(len(wave), 1), 4
            ),
            "avg_expansion_ig": round(
                sum(x["expansion_info_gain"] or 0 for x in wave if x["expansion_info_gain"]) /
                max(sum(1 for x in wave if x["expansion_info_gain"]), 1),
                4,
            )
            if any(x["expansion_info_gain"] for x in wave)
            else None,
        })

    # --- Theme evolution from frontier snapshots ---
    snap_events = [
        e for e in events
        if (e.get("step") or "") == "campaign.frontier_snapshot"
        or e.get("event_type") == "campaign.frontier_snapshot"
    ]
    theme_timeline: list[dict] = []
    ig_timeline: list[dict] = []
    for i, ev in enumerate(snap_events):
        p = ev.get("payload") or {}
        th = p.get("theme_histogram") or p.get("themes") or {}
        top_frontier = p.get("top_frontier") or p.get("frontier_top") or []
        theme_timeline.append({
            "index": i,
            "at": ev.get("created_at"),
            "theme_histogram": th,
            "frontier_size": p.get("frontier_size"),
            "tested": p.get("total_tested") or p.get("tested"),
            "verdict_counts": p.get("verdict_counts"),
        })
        if isinstance(top_frontier, list):
            for item in top_frontier[:5]:
                if isinstance(item, dict):
                    ig_timeline.append({
                        "snapshot": i,
                        "at": ev.get("created_at"),
                        "node_id": item.get("id"),
                        "frontier_score": item.get("frontier_score") or item.get("score"),
                        "text": (item.get("text") or "")[:120],
                    })

    # --- Mechanism evolution ---
    mech_by_gen: dict[int, list[str]] = defaultdict(list)
    mechanisms_all: list[dict] = []
    for rec in node_records:
        mech = rec.get("mechanism")
        if isinstance(mech, dict):
            mech_str = json.dumps(mech, default=str)[:500]
        elif mech:
            mech_str = str(mech)[:500]
        else:
            mech_str = None
        if mech_str:
            mechanisms_all.append({
                "generation": rec.get("generation"),
                "verdict": rec.get("verdict"),
                "theme": rec.get("primary_theme"),
                "mechanism": mech_str,
                "id": rec.get("id"),
            })
            mech_by_gen[int(rec.get("generation") or 0)].append(mech_str[:200])

    # --- Ledger ---
    ledger_summary = {
        "entries": len(ledger),
        "by_verdict": dict(Counter(e.get("verdict") for e in ledger)),
        "by_theme": dict(Counter(e.get("theme") or e.get("primary_theme") for e in ledger)),
        "by_replication": dict(Counter(e.get("replication_level") for e in ledger)),
        "sample": ledger[:15],
    }

    # --- Replication evolution ---
    repl_by_gen = {
        str(g): dict(Counter(r.get("replication_level") or "?" for r in wave))
        for g, wave in by_gen.items()
    }
    repl_confirmed = [
        {
            "generation": r.get("generation"),
            "replication_level": r.get("replication_level"),
            "confidence": r.get("confidence"),
            "theme": r.get("primary_theme"),
            "text": r.get("text_snippet"),
        }
        for r in node_records
        if r.get("verdict") == "confirmed"
    ]

    # --- Information gain / frontier merit ---
    expansion_events = [
        e for e in events
        if "expand" in (e.get("step") or "").lower()
        or e.get("event_type") in ("hypothesis.tree_expanded", "campaign.tree_expand_inconclusive")
    ]
    ig_from_expansions: list[dict] = []
    for e in expansion_events:
        p = e.get("payload") or {}
        reason = p.get("expansion_reason") or p.get("reason") or ""
        ig_from_expansions.append({
            "at": e.get("created_at"),
            "step": e.get("step"),
            "reason": reason[:200] if reason else None,
            "info_gain": _info_gain_from_expansion(reason),
            "relevance": _relevance_from_expansion(reason),
            "parent_verdict": p.get("parent_verdict"),
            "children_added": p.get("children_added") or p.get("n_children"),
        })

    frontier_scores = [r["frontier_score"] for r in node_records if r.get("frontier_score") is not None]
    expansion_igs = [r["expansion_info_gain"] for r in node_records if r.get("expansion_info_gain")]

    # --- Event-driven search trajectory (hypothesis tested order proxy) ---
    test_events = [
        e for e in events
        if e.get("event_type") in ("hypothesis.result", "experiment.completed", "hypothesis.verified")
        or (e.get("step") or "").startswith("hypothesis.")
    ]
    event_type_counts = Counter(e.get("event_type") for e in events)
    step_counts = Counter(e.get("step") for e in events if e.get("step"))

    # --- Expansion reason taxonomy ---
    expand_reasons = Counter()
    for r in node_records:
        reason = r.get("expansion_reason") or ""
        if reason:
            tag = reason.split(";")[0].split(",")[0][:80]
            expand_reasons[tag] += 1

    inconclusive_reasons = Counter(r.get("inconclusive_reason") for r in node_records if r.get("verdict") == "inconclusive")

    # --- Duplicate evidence (ledger hygiene) ---
    evidence_hashes = Counter(r.get("evidence_hash") for r in node_records if r.get("evidence_hash"))
    dup_hashes = {h: c for h, c in evidence_hashes.items() if c > 1}

    out = {
        "campaign_id": CAMPAIGN_ID,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "session": {
            "status": sess[0] if sess else None,
            "stage": sess[1] if sess else None,
            "completed_at": sess[2].isoformat() if sess and sess[2] else None,
        },
        "headline": {
            "status": campaign.get("status"),
            "question": campaign.get("question"),
            "baseline_metric": campaign.get("baseline_metric"),
            "best_metric": campaign.get("best_metric"),
            "improvement_pct": campaign.get("improvement_pct"),
            "total_hypotheses": campaign.get("total_hypotheses"),
            "total_confirmed": campaign.get("total_confirmed"),
            "compute_used_sec": campaign.get("compute_seconds_used"),
            "compute_budget_sec": campaign.get("compute_budget_seconds"),
            "best_finding": campaign.get("best_finding_json"),
        },
        "tree_structure": {
            "total_nodes": len(nodes),
            "max_depth": max((n.get("depth", 0) for n in nodes.values()), default=0),
            "max_generation": max((n.get("generation", 0) for n in nodes.values()), default=0),
            "frontier_remaining": len(frontier_ids),
            "exhausted": len(exhausted_ids),
            "verdict_counts": dict(Counter(n.get("verdict") for n in nodes.values())),
            "role_counts": dict(Counter(n.get("node_role") for n in nodes.values())),
        },
        "search_trajectory": {
            "waves_by_generation": trajectory_waves,
            "event_window": {
                "first": t0.isoformat() if t0 else None,
                "last": t1.isoformat() if t1 else None,
                "n_events": n_events,
                "llm_ms": llm_ms,
            },
            "expansion_reason_taxonomy": dict(expand_reasons.most_common(30)),
            "n_expansion_events": len(expansion_events),
            "expansion_event_sample": ig_from_expansions[:40],
        },
        "theme_evolution": {
            "final_theme_histogram": dict(Counter(
                n.get("primary_theme") or n.get("theme_id") or "general" for n in nodes.values()
            )),
            "confirmed_by_theme": dict(Counter(
                n.get("primary_theme") or n.get("theme_id") or "general"
                for n in nodes.values() if n.get("verdict") == "confirmed"
            )),
            "frontier_snapshot_count": len(snap_events),
            "theme_timeline": theme_timeline,
            "frontier_leader_samples": ig_timeline[:60],
        },
        "mechanism_evolution": {
            "nodes_with_mechanism": len(mechanisms_all),
            "by_generation_count": {str(g): len(v) for g, v in sorted(mech_by_gen.items())},
            "mechanisms_sample": mechanisms_all[:25],
        },
        "ledger_usage": ledger_summary,
        "replication_evolution": {
            "by_generation": repl_by_gen,
            "confirmed_nodes": repl_confirmed,
            "tier_totals": dict(Counter(r.get("replication_level") for r in node_records)),
        },
        "information_gain": {
            "frontier_score_stats": {
                "n": len(frontier_scores),
                "min": min(frontier_scores) if frontier_scores else None,
                "max": max(frontier_scores) if frontier_scores else None,
                "mean": round(sum(frontier_scores) / len(frontier_scores), 4) if frontier_scores else None,
            },
            "expansion_info_gain_stats": {
                "n": len(expansion_igs),
                "min": min(expansion_igs) if expansion_igs else None,
                "max": max(expansion_igs) if expansion_igs else None,
                "mean": round(sum(expansion_igs) / len(expansion_igs), 4) if expansion_igs else None,
            },
            "top_frontier_at_test": sorted(
                [r for r in node_records if r.get("frontier_score")],
                key=lambda x: -(x.get("frontier_score") or 0),
            )[:20],
            "confirmed_avg_frontier_score": round(
                sum(r.get("frontier_score") or 0 for r in node_records if r.get("verdict") == "confirmed")
                / max(sum(1 for r in node_records if r.get("verdict") == "confirmed"), 1),
                4,
            ),
        },
        "inconclusive_diagnostics": {
            "reason_counts": dict(inconclusive_reasons),
            "with_expansion_attempted": sum(1 for r in node_records if (r.get("inconclusive_expansions") or 0) > 0),
        },
        "evidence_hygiene": {
            "duplicate_evidence_hashes": len(dup_hashes),
            "top_duplicate_hashes": dict(list(sorted(dup_hashes.items(), key=lambda x: -x[1]))[:10]),
        },
        "event_summary": {
            "total": sum(event_type_counts.values()),
            "by_type_top": dict(event_type_counts.most_common(25)),
            "by_step_top": dict(step_counts.most_common(25)),
        },
        "db_hypotheses_verdicts": dict(Counter(h.get("verdict") or "empty" for h in hyps)),
        "confirmed_key_findings": [
            {
                "text": (h.get("text") or "")[:300],
                "key_finding": h.get("key_finding"),
                "confidence": h.get("confidence"),
            }
            for h in hyps
            if h.get("verdict") == "confirmed"
        ][:20],
        "node_records": node_records,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(json.dumps({k: out[k] for k in ("headline", "tree_structure", "theme_evolution", "replication_evolution", "information_gain", "ledger_usage") if k in out}, indent=2, default=str)[:8000])


if __name__ == "__main__":
    main()
