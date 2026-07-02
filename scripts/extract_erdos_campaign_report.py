#!/usr/bin/env python3
"""Extract Erdős campaign data from Postgres for post-run analysis."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import psycopg
except ImportError:
    print("psycopg required", file=sys.stderr)
    sys.exit(1)

CAMPAIGN_ID = "d1687226-3e57-4222-9926-a720c885bd1e"
DSN = "postgresql://propab:propab@localhost:5432/propab"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "erdos_campaign_extract.json"


def main() -> None:
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, status, baseline_metric, best_metric, improvement_pct,
                       total_hypotheses, total_confirmed, compute_budget_seconds,
                       compute_seconds_used, started_at, completed_at, hypothesis_tree_json,
                       breakthrough_criteria_json
                FROM research_campaigns WHERE id = %s
                """,
                (CAMPAIGN_ID,),
            )
            row = cur.fetchone()
            if not row:
                sys.exit("campaign not found")
            cols = [d[0] for d in cur.description]
            campaign = dict(zip(cols, row, strict=True))
            for k in ("hypothesis_tree_json", "breakthrough_criteria_json"):
                if isinstance(campaign[k], str):
                    campaign[k] = json.loads(campaign[k])

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
                SELECT event_type, COUNT(*) FROM events
                WHERE session_id = %s GROUP BY event_type ORDER BY COUNT(*) DESC
                """,
                (CAMPAIGN_ID,),
            )
            event_counts = dict(cur.fetchall())

            cur.execute(
                """
                SELECT payload_json FROM events
                WHERE session_id = %s AND event_type = 'hypothesis.tree_expanded'
                ORDER BY created_at
                """,
                (CAMPAIGN_ID,),
            )
            expansions = [r[0] for r in cur.fetchall()]

            cur.execute(
                """
                SELECT payload_json FROM events
                WHERE session_id = %s AND event_type IN ('campaign.budget_exhausted','campaign.completed','paper.ready')
                """,
                (CAMPAIGN_ID,),
            )
            terminal = cur.fetchall()

            cur.execute(
                """
                SELECT payload_json FROM events
                WHERE session_id = %s AND event_type = 'code.result'
                AND payload_json::text ILIKE '%%verified%%'
                ORDER BY created_at DESC LIMIT 30
                """,
                (CAMPAIGN_ID,),
            )
            verified_samples = [r[0] for r in cur.fetchall()]

    tree = campaign.get("hypothesis_tree_json") or {}
    nodes = tree.get("nodes") or {}

    # Verdict counts from tree nodes
    tree_verdicts = Counter(n.get("verdict", "?") for n in nodes.values())

    # Group nodes by generation
    by_gen: dict[int, list] = defaultdict(list)
    for n in nodes.values():
        by_gen[int(n.get("generation", 0))].append(n)

    # Seed hypotheses (generation 0)
    seeds = sorted(by_gen.get(0, []), key=lambda x: x.get("id", ""))

    # Confirmed from tree with text
    confirmed_tree = [
        n for n in nodes.values() if n.get("verdict") == "confirmed"
    ]
    confirmed_tree.sort(key=lambda x: (-x.get("confidence", 0), x.get("generation", 0)))

    # Refuted
    refuted_tree = [n for n in nodes.values() if n.get("verdict") == "refuted"]

    # Inconclusive
    inconclusive_tree = [n for n in nodes.values() if n.get("verdict") == "inconclusive"]

    # Pending (never tested - still on frontier)
    pending_tree = [n for n in nodes.values() if n.get("verdict") == "pending"]

    # Roots (depth-0 seeds in tree)
    roots = [n for n in nodes.values() if not n.get("parent_id")]

    ml_kw = (
        "token", "adam", "huber", "gradient", "batch size", "attention",
        "embedding", "mse", "sgd", "cauchy", "laplace", "shannon", "jackknife",
    )
    off_topic_confirmed = [
        n for n in confirmed_tree
        if any(k in (n.get("text") or "").lower() for k in ml_kw)
    ]
    on_topic_confirmed = [n for n in confirmed_tree if n not in off_topic_confirmed]

    # Seed batch from hypothesis.generated event
    seed_batch: list[dict] = []
    with psycopg.connect(DSN) as conn2:
        with conn2.cursor() as cur2:
            cur2.execute(
                """
                SELECT payload_json FROM events
                WHERE session_id = %s AND event_type = 'hypothesis.generated'
                LIMIT 1
                """,
                (CAMPAIGN_ID,),
            )
            row = cur2.fetchone()
            if row and row[0]:
                seed_batch = row[0].get("hypotheses") or []

            cur2.execute(
                "SELECT MIN(created_at), MAX(created_at) FROM events WHERE session_id = %s",
                (CAMPAIGN_ID,),
            )
            t0, t1 = cur2.fetchone()

            cur2.execute(
                "SELECT COALESCE(SUM(duration_ms),0) FROM llm_calls WHERE session_id = %s",
                (CAMPAIGN_ID,),
            )
            llm_ms = int(cur2.fetchone()[0] or 0)

            cur2.execute(
                """
                SELECT COALESCE(SUM(e.duration_ms),0)
                FROM experiment_steps e
                JOIN hypotheses h ON h.id = e.hypothesis_id
                WHERE h.session_id = %s
                """,
                (CAMPAIGN_ID,),
            )
            step_ms = int(cur2.fetchone()[0] or 0)

    wall_sec = None
    if t0 and t1:
        wall_sec = (t1 - t0).total_seconds()

    db_by_verdict = Counter(h.get("verdict") or "empty" for h in hyps)

    # Expansion reasons
    expand_reasons = Counter()
    for p in expansions:
        if isinstance(p, dict):
            reason = p.get("expansion_reason") or p.get("step") or "expand"
            expand_reasons[reason] += 1

    # Dedupe confirmed texts (many near-duplicates)
    def norm_text(t: str) -> str:
        t = re.sub(r"\s+", " ", (t or "").strip().lower())
        t = re.sub(r"^hypothesis\s+\d+\s*:\s*", "", t)
        return t[:120]

    confirmed_unique: dict[str, dict] = {}
    for n in confirmed_tree:
        key = norm_text(n.get("text", ""))
        if key and key not in confirmed_unique:
            confirmed_unique[key] = n

    # Also from DB key_finding
    confirmed_db = [h for h in hyps if h.get("verdict") == "confirmed"]
    confirmed_db.sort(key=lambda x: -(x.get("confidence") or 0))

    out = {
        "campaign_id": CAMPAIGN_ID,
        "campaign": {
            k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in campaign.items()
            if k != "hypothesis_tree_json"
        },
        "tree_summary": {
            "total_nodes": len(nodes),
            "frontier_size": len(tree.get("frontier") or []),
            "exhausted_count": len(tree.get("exhausted") or []),
            "confirmed_list_len": len(tree.get("confirmed") or []),
            "verdict_counts": dict(tree_verdicts),
            "max_depth": max((n.get("depth", 0) for n in nodes.values()), default=0),
            "generations": {str(g): len(v) for g, v in sorted(by_gen.items())},
        },
        "db_hypothesis_verdicts": dict(db_by_verdict),
        "event_counts_top": dict(list(event_counts.items())[:30]),
        "total_events": sum(event_counts.values()),
        "expansion_count": len(expansions),
        "expansion_reasons": dict(expand_reasons),
        "terminal_events": terminal,
        "seeds": [{"id": s.get("id"), "text": s.get("text"), "verdict": s.get("verdict")} for s in seeds],
        "root_hypotheses": [
            {"id": r.get("id"), "verdict": r.get("verdict"), "text": r.get("text")}
            for r in roots
        ],
        "seed_batch_from_llm": seed_batch,
        "confirmed_on_topic_count": len(on_topic_confirmed),
        "confirmed_off_topic_count": len(off_topic_confirmed),
        "confirmed_off_topic_sample": [
            {"text": n.get("text")} for n in off_topic_confirmed[:25]
        ],
        "timing": {
            "first_event": t0.isoformat() if t0 else None,
            "last_event": t1.isoformat() if t1 else None,
            "wall_seconds": wall_sec,
            "llm_ms": llm_ms,
            "experiment_step_ms": step_ms,
        },
        "confirmed_tree_sample": [
            {
                "id": n.get("id"),
                "generation": n.get("generation"),
                "depth": n.get("depth"),
                "confidence": n.get("confidence"),
                "text": n.get("text"),
                "evidence_summary": (n.get("evidence_summary") or "")[:500],
            }
            for n in confirmed_tree[:120]
        ],
        "confirmed_unique_themes": [
            {"text": n.get("text"), "generation": n.get("generation"), "confidence": n.get("confidence")}
            for n in list(confirmed_unique.values())[:80]
        ],
        "refuted_sample": [
            {"text": n.get("text"), "generation": n.get("generation"), "evidence_summary": (n.get("evidence_summary") or "")[:300]}
            for n in refuted_tree[:40]
        ],
        "inconclusive_sample": [
            {"text": n.get("text"), "generation": n.get("generation")}
            for n in inconclusive_tree[:30]
        ],
        "pending_count": len(pending_tree),
        "confirmed_db_key_findings": [
            {
                "text": (h.get("text") or "")[:400],
                "key_finding": h.get("key_finding"),
                "confidence": h.get("confidence"),
                "evidence_summary": (h.get("evidence_summary") or "")[:400],
            }
            for h in confirmed_db[:80]
        ],
        "verified_code_samples": verified_samples[:15],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"nodes={len(nodes)} confirmed_tree={len(confirmed_tree)} refuted={len(refuted_tree)} pending={len(pending_tree)}")
    print(f"budget {campaign['compute_seconds_used']}/{campaign['compute_budget_seconds']} status={campaign['status']}")


if __name__ == "__main__":
    main()
