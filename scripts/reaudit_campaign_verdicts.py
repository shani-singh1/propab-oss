#!/usr/bin/env python3
"""Replay campaign hypothesis evidence through run_verdict_pipeline (fixes.md Task 6)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.verdict_pipeline import run_verdict_pipeline  # noqa: E402


def _parse_evidence_blob(summary: str | None) -> dict:
    if not summary:
        return {}
    m = re.search(r"evidence=(\{.*?\});", summary, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-id", required=True)
    p.add_argument("--db-url", default="postgresql://propab:propab@localhost:5432/propab")
    p.add_argument("--out", default=str(ROOT / "artifacts" / "verdict_pipeline_reaudit.json"))
    args = p.parse_args()

    stored_verdicts: dict[str, str] = {}
    try:
        with urlopen(f"http://localhost:8000/campaigns/{args.campaign_id}", timeout=60) as resp:
            data = json.loads(resp.read())
        nodes = (data.get("campaign") or {}).get("hypothesis_tree", {}).get("nodes", {})
        for nid, node in nodes.items():
            if isinstance(node, dict) and node.get("verdict"):
                stored_verdicts[str(nid)] = str(node["verdict"])
    except Exception:
        nodes = {}

    rows: list[dict] = []
    with psycopg.connect(args.db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, text, verdict, evidence_summary
                FROM hypotheses
                WHERE session_id = %s::uuid
                ORDER BY created_at
                """,
                (args.campaign_id,),
            )
            for hid, text, stored, summary in cur.fetchall():
                ev = _parse_evidence_blob(summary)
                if not ev:
                    continue
                ev_copy = json.loads(json.dumps(ev))
                new_verdict, confidence, reason = run_verdict_pipeline(ev_copy)
                rows.append({
                    "hypothesis_id": hid,
                    "text": (text or "")[:200],
                    "stored_verdict": stored,
                    "replayed_verdict": new_verdict,
                    "confidence": confidence,
                    "reason": reason[:300],
                    "changed": stored != new_verdict,
                    "would_confirm": new_verdict == "confirmed" and stored != "confirmed",
                })

    would_confirm = sum(1 for r in rows if r["would_confirm"])
    changed = sum(1 for r in rows if r["changed"])
    report = {
        "campaign_id": args.campaign_id,
        "hypotheses_replayed": len(rows),
        "verdict_changes": changed,
        "would_newly_confirm": would_confirm,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "campaign_id": args.campaign_id,
        "hypotheses_replayed": len(rows),
        "verdict_changes": changed,
        "would_newly_confirm": would_confirm,
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
