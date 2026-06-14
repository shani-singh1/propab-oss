#!/usr/bin/env python3
"""
Level 1 residual export (fixes.md) — (policy, prediction, actual, residual) from Postgres.

Reads lifetime.ingested events; avoids depending on JSON file-store merges.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import text

from propab.config import settings
from propab.db import create_engine, create_session_factory

ROOT = Path(__file__).resolve().parents[1]


async def _export(*, campaign_ids: list[str] | None, out: Path) -> int:
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    rows_out: list[dict] = []

    async with session_factory() as db:
        if campaign_ids:
            for cid in campaign_ids:
                row = (
                    await db.execute(
                        text("""
                            SELECT e.session_id, e.payload_json, e.created_at,
                                   c.question, c.status, c.total_hypotheses, c.total_confirmed
                            FROM events e
                            LEFT JOIN research_campaigns c ON c.id = e.session_id
                            WHERE e.session_id = CAST(:id AS uuid)
                              AND e.step = 'lifetime.ingested'
                            ORDER BY e.created_at DESC
                            LIMIT 1
                        """),
                        {"id": cid},
                    )
                ).mappings().first()
                if row:
                    rows_out.append(_row_to_residual(row))
        else:
            result = await db.execute(
                text("""
                    SELECT DISTINCT ON (e.session_id)
                           e.session_id, e.payload_json, e.created_at,
                           c.question, c.status, c.total_hypotheses, c.total_confirmed
                    FROM events e
                    LEFT JOIN research_campaigns c ON c.id = e.session_id
                    WHERE e.step = 'lifetime.ingested'
                    ORDER BY e.session_id, e.created_at DESC
                """),
            )
            for row in result.mappings():
                rows_out.append(_row_to_residual(row))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"residuals": rows_out, "count": len(rows_out)}, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(rows_out), "out": str(out)}, indent=2))
    await engine.dispose()
    return 0


def _row_to_residual(row) -> dict:
    payload = row["payload_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    elif payload is None:
        payload = {}
    ev = payload.get("evaluation") or {}
    return {
        "campaign_id": str(row["session_id"]),
        "recorded_at": str(row["created_at"]),
        "question": (row.get("question") or "")[:200],
        "status": row.get("status"),
        "tested": row.get("total_hypotheses"),
        "confirmed": row.get("total_confirmed"),
        "policy_id": ev.get("policy_id") or payload.get("active_policy_id"),
        "budget_bucket": payload.get("budget_bucket"),
        "domain_bucket": payload.get("domain_bucket"),
        "prediction": ev.get("predicted"),
        "actual": ev.get("observed"),
        "residual": ev.get("residuals"),
        "tolerance": ev.get("tolerance"),
        "accept_or_reject": ev.get("accept_or_reject"),
        "pred_ok": ev.get("pred_ok"),
        "closure_ok": ev.get("closure_ok"),
        "entropy_eval_mode": ev.get("entropy_eval_mode"),
        "observed_trajectory": ev.get("observed_trajectory"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-file",
        help="JSON from launch_residual_batch.py (exports those campaign IDs)",
    )
    parser.add_argument(
        "--campaign-id",
        action="append",
        dest="campaign_ids",
        help="Explicit campaign UUID (repeatable)",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "policy_residuals.json"),
    )
    args = parser.parse_args()

    ids: list[str] | None = None
    if args.batch_file:
        data = json.loads(Path(args.batch_file).read_text(encoding="utf-8"))
        ids = [c["campaign_id"] for c in data.get("campaigns") or []]
    elif args.campaign_ids:
        ids = args.campaign_ids

    return asyncio.run(_export(campaign_ids=ids, out=Path(args.out)))


if __name__ == "__main__":
    raise SystemExit(main())
