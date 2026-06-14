#!/usr/bin/env python3
"""Extract within-campaign theme_entropy trajectories from frontier_snapshot events."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from sqlalchemy import text  # noqa: E402

from propab.config import settings  # noqa: E402
from propab.db import create_engine, create_session_factory  # noqa: E402


from propab.entropy_trajectory import summarize_entropy_trajectory  # noqa: E402


async def extract(campaign_ids: list[str]) -> dict:
    eng = create_engine(settings.database_url)
    sf = create_session_factory(eng)
    out: dict = {"campaigns": [], "baseline_campaign_id": None}
    async with sf() as db:
        for cid in campaign_ids:
            ingest = (
                await db.execute(
                    text(
                        """
                        SELECT payload_json FROM events
                        WHERE session_id = CAST(:id AS uuid)
                          AND step = 'lifetime.ingested'
                        ORDER BY created_at DESC LIMIT 1
                        """
                    ),
                    {"id": cid},
                )
            ).scalar_one_or_none()
            baseline_id = None
            end_entropy = None
            if ingest:
                p = ingest if isinstance(ingest, dict) else json.loads(ingest)
                ev = p.get("evaluation") or {}
                baseline_id = ev.get("baseline_campaign_id")
                obs = p.get("observations")
                if isinstance(obs, dict):
                    end_entropy = obs.get("theme_entropy")
                if end_entropy is None and ev.get("observed"):
                    # delta only in evaluation; reconstruct if we have baseline later
                    pass

            rows = (
                await db.execute(
                    text(
                        """
                        SELECT created_at, payload_json
                        FROM events
                        WHERE session_id = CAST(:id AS uuid)
                          AND step = 'campaign.frontier_snapshot'
                        ORDER BY created_at ASC
                        """
                    ),
                    {"id": cid},
                )
            ).fetchall()

            points = []
            for row in rows:
                payload = row[1] if isinstance(row[1], dict) else json.loads(row[1])
                points.append(
                    {
                        "ts": str(row[0]),
                        "tested": payload.get("tested") or payload.get("total_tested"),
                        "theme_entropy": float(payload.get("theme_entropy") or 0),
                        "top_theme": payload.get("top_theme"),
                        "closure_ratio": payload.get("closure_ratio"),
                    }
                )

            summary = summarize_entropy_trajectory(points)
            campaign = {
                "campaign_id": cid,
                "n_snapshots": len(points),
                "shape": summary.growth_pattern,
                "start_entropy": summary.H_start,
                "mid_entropy": summary.H_mid,
                "end_snapshot_entropy": summary.H_end,
                "end_ingest_entropy": end_entropy,
                "baseline_campaign_id": baseline_id,
                "cross_entropy_1_5_at_tested": summary.cross_H_1_5_at_tested,
                "cross_entropy_2_0_at_tested": summary.cross_H_2_0_at_tested,
                "plateau_at_tested": summary.plateau_at_tested,
                "growth_rate": summary.growth_rate,
                "trajectory": points,
            }
            out["campaigns"].append(campaign)
            if baseline_id and not out["baseline_campaign_id"]:
                out["baseline_campaign_id"] = baseline_id

        if out["baseline_campaign_id"]:
            bid = out["baseline_campaign_id"]
            row = (
                await db.execute(
                    text(
                        """
                        SELECT payload_json FROM events
                        WHERE session_id = CAST(:id AS uuid)
                          AND step = 'lifetime.ingested'
                        ORDER BY created_at DESC LIMIT 1
                        """
                    ),
                    {"id": bid},
                )
            ).scalar_one_or_none()
            if row:
                p = row if isinstance(row, dict) else json.loads(row)
                obs = p.get("observations")
                if isinstance(obs, dict):
                    out["baseline_theme_entropy"] = obs.get("theme_entropy")
                    out["baseline_closure_ratio"] = obs.get("closure_ratio")

    await eng.dispose()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        default=str(ROOT / "artifacts" / "residual_batch_latest.json"),
    )
    parser.add_argument(
        "--outfile",
        default=str(ROOT / "artifacts" / "entropy_trajectories.json"),
    )
    args = parser.parse_args()
    batch = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    ids = [c["campaign_id"] for c in batch.get("campaigns") or []]
    if not ids:
        print("No campaign IDs in infile", file=sys.stderr)
        return 1

    report = asyncio.run(extract(ids))
    Path(args.outfile).write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = []
    for c in report["campaigns"]:
        summary.append(
            {
                "id": c["campaign_id"][:8],
                "shape": c["shape"],
                "H_start": c["start_entropy"],
                "H_mid": c["mid_entropy"],
                "H_end_snap": c["end_snapshot_entropy"],
                "H_end_ingest": c["end_ingest_entropy"],
                "cross_1.5": c["cross_entropy_1_5_at_tested"],
                "cross_2.0": c["cross_entropy_2_0_at_tested"],
                "n_snaps": c["n_snapshots"],
            }
        )
    print(json.dumps({"baseline": report.get("baseline_theme_entropy"), "runs": summary}, indent=2))
    print(f"Wrote {args.outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
