#!/usr/bin/env python3
"""Mark a campaign paused in Postgres and restart the API to stop the in-process loop."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

import psycopg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--campaign-id", help="UUID (default: read from --state-file)")
    p.add_argument("--state-file", help="JSON from start_*_campaign.py")
    p.add_argument("--db-url", default="postgresql://propab:propab@localhost:5432/propab")
    p.add_argument("--no-restart-api", action="store_true")
    args = p.parse_args()

    cid = args.campaign_id
    if not cid and args.state_file:
        cid = json.loads(Path(args.state_file).read_text(encoding="utf-8"))["campaign_id"]
    if not cid:
        print("Provide --campaign-id or --state-file", file=sys.stderr)
        return 1

    with psycopg.connect(args.db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE research_campaigns SET status = 'paused' WHERE id = %s::uuid",
                (cid,),
            )
            cur.execute(
                """
                UPDATE research_sessions
                SET status = 'completed', stage = 'paused', completed_at = NOW()
                WHERE id = %s::uuid
                """,
                (cid,),
            )
        conn.commit()

    print(f"Paused campaign {cid} in database.", flush=True)

    if not args.no_restart_api:
        subprocess.run(
            ["docker", "compose", "restart", "api"],
            cwd=str(Path(__file__).resolve().parents[1]),
            check=False,
        )
        print("Restarted propab-oss-api to stop the background campaign task.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
