#!/usr/bin/env python3
"""Poll campaign until complete or budget exhausted."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_id")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--max-polls", type=int, default=90)
    args = parser.parse_args()

    for i in range(args.max_polls):
        try:
            with urllib.request.urlopen(f"{args.api.rstrip('/')}/campaigns/{args.campaign_id}", timeout=15) as r:
                d = json.load(r)
            s = d["summary"]
            c = d["campaign"]
            ev = d.get("event_counts_by_type", {})
            sg = ev.get("hypothesis.scope_gate", 0)
            print(
                f"{i}: status={c['status']} hyps={s['total_hypotheses']} "
                f"confirmed={s['tree']['confirmed_count']} elapsed={int(s['elapsed_sec'])}s "
                f"scope_gate={sg}",
                flush=True,
            )
            if c["status"] != "active":
                print(f"DONE status={c['status']}", flush=True)
                return 0
            if s["remaining_sec"] <= 0:
                print("BUDGET EXHAUSTED", flush=True)
                return 0
        except Exception as exc:
            print(f"err {exc}", flush=True)
        time.sleep(args.interval)
    print("TIMEOUT", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
