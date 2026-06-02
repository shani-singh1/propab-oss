#!/usr/bin/env python3
"""Poll campaign until terminal status or timeout."""
from __future__ import annotations

import json
import sys
import time
import urllib.request

TERMINAL = frozenset({"budget_exhausted", "breakthrough", "completed", "failed", "cancelled"})


def main() -> None:
    cid = sys.argv[1]
    api = (sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000").rstrip("/")
    timeout_s = int(sys.argv[3]) if len(sys.argv) > 3 else 2400
    interval = int(sys.argv[4]) if len(sys.argv) > 4 else 90
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        with urllib.request.urlopen(f"{api}/campaigns/{cid}", timeout=60) as resp:
            data = json.loads(resp.read())
        camp = data.get("campaign") or data
        summary = data.get("summary") or {}
        status = camp.get("status") or summary.get("status") or "unknown"
        tested = summary.get("total_hypotheses") or camp.get("total_hypotheses")
        confirmed = summary.get("total_confirmed") or camp.get("total_confirmed")
        budget = summary.get("compute_seconds_used") or camp.get("compute_seconds_used")
        print(
            f"[{int(time.time()-t0)}s] status={status} tested={tested} confirmed={confirmed} compute_s={budget}",
            flush=True,
        )
        if status in TERMINAL:
            print(json.dumps({"status": status, "summary": summary, "campaign": camp}, indent=2))
            sys.exit(0)
        time.sleep(interval)
    print("TIMEOUT", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
