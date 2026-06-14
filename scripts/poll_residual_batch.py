#!/usr/bin/env python3
"""Poll all campaigns in a residual batch until terminal or timeout."""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

TERMINAL = frozenset({"budget_exhausted", "breakthrough", "completed", "failed", "cancelled", "paused"})


def main() -> int:
    batch_file = sys.argv[1] if len(sys.argv) > 1 else "artifacts/residual_batch_latest.json"
    api = (sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000").rstrip("/")
    timeout_s = int(sys.argv[3]) if len(sys.argv) > 3 else 14400
    interval = int(sys.argv[4]) if len(sys.argv) > 4 else 180

    data = json.loads(Path(batch_file).read_text(encoding="utf-8"))
    campaigns = [c["campaign_id"] for c in data.get("campaigns") or []]
    if not campaigns:
        print("No campaigns in batch file", file=sys.stderr)
        return 1

    t0 = time.time()
    done: set[str] = set()
    while time.time() - t0 < timeout_s and len(done) < len(campaigns):
        for cid in campaigns:
            if cid in done:
                continue
            try:
                with urllib.request.urlopen(f"{api}/campaigns/{cid}", timeout=60) as resp:
                    body = json.loads(resp.read())
                camp = body.get("campaign") or body
                status = camp.get("status") or "unknown"
                tested = camp.get("total_hypotheses")
                confirmed = camp.get("total_confirmed")
                compute_s = camp.get("compute_seconds_used")
                print(
                    f"[{int(time.time()-t0)}s] {cid[:8]}… status={status} "
                    f"tested={tested} confirmed={confirmed} compute_s={compute_s}",
                    flush=True,
                )
                if status in TERMINAL:
                    done.add(cid)
            except Exception as exc:  # noqa: BLE001
                print(f"  poll error {cid[:8]}: {exc}", flush=True)
        if len(done) < len(campaigns):
            time.sleep(interval)

    print(json.dumps({"finished": len(done), "total": len(campaigns), "ids": list(done)}, indent=2))
    return 0 if len(done) == len(campaigns) else 2


if __name__ == "__main__":
    raise SystemExit(main())
