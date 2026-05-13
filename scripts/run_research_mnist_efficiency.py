"""
Submit the MNIST parameter-efficiency research session from fixes.md.

Requires stack running with PROPAB_PROFILE=dev or campaign (see docker-compose env).

Example:
  set PROPAB_PROFILE=dev & docker compose up -d --build
  python scripts/run_research_mnist_efficiency.py

Then in another terminal:
  python scripts/monitor_existing_sessions.py \\
    --input artifacts/run_research_mnist_sessions.json \\
    --live-log artifacts/run_research_mnist_live_monitor.txt
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

API = "http://localhost:8000"
SESSIONS_OUT = "artifacts/run_research_mnist_sessions.json"

QUESTION = (
    "What is the most parameter-efficient MLP architecture for MNIST classification "
    "under a 50,000 parameter budget?"
)


def post_json(url: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main() -> None:
    print(
        "Reminder: api/orchestrator/worker should run with PROPAB_PROFILE=dev (fast) or campaign "
        "(research_max_rounds=4 from profile; POST uses max_hypotheses=8).\n"
    )
    payload = {
        "question": QUESTION,
        "config": {
            "max_hypotheses": 8,
            "paper_ttl_days": 30,
        },
    }
    print(f"POST {API}/research ...")
    try:
        resp = post_json(f"{API}/research", payload)
    except urllib.error.URLError as exc:
        print(f"ERROR: could not reach API ({exc}). Start compose and retry.", file=sys.stderr)
        sys.exit(1)
    sid = resp["session_id"]
    print(f"session_id={sid}\nstream_url={resp.get('stream_url')}")

    sessions = [{"id": "MNIST-EFF", "session_id": sid, "status": "submitted"}]
    with open(SESSIONS_OUT, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)
    print(f"\nWrote {SESSIONS_OUT}")
    print(
        "\nMonitor (optional tee):\n"
        f"  python scripts/monitor_existing_sessions.py "
        f"--input {SESSIONS_OUT} "
        f"--live-log artifacts/run_research_mnist_live_monitor.txt"
    )


if __name__ == "__main__":
    main()
