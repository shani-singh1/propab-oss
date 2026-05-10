from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Poll research sessions until completed or failed.")
    p.add_argument("--api", default="http://localhost:8000", help="Propab API base URL")
    p.add_argument(
        "--input",
        default="artifacts/run_5q_sessions.json",
        help="JSON list of {id, session_id, status?}; updated in place",
    )
    p.add_argument(
        "--live-log",
        default="",
        help="Append stdout lines with timestamps to this file (optional)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api = args.api.rstrip("/")

    class Tee:
        def __init__(self, path: str | None) -> None:
            self.path = path
            self._fh = open(path, "a", encoding="utf-8") if path else None

        def log(self, line: str) -> None:
            print(line)
            if self._fh:
                self._fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}\n")
                self._fh.flush()

        def close(self) -> None:
            if self._fh:
                self._fh.close()

    tee = Tee(args.live_log or None)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            sessions: list[dict] = json.load(f)
    except OSError as exc:
        tee.log(f"ERROR: cannot read {args.input}: {exc}")
        sys.exit(1)

    tee.log("=== Monitoring existing sessions ===")
    tee.log(f"Loaded {len(sessions)} sessions from {args.input}")

    try:
        while True:
            all_done = True
            tee.log(f"\n--- Status check {time.strftime('%H:%M:%S')} ---")
            for s in sessions:
                sid = s.get("session_id")
                qid = s.get("id", "?")
                if not sid:
                    continue
                if s.get("status") in ("completed", "failed"):
                    tee.log(f"  {qid}: {s['status']}")
                    continue
                try:
                    data = get_json(f"{api}/sessions/{sid}")
                    status = data.get("status", "unknown")
                    stage = data.get("stage", "")
                    s["status"] = status
                    tee.log(f"  {qid}: status={status} stage={stage}")
                    if status not in ("completed", "failed"):
                        all_done = False
                except Exception as exc:
                    tee.log(f"  {qid}: poll error={exc}")
                    all_done = False

            with open(args.input, "w", encoding="utf-8") as f:
                json.dump(sessions, f, indent=2)

            if all_done:
                tee.log("\n=== All sessions finished ===")
                break

            tee.log("  (waiting 30s...)")
            time.sleep(30)
    finally:
        tee.close()


if __name__ == "__main__":
    main()
