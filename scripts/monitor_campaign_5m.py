#!/usr/bin/env python3
"""Poll campaign every N seconds with synthesis/belief summary for live monitoring."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def _payload(ev: dict) -> dict:
    raw = ev.get("payload_json") or ev.get("payload") or {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_id")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between polls (default 5 min)")
    parser.add_argument("--max-polls", type=int, default=30, help="~2.5h at 5min intervals")
    parser.add_argument("--log", default="artifacts/mandrake_campaign_live.txt")
    args = parser.parse_args()

    base = args.api.rstrip("/")
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cid = args.campaign_id

    for i in range(args.max_polls):
        ts = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = ""
        try:
            d = _get(f"{base}/campaigns/{cid}")
            s = d.get("summary") or {}
            c = d.get("campaign") or {}
            rs = d.get("research_session") or {}
            tree = s.get("tree") or {}
            verdicts = tree.get("verdict_counts") or {}
            ev_counts = d.get("event_counts_by_type") or {}

            # Latest synthesis from events (cheap tail)
            synth_n = 0
            last_beliefs: list[str] = []
            failed = False
            fail_msg = ""
            try:
                ev_data = _get(f"{base}/sessions/{cid}/events?limit=2000")
                evs = ev_data.get("events") if isinstance(ev_data, dict) else ev_data
                if isinstance(evs, list):
                    for e in evs:
                        if e.get("step") == "campaign.synthesis":
                            synth_n += 1
                            active = _payload(e).get("active_beliefs") or []
                            last_beliefs = [
                                str(b.get("statement") or "")[:100]
                                for b in active if isinstance(b, dict)
                            ]
                        if e.get("step") == "campaign.failed" or e.get("event_type") == "session.failed":
                            failed = True
                            fail_msg = str(_payload(e).get("traceback") or "")[-200:]
            except Exception:
                pass

            elapsed = int(s.get("elapsed_sec") or 0)
            remaining = int(s.get("remaining_sec") or 0)
            status = c.get("status")
            line = (
                f"[{ts}] poll={i} status={status} elapsed={elapsed}s remaining={remaining}s "
                f"hyps={s.get('total_hypotheses')} "
                f"confirmed={tree.get('confirmed_count')} "
                f"refuted={verdicts.get('refuted')} inconclusive={verdicts.get('inconclusive')} "
                f"pending={verdicts.get('pending')} "
                f"frontier={tree.get('frontier_size')} "
                f"synthesis={synth_n} scope_gate={ev_counts.get('hypothesis.scope_gate', 0)} "
                f"failed={failed}"
            )
            if last_beliefs:
                line += f" beliefs={last_beliefs!r}"
            if fail_msg:
                line += f" err=...{fail_msg}"
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

            if failed:
                return 1
            sess_status = rs.get("status")
            if sess_status == "completed" and c.get("status") == "active":
                print(
                    f"STALE_CAMPAIGN_ROW session=completed campaign=active "
                    f"(run scripts/investigate_campaign.py {cid})",
                    flush=True,
                )
                return 1
            if status != "active":
                print(f"DONE status={status}", flush=True)
                return 0
            if remaining <= 0 and elapsed > 60 and sess_status != "running":
                print("BUDGET EXHAUSTED", flush=True)
                return 0
        except Exception as exc:
            line = f"[{ts}] poll={i} ERROR {exc}"
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        if i + 1 < args.max_polls:
            time.sleep(args.interval)

    print("TIMEOUT max polls reached", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
