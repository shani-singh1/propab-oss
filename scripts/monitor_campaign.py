#!/usr/bin/env python3
"""
Poll campaign progress via GET /campaigns/{id} and session snapshot.

Reads campaign_id from --state-file (written by start_campaign_v1.py) or --campaign-id.

Example:
  python scripts/monitor_campaign.py --state-file artifacts/campaign_v1_latest.json
  python scripts/monitor_campaign.py --campaign-id <uuid> --interval 60 --log artifacts/campaign_v1_live.txt

If GET /campaigns returns 500, ensure Postgres has migrations/006_campaigns.sql applied:
  docker compose exec -T postgres psql -U propab -d propab -f - < migrations/006_campaigns.sql

Uses the same UUID for GET /sessions/{id}/events (campaign_id == session_id). Pass --event-tail 0
to skip the extra request; default fetches recent agent/tool/LLM events each poll.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    from services.worker.failure_classify import compact_failure_summary
except ImportError:

    def compact_failure_summary(payload: dict) -> str:
        return ""


def _event_payload(ev: dict) -> dict:
    raw = ev.get("payload_json", ev.get("payload"))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return dict(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def get_json(url: str, timeout: float = 45.0) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def fmt_summary(blob: dict) -> str:
    s = blob.get("summary") or {}
    sess = blob.get("research_session") or {}
    tree = s.get("tree") or {}
    lines = [
        f"session_status={sess.get('status', '?')} stage={sess.get('stage', '?')}",
        f"campaign_status={s.get('status', '?')}",
        f"hypotheses_tested={s.get('total_hypotheses', 0)} confirmed={s.get('total_confirmed', 0)}",
        f"baseline_metric={s.get('baseline_metric')} best_metric={s.get('best_metric')} "
        f"improvement_pct={s.get('improvement_pct')}% (threshold {s.get('breakthrough_threshold_pct')}%)",
        f"elapsed_sec={s.get('elapsed_sec')} remaining_sec={s.get('remaining_sec')}",
        f"tree_nodes={tree.get('total_nodes')} frontier={tree.get('frontier_size')} "
        f"max_depth={tree.get('max_depth')}",
    ]
    ev = blob.get("event_counts_by_type") or {}
    if ev:
        interesting = (
            "campaign.started",
            "campaign.progress",
            "campaign.breakthrough",
            "campaign.budget_exhausted",
            "campaign.completed",
            "campaign.baseline_measured",
            "hypothesis.tree_expanded",
            "hypothesis.tree_frontier_empty",
            "hypothesis.dispatched",
            "agent.started",
            "agent.time_budget_exceeded",
            "agent.step_started",
            "agent.step_completed",
            "agent.completed",
            "agent.failed",
            "code.timeout",
            "tool.error",
            "tool.called",
            "tool.result",
            "llm.prompt",
            "literature.fetch_started",
        )
        parts = [f"{k}:{ev[k]}" for k in interesting if k in ev]
        if parts:
            lines.append("events: " + ", ".join(parts))
        # Second line: highest-volume event types (so agent/tool noise is visible even
        # when campaign DB totals are still zero during long baseline / prior phases).
        top = sorted(ev.items(), key=lambda kv: (-kv[1], kv[0]))[:14]
        lines.append("events_top: " + ", ".join(f"{k}:{v}" for k, v in top))
    return " | ".join(lines)


def _short(s: str, n: int = 72) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _compact_step(step: str, max_len: int = 44) -> str:
    """Shorten experiment.<uuid>... noise in log lines."""
    s = (step or "").replace("\n", " ").strip()
    if "experiment." in s and len(s) > max_len:
        s = re.sub(
            r"(experiment\.)([0-9a-f]{8})-[0-9a-f-]{27}",
            r"\1\2..",
            s,
            count=1,
            flags=re.I,
        )
    return _short(s, max_len)


def fmt_recent_events(events: list[dict], max_show: int) -> str:
    """One-line digest of the last N session events (agent/tool/LLM steps)."""
    if not events or max_show <= 0:
        return ""
    tail = events[-max_show:]
    parts: list[str] = []
    for ev in tail:
        et = str(ev.get("event_type") or "?")
        step = str(ev.get("step") or "")
        src = str(ev.get("source") or "")
        hid = ev.get("hypothesis_id")
        hid_s = (str(hid)[:8] + "..") if hid else "-"
        extra = ""
        pl = _event_payload(ev)
        if et == "agent.failed" and pl:
            extra = compact_failure_summary(pl)
        elif et == "tool.error" and pl:
            fk = str(pl.get("failure_kind") or "?")
            err = pl.get("error") if isinstance(pl.get("error"), dict) else {}
            etyp = str((err or {}).get("type") or "")[:48]
            extra = f"[{fk}" + (f"|{etyp}]" if etyp else "]")
        elif et in ("code.timeout", "code.error") and pl:
            fk = str(pl.get("failure_kind") or et)
            extra = f"[{fk}"
            ts = pl.get("timeout_sec")
            if ts is not None:
                extra += f"|sandbox_sec={ts}"
            extra += "]"
        elif et == "agent.time_budget_exceeded" and pl:
            extra = f"[budget_sec={pl.get('agent_max_seconds')} steps={pl.get('steps_taken')}]"
        parts.append(f"{et}@{_compact_step(step, 48)}({src},{hid_s}){extra}")
    return "recent: " + " | ".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor a Propab research campaign.")
    p.add_argument("--api", default="http://localhost:8000", help="API base URL")
    p.add_argument("--campaign-id", default="", help="Campaign UUID")
    p.add_argument(
        "--state-file",
        default="",
        help="JSON file with campaign_id (from start_campaign_v1.py)",
    )
    p.add_argument("--interval", type=float, default=30.0, help="Seconds between polls")
    p.add_argument("--once", action="store_true", help="Single snapshot then exit")
    p.add_argument("--log", default="", help="Append timestamped lines to this file")
    p.add_argument(
        "--event-tail",
        type=int,
        default=18,
        help="Also fetch last N session events (GET /sessions/{id}/events?limit=N) "
        "and log a one-line digest (0 to disable).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api = args.api.rstrip("/")

    cid = args.campaign_id.strip()
    if not cid and args.state_file:
        path = Path(args.state_file)
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                cid = json.load(f).get("campaign_id") or ""
    if not cid:
        print("Provide --campaign-id or --state-file with campaign_id.", file=sys.stderr)
        sys.exit(2)

    log_fh = open(args.log, "a", encoding="utf-8") if args.log else None

    def tee(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line)
        if log_fh:
            log_fh.write(line + "\n")
            log_fh.flush()

    url = f"{api}/campaigns/{cid}"
    events_url = f"{api}/sessions/{cid}/events"
    tee(f"Monitoring campaign {cid}")
    tee(f"GET {url}")
    if args.event_tail > 0:
        tee(f"GET {events_url}?limit={args.event_tail} (recent agent/tool steps)")

    last_event_id: str | None = None
    try:
        while True:
            try:
                blob = get_json(url)
                tee(fmt_summary(blob))
                if args.event_tail > 0:
                    ev_blob = get_json(f"{events_url}?limit={args.event_tail}")
                    evs = ev_blob.get("events") or []
                    if evs:
                        cur_id = str(evs[-1].get("id") or "")
                        if cur_id != last_event_id:
                            tee(fmt_recent_events(evs, args.event_tail))
                            last_event_id = cur_id
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:1500]
                tee(f"HTTP {exc.code} — {body}")
            except URLError as exc:
                tee(f"poll error: {exc}")
            except Exception as exc:  # noqa: BLE001
                # Handles transient disconnects (e.g. RemoteDisconnected) without killing monitor.
                tee(f"poll error: {type(exc).__name__}: {exc}")

            if args.once:
                break
            time.sleep(max(5.0, args.interval))
    finally:
        if log_fh:
            log_fh.close()


if __name__ == "__main__":
    main()
