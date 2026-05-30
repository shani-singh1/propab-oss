#!/usr/bin/env python3
"""Post-run timing audit for a Propab campaign.

Pulls events, experiment trace (duration_ms per step), and LLM calls from the API and
writes a human-readable report + JSON artifact identifying where wall-clock time went.

Usage::

    python scripts/audit_campaign_timing.py --state-file artifacts/e2e_campaign_latest.json
    python scripts/audit_campaign_timing.py --campaign-id <uuid> --out artifacts/e2e_timing_audit.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

_REPO = Path(__file__).resolve().parents[1]


def get_json(url: str, timeout: float = 120.0) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _sec(a: datetime | None, b: datetime | None) -> float | None:
    if a is None or b is None:
        return None
    return max(0.0, (b - a).total_seconds())


def _fmt_sec(s: float | None) -> str:
    if s is None:
        return "n/a"
    if s < 60:
        return f"{s:.1f}s"
    return f"{s / 60:.1f}m ({s:.0f}s)"


def _payload(ev: dict) -> dict:
    raw = ev.get("payload_json", ev.get("payload"))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return dict(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _phase_milestones(events: list[dict]) -> dict[str, datetime | None]:
    """Map logical campaign phases to first-seen timestamps."""
    out: dict[str, datetime | None] = {
        "campaign_started": None,
        "prior_build_start": None,
        "baseline_start": None,
        "baseline_measured": None,
        "first_hypothesis_dispatched": None,
        "first_agent_started": None,
        "first_agent_completed": None,
        "paper_started": None,
        "paper_ready": None,
        "campaign_terminal": None,
    }
    for ev in events:
        et = str(ev.get("event_type") or "")
        ts = _parse_ts(ev.get("created_at"))
        if ts is None:
            continue
        pl = _payload(ev)
        if et == "campaign.started" and out["campaign_started"] is None:
            out["campaign_started"] = ts
        if et == "campaign.progress":
            phase = str(pl.get("phase") or "")
            if phase == "prior_build" and out["prior_build_start"] is None:
                out["prior_build_start"] = ts
            if phase == "baseline_measure" and out["baseline_start"] is None:
                out["baseline_start"] = ts
        if et == "campaign.baseline_measured" and out["baseline_measured"] is None:
            out["baseline_measured"] = ts
        if et == "hypothesis.dispatched" and out["first_hypothesis_dispatched"] is None:
            out["first_hypothesis_dispatched"] = ts
        if et == "agent.started" and out["first_agent_started"] is None:
            out["first_agent_started"] = ts
        if et == "agent.completed" and out["first_agent_completed"] is None:
            out["first_agent_completed"] = ts
        if et == "paper.section_started" and out["paper_started"] is None:
            out["paper_started"] = ts
        if et == "paper.ready" and out["paper_ready"] is None:
            out["paper_ready"] = ts
        if et in ("campaign.budget_exhausted", "campaign.breakthrough", "campaign.completed"):
            if out["campaign_terminal"] is None:
                out["campaign_terminal"] = ts
    return out


def _trace_breakdown(trace: list[dict]) -> dict[str, Any]:
    by_type: dict[str, list[int]] = defaultdict(list)
    by_tool: dict[str, list[int]] = defaultdict(list)
    for row in trace:
        st = str(row.get("step_type") or "unknown")
        ms = int(row.get("duration_ms") or 0)
        by_type[st].append(ms)
        if st == "tool_call":
            inp = row.get("input_json")
            tool = "?"
            if isinstance(inp, dict):
                tool = str(inp.get("tool") or "?")
            elif isinstance(inp, str):
                try:
                    tool = str(json.loads(inp).get("tool") or "?")
                except json.JSONDecodeError:
                    pass
            by_tool[tool].append(ms)

    def _sum_stats(vals: list[int]) -> dict[str, Any]:
        if not vals:
            return {"count": 0, "total_ms": 0, "mean_ms": 0, "max_ms": 0}
        return {
            "count": len(vals),
            "total_ms": sum(vals),
            "mean_ms": round(sum(vals) / len(vals)),
            "max_ms": max(vals),
        }

    tool_rank = sorted(
        ((t, _sum_stats(v)) for t, v in by_tool.items()),
        key=lambda x: -x[1]["total_ms"],
    )
    return {
        "by_step_type": {k: _sum_stats(v) for k, v in sorted(by_type.items())},
        "top_tools_by_total_ms": [
            {"tool": t, **stats} for t, stats in tool_rank[:15]
        ],
    }


def _llm_breakdown(calls: list[dict]) -> dict[str, Any]:
    by_purpose: dict[str, list[int]] = defaultdict(list)
    for c in calls:
        purpose = str(c.get("call_purpose") or "unknown")
        ms = int(c.get("duration_ms") or 0)
        by_purpose[purpose].append(ms)

    def _stats(vals: list[int]) -> dict[str, Any]:
        if not vals:
            return {"count": 0, "total_ms": 0}
        return {"count": len(vals), "total_ms": sum(vals), "mean_ms": round(sum(vals) / len(vals))}

    ranked = sorted(
        ((p, _stats(v)) for p, v in by_purpose.items()),
        key=lambda x: -x[1]["total_ms"],
    )
    total_ms = sum(int(c.get("duration_ms") or 0) for c in calls)
    return {
        "total_calls": len(calls),
        "total_ms": total_ms,
        "by_purpose": {p: s for p, s in ranked},
        "top_purposes": [{"purpose": p, **s} for p, s in ranked[:12]],
    }


def _event_counts(events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ev in events:
        counts[str(ev.get("event_type") or "?")] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def build_audit(api: str, campaign_id: str) -> dict[str, Any]:
    api = api.rstrip("/")
    t_fetch = time.monotonic()

    session = get_json(f"{api}/sessions/{campaign_id}")
    events_blob = get_json(f"{api}/sessions/{campaign_id}/events")
    trace_blob = get_json(f"{api}/sessions/{campaign_id}/trace")
    llm_blob = get_json(f"{api}/sessions/{campaign_id}/llm-calls")
    campaign_blob = get_json(f"{api}/campaigns/{campaign_id}")

    events = events_blob.get("events") or []
    trace = trace_blob.get("trace") or []
    llm_calls = llm_blob.get("llm_calls") or []
    milestones = _phase_milestones(events)

    t0 = milestones.get("campaign_started") or _parse_ts(session.get("created_at"))
    t_end = (
        milestones.get("paper_ready")
        or milestones.get("campaign_terminal")
        or _parse_ts(session.get("completed_at"))
    )
    if events and t_end is None:
        t_end = _parse_ts(events[-1].get("created_at"))

    phases: dict[str, Any] = {}
    if t0 and milestones.get("baseline_start"):
        phases["prior_literature"] = _sec(t0, milestones["baseline_start"])
    if milestones.get("baseline_start") and milestones.get("baseline_measured"):
        phases["baseline_measurement"] = _sec(milestones["baseline_start"], milestones["baseline_measured"])
    if milestones.get("baseline_measured") and milestones.get("first_hypothesis_dispatched"):
        phases["pre_first_dispatch"] = _sec(
            milestones["baseline_measured"], milestones["first_hypothesis_dispatched"]
        )
    if milestones.get("first_agent_started") and milestones.get("first_agent_completed"):
        phases["first_agent_wall"] = _sec(
            milestones["first_agent_started"], milestones["first_agent_completed"]
        )
    if milestones.get("first_hypothesis_dispatched") and milestones.get("paper_started"):
        phases["experiment_wave_to_paper"] = _sec(
            milestones["first_hypothesis_dispatched"], milestones["paper_started"]
        )
    if milestones.get("paper_started") and milestones.get("paper_ready"):
        phases["paper_generation"] = _sec(milestones["paper_started"], milestones["paper_ready"])

    trace_bd = _trace_breakdown(trace)
    llm_bd = _llm_breakdown(llm_calls)
    ev_counts = _event_counts(events)

    tool_total_ms = sum(
        s["total_ms"] for s in trace_bd["by_step_type"].values()
    )
    llm_total_ms = llm_bd["total_ms"]
    wall_sec = _sec(t0, t_end)

    blockers: list[str] = []
    if phases.get("baseline_measurement") and phases["baseline_measurement"] > 600:
        blockers.append(
            f"Baseline measurement took {_fmt_sec(phases['baseline_measurement'])} — "
            "often dominates cold-start (Celery + MNIST train_model sub-agent)."
        )
    if ev_counts.get("code.timeout", 0) > 0:
        blockers.append(
            f"{ev_counts['code.timeout']} sandbox code timeout(s) — check Docker sandbox wall limits."
        )
    if ev_counts.get("agent.time_budget_exceeded", 0) > 0:
        blockers.append(
            f"{ev_counts['agent.time_budget_exceeded']} agent(s) hit wall-clock budget "
            "(settings.agent_max_seconds / PROPAB_PROFILE)."
        )
    if llm_total_ms > tool_total_ms and llm_total_ms > 60_000:
        blockers.append(
            f"LLM calls account for {_fmt_sec(llm_total_ms / 1000)} vs "
            f"{_fmt_sec(tool_total_ms / 1000)} in traced tool/code steps — LLM latency may dominate."
        )
    top_tool = trace_bd.get("top_tools_by_total_ms") or []
    if top_tool and top_tool[0]["total_ms"] > 120_000:
        blockers.append(
            f"Slowest tool: {top_tool[0]['tool']} ({_fmt_sec(top_tool[0]['total_ms'] / 1000)} total)."
        )

    summ = (campaign_blob.get("summary") or {})
    return {
        "campaign_id": campaign_id,
        "audited_at": datetime.now().isoformat(timespec="seconds"),
        "fetch_sec": round(time.monotonic() - t_fetch, 2),
        "session_status": session.get("status"),
        "session_stage": session.get("stage"),
        "campaign_status": summ.get("status"),
        "total_hypotheses": summ.get("total_hypotheses"),
        "total_confirmed": summ.get("total_confirmed"),
        "elapsed_sec_reported": summ.get("elapsed_sec"),
        "wall_clock_sec_estimated": wall_sec,
        "milestones_iso": {k: (v.isoformat() if v else None) for k, v in milestones.items()},
        "phase_durations_sec": phases,
        "accounted_ms": {
            "experiment_trace_total_ms": tool_total_ms,
            "llm_calls_total_ms": llm_total_ms,
            "unaccounted_sec_est": (
                None
                if wall_sec is None
                else max(0.0, wall_sec - (tool_total_ms + llm_total_ms) / 1000.0)
            ),
        },
        "trace_breakdown": trace_bd,
        "llm_breakdown": llm_bd,
        "event_counts": ev_counts,
        "likely_blockers": blockers,
    }


def render_report(audit: dict[str, Any]) -> str:
    lines = [
        f"=== Campaign timing audit: {audit['campaign_id']} ===",
        f"Session: {audit.get('session_status')} / stage={audit.get('session_stage')}",
        f"Campaign: {audit.get('campaign_status')} | "
        f"tested={audit.get('total_hypotheses')} confirmed={audit.get('total_confirmed')}",
        f"Wall clock (est): {_fmt_sec(audit.get('wall_clock_sec_estimated'))} | "
        f"API elapsed_sec: {audit.get('elapsed_sec_reported')}",
        "",
        "Phase durations:",
    ]
    for phase, sec in (audit.get("phase_durations_sec") or {}).items():
        lines.append(f"  {phase}: {_fmt_sec(sec)}")
    lines.extend(["", "Accounted time:"])
    acc = audit.get("accounted_ms") or {}
    lines.append(f"  experiment trace (tools+code): {_fmt_sec((acc.get('experiment_trace_total_ms') or 0) / 1000)}")
    lines.append(f"  LLM calls (logged duration_ms): {_fmt_sec((acc.get('llm_calls_total_ms') or 0) / 1000)}")
    lines.append(f"  unaccounted (queue/orchestrator/gaps): {_fmt_sec(acc.get('unaccounted_sec_est'))}")
    lines.extend(["", "Top tools by total duration:"])
    for row in (audit.get("trace_breakdown") or {}).get("top_tools_by_total_ms") or []:
        lines.append(
            f"  {row['tool']}: {_fmt_sec(row['total_ms'] / 1000)} "
            f"({row['count']} calls, max {_fmt_sec(row['max_ms'] / 1000)})"
        )
    lines.extend(["", "Top LLM call purposes:"])
    for row in (audit.get("llm_breakdown") or {}).get("top_purposes") or []:
        lines.append(
            f"  {row['purpose']}: {_fmt_sec(row['total_ms'] / 1000)} ({row['count']} calls)"
        )
    lines.extend(["", "High-volume events:"])
    for et, n in list((audit.get("event_counts") or {}).items())[:18]:
        lines.append(f"  {et}: {n}")
    if audit.get("likely_blockers"):
        lines.extend(["", "Likely blockers / slow components:"])
        for b in audit["likely_blockers"]:
            lines.append(f"  • {b}")
    return "\n".join(lines)


def _abstract_matches_confirmed(abstract: str, n_confirmed: int) -> bool:
    if n_confirmed <= 0:
        return bool(abstract.strip())
    patterns = [
        rf"(?i)\b{n_confirmed}\s+were\s+supported\b",
        rf"(?i)\b{n_confirmed}\s+hypotheses?\s+were\s+supported\b",
        rf"(?i)confirmed\s*=\s*{n_confirmed}\b",
    ]
    return any(re.search(p, abstract) for p in patterns)


def main() -> int:
    p = argparse.ArgumentParser(description="Timing audit for a completed Propab campaign.")
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--campaign-id", default="")
    p.add_argument(
        "--state-file",
        default=str(_REPO / "artifacts" / "e2e_campaign_latest.json"),
    )
    p.add_argument(
        "--out",
        default=str(_REPO / "artifacts" / "e2e_timing_audit.json"),
        help="Write full JSON audit here",
    )
    p.add_argument("--report", default=str(_REPO / "artifacts" / "e2e_timing_audit.txt"))
    p.add_argument("--check-paper", action="store_true", help="Verify paper abstract vs confirmed count")
    args = p.parse_args()

    cid = args.campaign_id.strip()
    if not cid and args.state_file:
        path = Path(args.state_file)
        if path.is_file():
            cid = (json.loads(path.read_text(encoding="utf-8")).get("campaign_id") or "").strip()
    if not cid:
        print("Need --campaign-id or --state-file with campaign_id", file=sys.stderr)
        return 2

    try:
        audit = build_audit(args.api, cid)
    except HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:1500]}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Cannot reach API: {exc}", file=sys.stderr)
        return 1

    report = render_report(audit)
    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON audit: {out_path}")

    if args.report:
        Path(args.report).write_text(report + "\n", encoding="utf-8")
        print(f"Text report: {args.report}")

    if args.check_paper:
        try:
            paper = get_json(f"{args.api.rstrip('/')}/sessions/{cid}/paper")
            payload = paper.get("paper") or {}
            abstract = str(payload.get("abstract_latex") or "")
            n = int(audit.get("total_confirmed") or 0)
            if _abstract_matches_confirmed(abstract, n):
                print(f"PAPER_OK: abstract matches total_confirmed={n}")
            else:
                print(f"PAPER_MISMATCH: expected {n} confirmed in prose abstract; snippet={abstract[:400]!r}")
                return 1
        except HTTPError as exc:
            if exc.code == 404:
                print("PAPER_NOT_READY (404)")
                return 1
            raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
