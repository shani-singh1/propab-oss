#!/usr/bin/env python3
"""Poll campaign every N minutes until terminal; then run post-completion analysis."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CID = "faaf394b-7f95-4778-9136-e922f2401e7f"
DEFAULT_LOG = ROOT / "artifacts" / "mandrake_resume_live.txt"
DEFAULT_BASELINE = ROOT / "artifacts" / "mandrake_duplicate_pair_analysis.json"
ANALYSIS_OUT = ROOT / "artifacts" / "mandrake_resume_completion_analysis.json"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def _payload(ev: dict) -> dict:
    raw = ev.get("payload_json") or ev.get("payload") or {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _poll_line(base: str, cid: str, poll: int) -> tuple[str, bool, str]:
    """Return (log_line, terminal, status)."""
    ts = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    d = _get(f"{base}/campaigns/{cid}")
    c = d.get("campaign") or {}
    s = d.get("summary") or {}
    tree = s.get("tree") or {}
    verdicts = tree.get("verdict_counts") or {}
    rs = d.get("research_session") or {}

    synth_n = 0
    rej_dup_total = 0
    bind_rej_total = 0
    fals_rej_total = 0
    cap_rej_total = 0
    api_beliefs = [
        str(b.get("statement") or "")[:100]
        for b in (c.get("belief_state") or {}).get("active_beliefs") or []
        if isinstance(b, dict)
    ]
    last_beliefs: list[str] = api_beliefs
    try:
        ev_data = _get(f"{base}/sessions/{cid}/events?limit=2000")
        evs = ev_data.get("events") if isinstance(ev_data, dict) else ev_data
        if isinstance(evs, list):
            for e in evs:
                if e.get("step") == "campaign.synthesis":
                    synth_n += 1
                    p = _payload(e)
                    if not api_beliefs:
                        last_beliefs = [
                            str(b.get("statement") or "")[:100]
                            for b in (p.get("active_beliefs") or []) if isinstance(b, dict)
                        ]
                    rej_dup_total += int(p.get("n_rejected_duplicate") or 0)
                    bind_rej_total += int(p.get("binding_rejected_count") or 0)
                    fals_rej_total += int(p.get("falsifiability_rejected_count") or 0)
                    cap_rej_total += int(p.get("belief_cap_rejected_count") or 0)
    except Exception:
        pass

    status = c.get("status")
    elapsed = int(s.get("elapsed_sec") or 0)
    remaining = int(s.get("remaining_sec") or 0)
    line = (
        f"[{ts}] poll={poll} status={status} stop={c.get('stop_reason')} "
        f"elapsed={elapsed}s remaining={remaining}s "
        f"hyps={s.get('total_hypotheses')} cap={c.get('max_hypotheses_cap')} "
        f"refuted={verdicts.get('refuted')} pending={verdicts.get('pending')} "
        f"frontier={tree.get('frontier_size')} synthesis={synth_n} "
        f"rej_dup_total={rej_dup_total} bind_rej={bind_rej_total} fals_rej={fals_rej_total} "
        f"cap_rej={cap_rej_total} rival_mode={(c.get('belief_state') or {}).get('rival_exhaustion_mode')} "
        f"session={rs.get('status')}"
    )
    if last_beliefs:
        line += f" beliefs={last_beliefs!r}"

    terminal = status not in ("active", None) or rs.get("status") == "completed"
    if status == "active" and remaining <= 0 and elapsed > 3600 and (verdicts.get("pending") or 0) == 0:
        terminal = True
    return line, terminal, str(status)


def _run_analysis(cid: str, api: str) -> dict:
    report: dict = {"campaign_id": cid, "ran_at": datetime.now(tz=UTC).isoformat()}
    py = sys.executable

    def run_script(script: str, extra: list[str] | None = None) -> dict | None:
        cmd = [py, str(ROOT / "scripts" / script)]
        if script == "validate_resume_readiness.py":
            cmd.append(cid)
        else:
            cmd.extend(["--campaign-id", cid])
        if extra:
            cmd.extend(extra)
        cmd.extend(["--api", api])
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
            if proc.stdout.strip():
                try:
                    return json.loads(proc.stdout)
                except json.JSONDecodeError:
                    return {"stdout": proc.stdout[-2000:], "stderr": proc.stderr[-500:]}
            return {"stderr": proc.stderr[-500:], "code": proc.returncode}
        except Exception as exc:
            return {"error": str(exc)}

    d = _get(f"{api.rstrip('/')}/campaigns/{cid}")
    c = d.get("campaign") or {}
    s = d.get("summary") or {}
    pre_hyps = 292  # pre-resume baseline

    report["final_summary"] = s
    report["final_status"] = c.get("status")
    report["stop_reason"] = c.get("stop_reason")
    report["max_hypotheses_cap"] = c.get("max_hypotheses_cap")
    report["hypotheses_pre_resume"] = pre_hyps
    report["hypotheses_post"] = s.get("total_hypotheses")
    report["new_hypotheses_since_resume"] = max(0, int(s.get("total_hypotheses") or 0) - pre_hyps)

    report["duplicate_check"] = run_script("post_resume_duplicate_check.py", ["--baseline", str(DEFAULT_BASELINE)])
    report["belief_audit"] = run_script("audit_mandrake_final_beliefs.py")
    report["evidence_binding_audit"] = run_script("audit_evidence_binding.py")
    report["readiness"] = run_script("validate_resume_readiness.py")

    # Synthesis dedup from events
    evs = _get(f"{api.rstrip('/')}/sessions/{cid}/events?limit=2000")
    if not isinstance(evs, list):
        evs = evs.get("events", [])
    post_resume_syn = [
        e for e in evs
        if (e.get("step") or "") == "campaign.synthesis"
        and str(e.get("created_at") or "") >= "2026-06-21T14:34"
    ]
    rej = sum(int(_payload(e).get("n_rejected_duplicate") or 0) for e in post_resume_syn)
    bind_rej = sum(int(_payload(e).get("binding_rejected_count") or 0) for e in post_resume_syn)
    fals_rej = sum(int(_payload(e).get("falsifiability_rejected_count") or 0) for e in post_resume_syn)
    cap_rej = sum(int(_payload(e).get("belief_cap_rejected_count") or 0) for e in post_resume_syn)
    report["post_resume_synthesis"] = {
        "rounds": len(post_resume_syn),
        "n_rejected_duplicate_total": rej,
        "binding_rejected_total": bind_rej,
        "falsifiability_rejected_total": fals_rej,
        "belief_cap_rejected_total": cap_rej,
        "dedup_observed": rej > 0,
        "binding_observed": bind_rej > 0 or fals_rej > 0 or cap_rej > 0,
    }
    active = (c.get("belief_state") or {}).get("active_beliefs") or []
    report["validation"] = {
        "beliefs_persisted": len(active) == 2,
        "two_belief_cap_held": len(active) <= 2,
        "explicit_stop_reason": bool(c.get("stop_reason")),
        "no_new_duplicates": (report.get("duplicate_check") or {}).get("fix_held"),
        "dedup_metric_emitted": rej > 0,
        "binding_metric_emitted": bind_rej > 0 or fals_rej > 0,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_id", nargs="?", default=DEFAULT_CID)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between polls (default 5 min)")
    parser.add_argument("--max-polls", type=int, default=48, help="~4h at 5min")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--analysis-out",
        type=Path,
        default=ANALYSIS_OUT,
        help="Post-completion JSON report path",
    )
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    base = args.api.rstrip("/")
    cid = args.campaign_id
    args.log.parent.mkdir(parents=True, exist_ok=True)

    for poll in range(args.max_polls):
        try:
            line, terminal, status = _poll_line(base, cid, poll)
        except Exception as exc:
            line = f"[{datetime.now(tz=UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}] poll={poll} ERROR {exc}"
            terminal = False
            status = "error"

        print(line, flush=True)
        with args.log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        if terminal:
            print(f"DONE status={status}", flush=True)
            if not args.skip_analysis:
                print("Running post-completion analysis...", flush=True)
                report = _run_analysis(cid, base)
                args.analysis_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps({"analysis_out": str(args.analysis_out), "validation": report.get("validation")}, indent=2))
            return 0

        if poll + 1 < args.max_polls:
            time.sleep(max(60, args.interval))

    print("TIMEOUT max polls reached", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
