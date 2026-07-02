#!/usr/bin/env python3
"""Confirm dedup emitter code is live in the API container (fixes.md preflight)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CID = "faaf394b-7f95-4778-9136-e922f2401e7f"


def _grep_local() -> bool:
    path = ROOT / "packages" / "propab-core" / "propab" / "campaign_synthesis.py"
    text = path.read_text(encoding="utf-8")
    return '"n_rejected_duplicate": metrics.get("n_rejected_duplicate"' in text


def _grep_container(service: str = "api") -> tuple[bool, str]:
    cmd = [
        "docker", "compose", "-f", "docker-compose.yml",
        "exec", "-T", service,
        "grep", "-n", "n_rejected_duplicate",
        "/app/packages/propab-core/propab/campaign_synthesis.py",
    ]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=30)
    except Exception as exc:
        return False, str(exc)
    ok = proc.returncode == 0 and "n_rejected_duplicate" in proc.stdout
    return ok, (proc.stdout or proc.stderr or "").strip()


def _last_synthesis_has_metric(api: str, campaign_id: str) -> dict:
    url = f"{api.rstrip('/')}/stream/{campaign_id}?tail=500"
    try:
        with urlopen(url, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    synth_events = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if ev.get("step") == "campaign.synthesis":
            synth_events.append(ev)
    if not synth_events:
        return {"ok": False, "error": "no synthesis events in stream tail"}
    last = synth_events[-1]
    payload = last.get("payload") or last.get("payload_json") or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {}
    has_key = "n_rejected_duplicate" in payload
    return {
        "ok": has_key,
        "n_rejected_duplicate": payload.get("n_rejected_duplicate"),
        "event_step": last.get("step"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CID)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--service", default="api")
    parser.add_argument("--skip-container", action="store_true")
    parser.add_argument("--skip-events", action="store_true")
    args = parser.parse_args()

    report: dict = {"local_source_ok": _grep_local()}
    if not report["local_source_ok"]:
        print(json.dumps(report, indent=2))
        return 1

    if not args.skip_container:
        container_ok, detail = _grep_container(args.service)
        report["container_source_ok"] = container_ok
        report["container_grep"] = detail[:500]
        if not container_ok:
            print(json.dumps(report, indent=2))
            return 1

    if not args.skip_events:
        report["last_synthesis_event"] = _last_synthesis_has_metric(args.api, args.campaign_id)
        if not report["last_synthesis_event"].get("ok"):
            report["note"] = (
                "Emitter key missing from last synthesis event — redeploy with mount-dev "
                "+ campaign-run before judging dedup effectiveness."
            )

    print(json.dumps(report, indent=2))
    if report.get("container_source_ok") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
