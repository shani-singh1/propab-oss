#!/usr/bin/env python3
"""Poll campaign for N minutes and print diagnostics (fixes.md review window)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import urlopen

ML_DRIFT_TOKENS = ("geonorm", "siamesenorm", "batch normalization", "language model", "mnist", "adam", "sgd")


def get_json(url: str) -> dict:
    with urlopen(url, timeout=60) as resp:
        return json.loads(resp.read())


def check_events(api: str, cid: str) -> dict:
    url = f"{api}/sessions/{cid}/events?limit=200"
    try:
        data = get_json(url)
    except Exception as exc:
        return {"error": str(exc)}
    events = data if isinstance(data, list) else data.get("events") or []
    out: dict = {"literature_titles": [], "hypothesis_themes": [], "verification_methods": [], "alerts": []}
    for ev in reversed(events):
        et = ev.get("event_type") or ev.get("type") or ""
        payload = ev.get("payload_json") or ev.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        if et == "campaign.progress" and payload.get("step") == "literature.prior_corpus":
            out["literature_titles"] = payload.get("paper_titles") or []
        if et == "literature.prior_built":
            diag = (payload.get("prior") or {}).get("retrieval_diagnostics") or {}
            kept = diag.get("papers_kept") or []
            out["literature_kept"] = [p.get("title") for p in kept[:12]]
            out["evidence_status"] = (payload.get("prior") or {}).get("evidence_status")
        if et == "hypothesis.generated":
            for h in payload.get("hypotheses") or []:
                if isinstance(h, dict) and h.get("theme"):
                    out["hypothesis_themes"].append(h.get("theme"))
        if et == "campaign.progress" and payload.get("step") == "campaign.verification_diagnostic":
            out["verification_methods"].append(payload.get("verification_method"))
        step = payload.get("step") or ev.get("step") or ""
        if step == "campaign.prior_insufficient":
            out["alerts"].append("INSUFFICIENT_EVIDENCE prior")
    for title in out.get("literature_titles") or []:
        tl = (title or "").lower()
        if any(tok in tl for tok in ML_DRIFT_TOKENS):
            out["alerts"].append(f"ML drift in literature: {title[:80]}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--state-file", required=True)
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--interval", type=int, default=90)
    args = p.parse_args()
    state = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    api = state["api"].rstrip("/")
    cid = state["campaign_id"]
    end = time.time() + args.minutes * 60
    n = 0
    while time.time() < end:
        n += 1
        blob = get_json(f"{api}/campaigns/{cid}")
        ev = check_events(api, cid)
        print(f"\n=== poll {n} ===", flush=True)
        s = blob.get("summary") or blob
        tree = s.get("tree") or {}
        print(
            f"status={s.get('status')} stage={blob.get('research_session', {}).get('stage')} "
            f"tested={s.get('total_hypotheses')} confirmed={s.get('total_confirmed')} "
            f"frontier={tree.get('frontier_size')} nodes={tree.get('total_nodes')} "
            f"budget_used={s.get('compute_seconds_used')}/{s.get('compute_budget_seconds')}",
            flush=True,
        )
        if ev.get("evidence_status"):
            print(f"prior evidence_status={ev['evidence_status']}", flush=True)
        if ev.get("literature_titles"):
            print("prior paper_titles:", ev["literature_titles"][:6], flush=True)
        if ev.get("literature_kept"):
            print("literature kept:", ev["literature_kept"][:6], flush=True)
        if ev.get("hypothesis_themes"):
            print("hypothesis themes:", ev["hypothesis_themes"][:8], flush=True)
        if ev.get("verification_methods"):
            print("verification methods:", ev["verification_methods"][-6:], flush=True)
        if ev.get("alerts"):
            print("ALERTS:", ev["alerts"], flush=True)
        if s.get("status") in ("completed", "breakthrough", "budget_exhausted"):
            print("Campaign terminal early.", flush=True)
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
