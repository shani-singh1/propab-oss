#!/usr/bin/env python3
"""Live terminal health dashboard (Agent 2 T2-004)."""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))


def _get(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _render(api: str = "http://localhost:8000") -> str:
    now = datetime.now(UTC).strftime("%H:%M:%S")
    lines = [f"PROPAB HEALTH DASHBOARD                         Updated: {now}", ""]

    state_path = ROOT / "artifacts" / "v1_frontier_campaign_latest.json"
    cid = None
    if state_path.is_file():
        cid = json.loads(state_path.read_text()).get("campaign_id")

    if not cid:
        lines.append("CURRENT CAMPAIGN: (none tracked)")
        return "\n".join(lines)

    raw = _get(f"{api.rstrip('/')}/campaigns/{cid}")
    if not raw:
        lines.append(f"CURRENT CAMPAIGN: {cid[:8]} (API unreachable)")
        return "\n".join(lines)

    camp = raw.get("campaign") or {}
    summary = raw.get("summary") or {}
    nodes = (camp.get("hypothesis_tree") or {}).get("nodes") or {}
    vc: dict[str, int] = {}
    for n in nodes.values():
        v = n.get("verdict", "?")
        vc[v] = vc.get(v, 0) + 1
    tested = sum(vc.get(k, 0) for k in ("confirmed", "refuted", "inconclusive"))
    conf = vc.get("confirmed", 0)
    inc_rate = (vc.get("inconclusive", 0) / tested * 100) if tested else 0
    elapsed = (summary.get("elapsed_sec") or 0) / 60
    bs = camp.get("belief_state") or {}
    active = len(bs.get("active_beliefs") or [])

    lines.extend([
        f"CURRENT CAMPAIGN: {cid[:8]} ({camp.get('status', '?')}, {elapsed:.0f} min)",
        f"Tested: {tested}  Confirmed: {conf} ({100*conf/max(tested,1):.1f}%)  "
        f"Refuted: {vc.get('refuted',0)}  Inconclusive: {vc.get('inconclusive',0)}",
        f"Active beliefs: {active}  Stop: {camp.get('stop_reason') or '-'}",
        "",
        "HEALTH METRICS vs TARGETS",
        f"inconclusive_rate:       {inc_rate:.0f}%  (target <50%)  {'OK' if inc_rate < 50 else 'WARN'}",
        f"beliefs_active:          {active}   (target >=1)  {'OK' if active >= 1 else 'WARN'}",
    ])
    return "\n".join(lines)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--interval", type=int, default=30)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    if args.once:
        print(_render(args.api))
        return 0

    try:
        while True:
            print("\033[2J\033[H", end="")
            print(_render(args.api))
            time.sleep(max(5, args.interval))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
