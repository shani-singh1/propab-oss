#!/usr/bin/env python3
"""
Operator credit assignment cycle (fixes.md operator_credit_assignment).

Experience → Replay → Counterfactuals → Operator Credits → Priors → Bench

Usage:
  python scripts/run_operator_credit.py
  python scripts/run_operator_credit.py --trajectories artifacts/entropy_trajectories.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))

from propab.operator_credit.credit_cycle import run_operator_credit_cycle


def main() -> int:
    parser = argparse.ArgumentParser(description="Operator credit assignment")
    parser.add_argument(
        "--trajectories",
        default=str(ROOT / "artifacts" / "entropy_trajectories.json"),
    )
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "operator_credit_report.json"),
    )
    args = parser.parse_args()

    traj = Path(args.trajectories)
    if not traj.is_file():
        print(f"Trajectories not found: {traj}", file=sys.stderr)
        return 1

    report, traces, credits = run_operator_credit_cycle(
        trajectory_path=traj,
        persist=not args.no_persist,
    )
    payload = {
        "layer": "operator_credit",
        "trajectories": str(traj),
        "report": report.to_dict(),
        "sample_traces": [t.to_dict() for t in traces.traces[:3]],
        "top_credits": sorted(
            [c.to_dict() for c in credits.credits],
            key=lambda c: c["contribution"],
            reverse=True,
        )[:5],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
