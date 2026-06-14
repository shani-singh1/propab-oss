#!/usr/bin/env python3
"""
Simulator calibration v2/v3 (fixes.md).

Usage:
  python scripts/calibrate_simulator.py --compare
  python scripts/calibrate_simulator.py --v3
  python scripts/calibrate_simulator.py --v3 --out artifacts/simulator_calibration_v3.json
  python scripts/calibrate_simulator.py --direction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))

from propab.layer05.ensemble_simulator import SIM_V3
from propab.layer05.hybrid_simulator import SIM_V2
from propab.layer05.simulator_calibration import run_calibration_cycle
from propab.layer05.direction_calibration import run_direction_calibration
from propab.layer05.simulator_calibration_v3 import run_v3_calibration
from propab.layer05.simulator_registry import SIM_V1


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulator calibration v2/v3/direction")
    parser.add_argument(
        "--trajectories",
        default=str(ROOT / "artifacts" / "entropy_trajectories.json"),
        help="entropy_trajectories.json path",
    )
    parser.add_argument("--version", default=SIM_V2, help="Simulator version to register (v2 mode)")
    parser.add_argument("--compare", action="store_true", help="Bench sim_v1 and sim_v2 (v2 mode)")
    parser.add_argument(
        "--v3",
        action="store_true",
        help="Run v3 grid search + LOO-CV + ensemble + error ledger",
    )
    parser.add_argument(
        "--direction",
        action="store_true",
        help="Direction error reduction: stage sim_v4 + asymmetric loss + error dataset",
    )
    parser.add_argument("--no-persist", action="store_true", help="Skip writing artifacts")
    parser.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: simulator_calibration.json or _v3.json)",
    )
    args = parser.parse_args()

    traj = Path(args.trajectories)
    if not traj.is_file():
        print(f"Trajectories file not found: {traj}", file=sys.stderr)
        return 1

    persist = not args.no_persist
    default_out = str(ROOT / "artifacts" / "simulator_calibration.json")
    if args.direction:
        default_out = str(ROOT / "artifacts" / "direction_calibration.json")
    elif args.v3:
        default_out = str(ROOT / "artifacts" / "simulator_calibration_v3.json")
    out = Path(args.out or default_out)

    if args.direction:
        report = run_direction_calibration(trajectory_path=traj, persist=persist)
        payload = {
            "layer": "0.5",
            "calibration": "direction_error_reduction",
            "trajectories": str(traj),
            "report": report.to_dict(),
            "meets_70pct_gate": report.meets_70pct_gate,
            "v4_accepted": report.v4_accepted,
            "direction_errors_before": report.direction_errors_before,
            "direction_errors_after": report.direction_errors_after,
        }
    elif args.v3:
        report = run_v3_calibration(trajectory_path=traj, persist=persist)
        payload = {
            "layer": "0.5",
            "calibration": "simulator_v3",
            "trajectories": str(traj),
            "report": report.to_dict(),
            "meets_80pct_loo_gate": report.meets_80pct_loo_gate,
            "v2_accepted": report.v2_accepted,
            "v3_accepted": report.v3_accepted,
        }
    else:
        reports: dict[str, dict] = {}
        if args.compare:
            for ver in (SIM_V1, SIM_V2):
                rep = run_calibration_cycle(
                    trajectory_path=traj,
                    simulator_version=ver,
                    persist=persist and ver == args.version,
                )
                reports[ver] = rep.to_dict()
        else:
            rep = run_calibration_cycle(
                trajectory_path=traj,
                simulator_version=args.version,
                persist=persist,
            )
            reports[args.version] = rep.to_dict()

        payload = {
            "layer": "0.5",
            "calibration": "simulator_v2",
            "trajectories": str(traj),
            "reports": reports,
        }
        if args.compare and SIM_V1 in reports and SIM_V2 in reports:
            v1_dir = reports[SIM_V1]["aggregate"].get("directional_agreement", 0)
            v2_dir = reports[SIM_V2]["aggregate"].get("directional_agreement", 0)
            payload["v2_improves_directional"] = v2_dir > v1_dir
            payload["meets_80pct_gate"] = v2_dir >= 0.80

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
