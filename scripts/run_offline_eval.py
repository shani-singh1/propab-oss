#!/usr/bin/env python3
"""
Layer 0.5 offline evaluation (fixes.md P0–P5).

Usage:
  python scripts/run_offline_eval.py
  python scripts/run_offline_eval.py --benchmark simulator
  python scripts/run_offline_eval.py --benchmark offline-policy --persist-sim
  python scripts/run_offline_eval.py --policy-id pol-xxx --trajectories artifacts/entropy_trajectories.json
  python scripts/calibrate_simulator.py --compare
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))

from propab.layer05.microbenchmarks import (
    analyst_bench,
    closure_bench,
    entropy_bench,
    frontier_bench,
    offline_policy_eval_bench,
    policy_bench,
    run_all_benchmarks,
    simulator_bench,
)
from propab.layer05.policy_offline_eval import evaluate_policy_offline
from propab.layer05.replay_loader import load_campaign_bundle, load_snapshots_from_json
from propab.layer05.simulation_fitness_ledger import SimulationFitnessLedger
from propab.policy_fitness_ledger import PolicyFitnessLedger
from propab.policy_store import PolicyStore


async def _resolve_snapshots(args) -> tuple[str | None, list[dict]]:
    if args.trajectories:
        by_id = load_snapshots_from_json(args.trajectories)
        if args.campaign_id:
            return args.campaign_id, by_id.get(args.campaign_id, [])
        if by_id:
            cid = next(iter(by_id))
            return cid, by_id[cid]
        return None, []
    if args.campaign_id:
        bundle = await load_campaign_bundle(args.campaign_id)
        return bundle.campaign_id, bundle.snapshots
    traj_path = ROOT / "artifacts" / "entropy_trajectories.json"
    if traj_path.is_file():
        by_id = load_snapshots_from_json(traj_path)
        if by_id:
            cid = args.campaign_id or next(iter(by_id))
            return cid, by_id.get(cid, [])
    return None, []


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer 0.5 offline replay and microbenchmarks")
    parser.add_argument("--campaign-id", help="Campaign UUID for replay")
    parser.add_argument("--policy-id", help="Policy to evaluate offline")
    parser.add_argument("--trajectories", default="", help="entropy_trajectories.json path")
    parser.add_argument(
        "--benchmark",
        default="all",
        choices=(
            "all", "frontier", "entropy", "simulator", "offline-policy",
            "closure", "policy", "analyst",
        ),
    )
    parser.add_argument("--persist-sim", action="store_true", help="Save SimulationFitnessLedger")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "layer05_report.json"))
    parser.add_argument(
        "--simulator-version",
        default="",
        help="sim_v1 or sim_v2 (default: registry active)",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    cid, snapshots = asyncio.run(_resolve_snapshots(args))
    fitness = PolicyFitnessLedger.load()
    sim_ledger = SimulationFitnessLedger.load() if args.persist_sim else None

    if args.benchmark == "all":
        report = run_all_benchmarks(
            campaign_id=cid,
            snapshots=snapshots or None,
            fitness=fitness,
            sim_ledger=sim_ledger,
        )
        payload = report.to_dict()
    elif args.benchmark == "offline-policy":
        if not snapshots:
            print("No snapshots", file=sys.stderr)
            return 1
        store = PolicyStore.load()
        policy = store.get_policy(args.policy_id) if args.policy_id else (
            store.latest_candidate(domain_bucket="graphs", budget_bucket="3h")
            or store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")
        )
        ledger = sim_ledger or SimulationFitnessLedger()
        result = evaluate_policy_offline(
            candidate=policy,
            campaign_id=cid or "unknown",
            snapshots=snapshots,
            ledger=ledger,
            persist=args.persist_sim,
            simulator_version=args.simulator_version or None,
            trajectory_path=args.trajectories or str(ROOT / "artifacts" / "entropy_trajectories.json"),
        )
        payload = {"benchmark": "offline-policy", "result": result.to_dict()}
    else:
        bench_map = {
            "frontier": lambda: frontier_bench(campaign_id=cid or "unknown", snapshots=snapshots),
            "entropy": lambda: entropy_bench(
                campaign_id=cid or "unknown", snapshots=snapshots, policy_id=args.policy_id,
            ),
            "simulator": lambda: simulator_bench(
                campaign_id=cid or "unknown", snapshots=snapshots, policy_id=args.policy_id,
            ),
            "closure": lambda: closure_bench(fitness),
            "policy": lambda: policy_bench(fitness),
            "analyst": lambda: analyst_bench(fitness),
        }
        if args.benchmark in ("frontier", "entropy", "simulator") and not snapshots:
            print("No snapshots — provide --campaign-id or --trajectories", file=sys.stderr)
            return 1
        payload = {
            "benchmark": args.benchmark,
            "campaign_id": cid,
            "n_snapshots": len(snapshots),
            "result": bench_map[args.benchmark]().to_dict(),
        }

    payload["total_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    payload["campaign_id"] = cid
    payload["layer"] = "0.5"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
