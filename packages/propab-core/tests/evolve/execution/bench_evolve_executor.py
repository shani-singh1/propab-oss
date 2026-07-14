"""WS-2 throughput benchmark: evaluations/sec at N workers, with a trivial program.

Evaluations/sec is THE metric — a FunSearch-class search is a lottery bought in bulk, so the
engine's ceiling is set by how many programs the runner+verifier can chew through per second.

    python packages/propab-core/tests/evolve/bench_evolve_executor.py
    python packages/propab-core/tests/evolve/bench_evolve_executor.py --workers 1,8,64 --evals 3000

Deliberately *not* a pytest test: throughput numbers depend on the host, and a machine-dependent
assertion in CI is a flaky test, not a measurement.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any

from propab.evolve.engine import Engine, EngineConfig
from propab.evolve.executor import ParallelExecutor
from propab.evolve.ledger import Ledger
from propab.evolve.problem import INVALID, Verdict
from propab.evolve.program import Program
from propab.evolve.runner import SandboxLimits, SandboxProgramRunner

TRIVIAL = "def build():\n    return [1, 2, 4, 8]\n"


class SumProblem:
    """The cheapest possible exact verifier, so the benchmark measures the *substrate*."""

    name = "sum-of-ints"

    def describe(self) -> str:
        return "Return a list of ints. Score = sum."

    def verify(self, candidate: Any) -> Verdict:
        if not isinstance(candidate, list) or not candidate:
            return INVALID
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in candidate):
            return INVALID
        return Verdict(valid=True, score=float(sum(candidate)))

    def best_known(self) -> float:
        return 1e9

    def is_improvement(self, verdict: Verdict) -> bool:
        return False

    def seed_programs(self) -> list[str]:
        return [TRIVIAL]


class _NullMutator:
    def mutate(self, parents: list[Program], problem: Any) -> Program:
        return Program(code=TRIVIAL)


def bench(workers: int, evals: int, repeats: int) -> dict[str, Any]:
    limits = SandboxLimits(preimport=(), memory_mb=512, spawn_timeout_s=180.0)
    runner = SandboxProgramRunner(pool_size=workers, limits=limits)
    engine = Engine(
        problem=SumProblem(),
        mutator=_NullMutator(),
        runner=runner,
        ledger=Ledger(root="artifacts/evolve-bench"),
        config=EngineConfig(workers=workers, program_timeout_s=10.0),
    )
    executor = ParallelExecutor(engine, workers=workers)

    try:
        # Warm the pool: spawning a child costs ~0.5s on this host, which has nothing to do with
        # steady-state throughput. A real run pays it once and then runs for hours.
        t0 = time.perf_counter()
        executor.evaluate_all([Program(code=TRIVIAL)] * (workers * 2))
        warmup_s = time.perf_counter() - t0

        rates = []
        for _ in range(repeats):
            batch = [Program(code=TRIVIAL)] * evals
            t0 = time.perf_counter()
            outcomes = executor.evaluate_all(batch)
            elapsed = time.perf_counter() - t0
            assert len(outcomes) == evals
            assert all(o.verdict.valid for o in outcomes), "benchmark is not measuring real work"
            rates.append(evals / elapsed)
    finally:
        runner.close()

    return {
        "workers": workers,
        "evals": evals,
        "warmup_s": warmup_s,
        "best_eps": max(rates),
        "median_eps": statistics.median(rates),
        "runner": runner.stats.snapshot(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="8,64", help="comma-separated worker counts")
    ap.add_argument("--evals", type=int, default=2000, help="evaluations per timed round")
    ap.add_argument("--repeats", type=int, default=3, help="timed rounds (best + median reported)")
    args = ap.parse_args()

    print(f"python {sys.version.split()[0]} on {sys.platform}")
    print(f"{'workers':>8} {'evals/sec':>12} {'median':>10} {'warmup':>9} {'mean run':>10}")
    print("-" * 54)

    rows = []
    for w in [int(x) for x in args.workers.split(",")]:
        row = bench(w, args.evals, args.repeats)
        rows.append(row)
        print(
            f"{row['workers']:>8} {row['best_eps']:>12.1f} {row['median_eps']:>10.1f} "
            f"{row['warmup_s']:>8.1f}s {row['runner']['mean_seconds'] * 1000:>8.2f}ms"
        )

    if len(rows) > 1:
        base, top = rows[0], rows[-1]
        speedup = top["best_eps"] / base["best_eps"] if base["best_eps"] else 0.0
        print(
            f"\nscaling {base['workers']} -> {top['workers']} workers: {speedup:.2f}x"
            f"  ({base['best_eps']:.0f} -> {top['best_eps']:.0f} evals/sec)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
