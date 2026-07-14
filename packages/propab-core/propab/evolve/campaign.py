"""Evolve — the campaign runner: wire everything and search.

    python -m propab.evolve.campaign ecc --n 32 --k 14 --steps 500
    python -m propab.evolve.campaign graph --conjecture elphick_linz_wocjan --steps 500
    python -m propab.evolve.campaign erdos143 --n-max 1000 --steps 200

Shape of a run (and why):

* ``workers`` >> ``pool_size``. A mutation costs ~1s at the model while the sandbox does ~430
  evals/sec — a ~400x gap. The LLM is the bottleneck and the sandbox has headroom to spare, so scale
  *concurrency*, not sandboxes. ``workers=64, pool_size=8`` is the right shape on an 8-core box.
* Nothing is banked without a passing adversarial audit. The default is REJECT.
* Every recorded result exports a self-contained bundle that a third party re-checks with
  ``python verify.py`` — zero propab imports, zero network. Credibility comes from the checker, not
  from us.

Read `docs/discovery/verification-discipline.md` before believing any number this prints.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from .auditor import Auditor
from .engine import EngineConfig
from .engine_impl import EvolutionEngine
from .executor import ParallelExecutor
from .ledger_impl import FileLedger
from .llm import GeminiMutationClient
from .mutator import LLMMutator
from .problem import Problem
from .runner import SandboxLimits, SandboxProgramRunner

logger = logging.getLogger("evolve.campaign")

DEFAULT_LEDGER_ROOT = "artifacts/evolve"


def build_problem(kind: str, args: argparse.Namespace) -> Problem:
    if kind == "ecc":
        from .targets.ecc import ECCProblem

        return ECCProblem(n=args.n, k=args.k)
    if kind == "graph":
        from .targets.graph_conj import GraphConjectureProblem

        return GraphConjectureProblem(args.conjecture)
    if kind == "erdos143":
        from .targets.erdos143 import Erdos143Problem

        return Erdos143Problem(n_max=args.n_max)
    raise SystemExit(f"unknown target {kind!r}")


def run_campaign(
    problem: Problem,
    *,
    steps: int | None = 500,
    workers: int = 64,
    pool_size: int = 8,
    islands: int = 8,
    ledger_root: str | Path = DEFAULT_LEDGER_ROOT,
    program_timeout_s: float = 10.0,
    model: str | None = None,
    temperature: float = 0.9,
) -> dict[str, Any]:
    """Run one search. Returns a summary dict; every genuine find is on disk in the ledger."""
    llm = GeminiMutationClient(model=model, temperature=temperature)
    ledger = FileLedger(root=ledger_root)

    started = time.time()
    with SandboxProgramRunner(pool_size=pool_size, limits=SandboxLimits()) as runner:
        engine = EvolutionEngine(
            problem,
            LLMMutator(llm),
            runner,
            ledger,
            EngineConfig(
                islands=islands,
                workers=workers,
                program_timeout_s=program_timeout_s,
                max_steps=steps,
            ),
            auditor=Auditor(),   # mandatory: with no auditor the ledger banks nothing
        )

        def on_progress(_: Any = None) -> None:
            best = engine.best_score()
            logger.info(
                "[evolve] steps=%d best=%.6f best_known=%.6f improvements=%d audit_kills=%d "
                "llm=%s elapsed=%.0fs",
                engine.steps, best, _safe_best_known(problem), engine.improvements,
                engine.audit_kills, llm.stats(), time.time() - started,
            )

        executor = ParallelExecutor(engine, workers=workers, on_progress=on_progress,
                                    progress_interval_s=20.0)
        try:
            stats = executor.run_steps(max_steps=steps)
        except KeyboardInterrupt:
            logger.warning("[evolve] interrupted — stopping workers")
            executor.stop()
            stats = None

    summary = {
        "problem": getattr(problem, "name", type(problem).__name__),
        "steps": engine.steps,
        "best_score": engine.best_score(),
        "best_known": _safe_best_known(problem),
        "improvements": engine.improvements,
        "audit_kills": engine.audit_kills,
        "pending_records": len(engine.pending_records),
        "llm": llm.stats(),
        "elapsed_s": round(time.time() - started, 1),
        "executor": getattr(stats, "__dict__", {}) if stats else {},
    }
    llm.close()

    # A find is only a find if someone else can check it. Export on the way out.
    best = ledger.best(summary["problem"])
    if best is not None:
        dest = Path(ledger_root) / "bundles" / summary["problem"]
        try:
            bundle = ledger.export_publishable(summary["problem"], dest)
            summary["bundle"] = str(bundle)
            logger.info("[evolve] BUNDLE: %s  (re-check with: python verify.py)", bundle)
        except Exception:  # noqa: BLE001 — a bundle that will not re-check is not shippable
            logger.exception("[evolve] export_publishable refused to ship the bundle")
    return summary


def _safe_best_known(problem: Problem) -> float:
    try:
        return float(problem.best_known())
    except Exception:  # noqa: BLE001
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run an evolve discovery campaign.")
    sub = p.add_subparsers(dest="target", required=True)

    ecc = sub.add_parser("ecc", help="best-known [n,k,d] linear codes")
    ecc.add_argument("--n", type=int, default=32)
    ecc.add_argument("--k", type=int, default=14)

    graph = sub.add_parser("graph", help="counterexample hunt on an open spectral conjecture")
    graph.add_argument("--conjecture", default="elphick_linz_wocjan")

    e143 = sub.add_parser("erdos143", help="Erdos #143 partial-sum probe (a BOUNDED side-run: a "
                                           "finite search can never settle it)")
    e143.add_argument("--n-max", type=int, default=1000)

    for s in (ecc, graph, e143):
        s.add_argument("--steps", type=int, default=500)
        s.add_argument("--workers", type=int, default=64)
        s.add_argument("--pool-size", type=int, default=8)
        s.add_argument("--islands", type=int, default=8)
        s.add_argument("--ledger-root", default=DEFAULT_LEDGER_ROOT)
        s.add_argument("--model", default=None)
        s.add_argument("--temperature", type=float, default=0.9)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    # httpx logs every request URL at INFO. Auth is in a header now, but a long campaign emits one
    # line per mutation and would bury the search behind HTTP noise regardless.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    problem = build_problem(args.target, args)
    logger.info("[evolve] target=%s best_known=%s", getattr(problem, "name", args.target),
                _safe_best_known(problem))

    summary = run_campaign(
        problem,
        steps=args.steps,
        workers=args.workers,
        pool_size=args.pool_size,
        islands=args.islands,
        ledger_root=args.ledger_root,
        model=args.model,
        temperature=args.temperature,
    )
    logger.info("[evolve] DONE %s", summary)
    # An improvement that survived the audit is the only thing that counts as a find.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
