"""Evolve — the massively-parallel executor (WS-2).

The whole method lives or dies on **evaluations/sec**. FunSearch-class search is a lottery bought in
bulk: the LLM's per-sample hit rate is low and roughly fixed, so the only lever we control is how
many samples we can verify per unit time. Parallelism here is not a nice-to-have, it is the method.

Concurrency model — **threads in the parent, processes in the sandbox** — and why that is not the
GIL mistake it looks like:

* The expensive, untrusted half of an evaluation (running the mutated program) happens **in a child
  process** (see `runner.SandboxProgramRunner`). The parent thread that called it is blocked on a
  pipe read, holding no GIL. So N parent threads really do drive N cores of program execution.
* The other half of a step — ``Mutator.mutate`` — is an LLM call. That is network I/O, and it is the
  *actual* wall-clock bottleneck of a real run (~1s, versus ~1ms for an evaluation). Threads overlap
  it perfectly; a process pool would just add IPC to a task that is already waiting on a socket.
* What genuinely does run under the parent's GIL is ``Problem.verify``. That is deliberate: the
  contract requires the verifier to be cheap and exact, it must stay in the trusted parent (never
  inside the sandbox), and numpy-based verifiers release the GIL anyway. If a target ever shows up
  with a verifier heavy enough to serialize the parent, that is a signal the *target* is wrong for
  this engine — not a signal to move the verifier into a process pool.

A ``ProcessPoolExecutor`` was rejected outright for the sandbox: it **cannot kill a running task**.
A single ``while True:`` in a mutated program would park one of its workers forever, and the pool
would silently bleed capacity until the search stalled. Containment has to own the process lifecycle,
so the pool is hand-rolled in `runner`.

Resilience contract: one dead worker never kills the run. Every fault — a crashed program, a killed
sandbox, a verifier that raised despite promising not to, an exception in a worker thread — is
counted and stepped over. The run stops when you ask it to, not because something broke.
"""
from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

from .engine import Engine
from .problem import INVALID, Verdict
from .program import ExecResult, Program

logger = logging.getLogger(__name__)

_FEED_DONE = object()      # feeder -> worker: no more programs
_WORKER_DONE = object()    # worker -> consumer: this worker has retired


@dataclass(frozen=True)
class EvalOutcome:
    """One completed evaluation. ``error`` is set only for an *executor-level* fault; a program that
    crashed or timed out is a normal outcome and reports through ``result.ok``/``result.error``."""

    program: Program
    verdict: Verdict
    result: ExecResult
    seconds: float = 0.0
    error: str | None = None

    @property
    def improved(self) -> bool:
        return self.verdict.valid


@dataclass
class ExecutorStats:
    """Live counters. ``evals_per_sec`` is THE number to watch."""

    steps: int = 0            # evaluations completed (ok or not)
    executed: int = 0         # programs that ran to completion in the sandbox
    valid: int = 0            # candidates the verifier accepted
    crashed: int = 0          # programs that raised / returned garbage
    timed_out: int = 0        # programs killed on the wall clock
    faults: int = 0           # executor-level faults (engine/verifier misbehaved)
    best_score: float = float("-inf")
    started_at: float = field(default_factory=time.time)
    elapsed: float = 0.0
    recent_errors: list[str] = field(default_factory=list)

    @property
    def evals_per_sec(self) -> float:
        return (self.steps / self.elapsed) if self.elapsed > 0 else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "steps": self.steps, "executed": self.executed, "valid": self.valid,
            "crashed": self.crashed, "timed_out": self.timed_out, "faults": self.faults,
            "best_score": self.best_score, "elapsed": round(self.elapsed, 2),
            "evals_per_sec": round(self.evals_per_sec, 1),
        }

    def format(self) -> str:
        return (
            f"[evolve] {self.steps} evals in {self.elapsed:6.1f}s "
            f"| {self.evals_per_sec:7.1f} eval/s "
            f"| valid={self.valid} crashed={self.crashed} timeout={self.timed_out} "
            f"faults={self.faults} | best={self.best_score:g}"
        )


ProgressHook = Callable[[ExecutorStats], None]


def log_progress(stats: ExecutorStats) -> None:
    """Default progress hook: a long run must be monitorable, never a silent hang."""
    logger.info("%s", stats.format())


class ParallelExecutor:
    """Drives ``Engine.evaluate`` (or ``Engine.step``) across many concurrent workers.

    ``Engine.evaluate`` is pure by contract, so it is safe to call from every worker at once; all the
    shared mutable state (islands, ledger) stays behind WS-1's ``Engine.step``.

    Sizing — the two knobs are deliberately independent, and they are not the same number:

    * ``workers`` (here) = how many steps are in flight. For a real run each step is dominated by an
      LLM call (~1s), so this wants to be LARGE (64+) to keep the mutation pipeline full.
    * ``pool_size`` (on the runner) = how many programs can be *executing* at once. Programs are
      CPU-bound, so this wants to be around the core count. Threads that find no free sandbox block,
      which is correct backpressure — not a misconfiguration.

    So ``workers=64, pool_size=8`` is the right shape for an LLM-driven run on an 8-core box, and
    ``workers == pool_size`` is the right shape for a pure evaluate-only sweep (no mutator in the
    loop), where nothing else overlaps the sandbox.
    """

    def __init__(
        self,
        engine: Engine,
        *,
        workers: int | None = None,
        on_progress: ProgressHook | None = None,
        progress_interval_s: float = 2.0,
        max_recent_errors: int = 20,
    ) -> None:
        self.engine = engine
        self.workers = int(workers or engine.config.workers)
        self.on_progress = on_progress
        self.progress_interval_s = progress_interval_s

        self._stats = ExecutorStats()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._errors: deque[str] = deque(maxlen=max_recent_errors)

        pool = getattr(engine.runner, "_pool", None)
        self.pool_size = getattr(pool, "size", None)
        if isinstance(self.pool_size, int) and self.pool_size < self.workers:
            # Informational, NOT a misconfiguration: with an LLM in the loop this is the right shape
            # (most workers are waiting on the model, not on a sandbox). It only costs you throughput
            # on an evaluate-only sweep, where nothing overlaps the sandbox.
            logger.info(
                "[evolve] %d workers over %d sandboxes: at most %d programs execute at once; the "
                "rest queue. Correct when a mutator overlaps the wait; raise pool_size for a pure "
                "evaluate-only sweep.",
                self.workers, self.pool_size, self.pool_size,
            )

    # -- control ---------------------------------------------------------------------------- #
    @property
    def stats(self) -> ExecutorStats:
        with self._lock:
            self._stats.elapsed = time.time() - self._stats.started_at
            self._stats.recent_errors = list(self._errors)
            return self._stats

    def stop(self) -> None:
        """Graceful stop: workers finish the evaluation in flight, then retire."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # -- the two drive modes ------------------------------------------------------------------ #
    def imap(self, programs: Iterable[Program]) -> Iterator[EvalOutcome]:
        """Stream evaluations of ``programs`` across ``workers``, yielding outcomes as they land.

        The source is consumed lazily behind a bounded queue, so an unbounded generator of programs
        streams instead of materializing, and a slow consumer applies backpressure rather than
        growing the heap.
        """
        in_q: queue.Queue[Any] = queue.Queue(maxsize=self.workers * 2)
        out_q: queue.Queue[Any] = queue.Queue()

        self._reset()
        feeder = threading.Thread(
            target=self._feed, args=(programs, in_q), name="evolve-feeder", daemon=True
        )
        pool = [
            threading.Thread(
                target=self._consume, args=(in_q, out_q), name=f"evolve-w{i}", daemon=True
            )
            for i in range(self.workers)
        ]
        monitor = self._start_monitor()
        feeder.start()
        for thread in pool:
            thread.start()

        try:
            retired = 0
            while retired < self.workers:
                item = out_q.get()
                if item is _WORKER_DONE:
                    retired += 1
                    continue
                yield item
        finally:
            self._stop.set()                      # a consumer that walks away must not orphan threads
            self._shutdown(monitor, [feeder, *pool])

    def evaluate_all(self, programs: Iterable[Program]) -> list[EvalOutcome]:
        """``imap`` drained. Convenience for tests and small batches."""
        return list(self.imap(programs))

    def run_steps(
        self,
        *,
        max_steps: int | None = None,
        step_fn: Callable[[], Verdict] | None = None,
    ) -> ExecutorStats:
        """Drive full engine steps (mutate -> evaluate -> insert) concurrently until stopped.

        This is the production loop. ``step_fn`` defaults to ``Engine.step`` (WS-1); the shared
        population state it touches is the Engine's problem, not ours.
        """
        step = step_fn or self.engine.step
        budget = max_steps if max_steps is not None else self.engine.config.max_steps

        self._reset()
        claims = itertools.count()
        aborted: list[BaseException] = []

        def worker() -> None:
            while not self._stop.is_set():
                if budget is not None and next(claims) >= budget:
                    return
                started = time.perf_counter()
                try:
                    verdict = step()
                except NotImplementedError as exc:
                    # WS-1 has not landed Engine.step. Fail loudly and immediately -- silently
                    # counting a million NotImplementedErrors as "faults" would look like a working
                    # run that finds nothing, which is the worst possible failure mode here.
                    aborted.append(exc)
                    self._stop.set()
                    return
                except BaseException as exc:  # noqa: BLE001 - one bad step never kills the run
                    self._fault(exc)
                    continue
                self._tally(verdict, None, time.perf_counter() - started)

        monitor = self._start_monitor()
        pool = [
            threading.Thread(target=self._guard(worker), name=f"evolve-s{i}", daemon=True)
            for i in range(self.workers)
        ]
        for thread in pool:
            thread.start()
        try:
            while any(t.is_alive() for t in pool):
                for thread in pool:
                    thread.join(timeout=0.2)
        except KeyboardInterrupt:               # Ctrl-C is a graceful stop, not a stack trace
            logger.warning("[evolve] interrupted — stopping workers")
            self._stop.set()
            for thread in pool:
                thread.join(timeout=5)
        finally:
            self._shutdown(monitor, [])

        if aborted:
            raise RuntimeError(
                "Engine.step() is not implemented (WS-1). Use ParallelExecutor.imap(programs) to "
                "drive Engine.evaluate directly, or pass step_fn=..."
            ) from aborted[0]
        return self.stats

    # -- internals ---------------------------------------------------------------------------- #
    def _reset(self) -> None:
        with self._lock:
            self._stats = ExecutorStats()
            self._errors.clear()
        self._stop.clear()

    def _guard(self, fn: Callable[[], None]) -> Callable[[], None]:
        """A worker thread must never die of an unhandled exception — that would silently shrink the
        pool and make the run look merely 'slow'."""

        def _wrapped() -> None:
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001
                self._fault(exc)
                logger.exception("[evolve] worker thread died")

        return _wrapped

    def _feed(self, programs: Iterable[Program], in_q: queue.Queue[Any]) -> None:
        try:
            for program in programs:
                while not self._stop.is_set():
                    try:
                        in_q.put(program, timeout=0.1)   # bounded: backpressure, not a memory leak
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    break
        except BaseException as exc:  # noqa: BLE001 - a bad program source is a fault, not a crash
            self._fault(exc)
        finally:
            # One sentinel per worker, delivered reliably: a dropped sentinel would leave a worker
            # polling an empty queue forever and imap() would never return. If we are stopping, the
            # workers exit on the stop flag anyway, so abandoning delivery is safe (and must not
            # block here -- nobody is draining).
            for _ in range(self.workers):
                while not self._stop.is_set():
                    try:
                        in_q.put(_FEED_DONE, timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def _consume(self, in_q: queue.Queue[Any], out_q: queue.Queue[Any]) -> None:
        try:
            while not self._stop.is_set():
                try:
                    program = in_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if program is _FEED_DONE:
                    break
                out_q.put(self._evaluate_one(program))
        except BaseException as exc:  # noqa: BLE001
            self._fault(exc)
            logger.exception("[evolve] worker thread died")
        finally:
            out_q.put(_WORKER_DONE)

    def _evaluate_one(self, program: Program) -> EvalOutcome:
        """One evaluation, fully defensive. ``Engine.evaluate`` is pure, so this is safe to run from
        every worker at once."""
        started = time.perf_counter()
        try:
            verdict, result = self.engine.evaluate(program)
        except BaseException as exc:  # noqa: BLE001 - the runner promises not to raise; verify() too.
            self._fault(exc)          # if either breaks its promise, we survive it anyway.
            return EvalOutcome(
                program=program, verdict=INVALID,
                result=ExecResult(ok=False, error=f"{type(exc).__name__}: {exc}"),
                seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
        seconds = time.perf_counter() - started
        self._tally(verdict, result, seconds)
        return EvalOutcome(program=program, verdict=verdict, result=result, seconds=seconds)

    def _tally(self, verdict: Verdict, result: ExecResult | None, seconds: float) -> None:
        with self._lock:
            stats = self._stats
            stats.steps += 1
            if result is not None:
                if result.ok:
                    stats.executed += 1
                else:
                    err = result.error or ""
                    if err.startswith("timeout:"):
                        stats.timed_out += 1
                    else:
                        stats.crashed += 1
            if verdict.valid:
                stats.valid += 1
                if verdict.score > stats.best_score:
                    stats.best_score = verdict.score

    def _fault(self, exc: BaseException) -> None:
        with self._lock:
            self._stats.faults += 1
            self._errors.append(f"{type(exc).__name__}: {exc}")

    def _start_monitor(self) -> threading.Event | None:
        """Stream progress so a long run can be watched. A search that has silently wedged and a
        search that is merely slow look identical without this."""
        hook = self.on_progress
        if hook is None or self.progress_interval_s <= 0:
            return None
        done = threading.Event()

        def _tick() -> None:
            while not done.wait(self.progress_interval_s):
                try:
                    hook(self.stats)
                except Exception:  # noqa: BLE001 - a broken progress hook must not stop the search
                    logger.exception("[evolve] progress hook raised")

        threading.Thread(target=_tick, name="evolve-monitor", daemon=True).start()
        return done

    def _shutdown(self, monitor: threading.Event | None, threads: list[threading.Thread]) -> None:
        for thread in threads:
            thread.join(timeout=5)
        if monitor is not None:
            monitor.set()
        if self.on_progress is not None:
            try:
                self.on_progress(self.stats)      # always emit a final line
            except Exception:  # noqa: BLE001
                logger.exception("[evolve] progress hook raised")


__all__ = [
    "EvalOutcome",
    "ExecutorStats",
    "ParallelExecutor",
    "ProgressHook",
    "log_progress",
]
