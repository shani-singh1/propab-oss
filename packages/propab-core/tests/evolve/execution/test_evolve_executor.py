"""WS-2: tests for the parallel executor.

Two properties carry the whole design:

  * **Throughput** — the method is a lottery bought in bulk, so evaluations/sec IS the result.
    Concurrency must actually be concurrent, not a thread pool serialized behind the GIL.
  * **Resilience** — one dead worker never kills the run. Programs crash, sandboxes get OOM-killed,
    and (here) even the verifier breaks its contract and raises; the run keeps going and counts it.
"""
from __future__ import annotations

import itertools
import threading
import time

import pytest

from propab.evolve.executor import EvalOutcome, ExecutorStats, ParallelExecutor
from propab.evolve.program import Program
from propab.evolve.runner import SandboxLimits, SandboxProgramRunner

GOOD = "def build():\n    return [{a}, {b}]\n"
CRASH = "def build():\n    raise RuntimeError('boom')\n"
HANG = "def build():\n    while True:\n        pass\n"


def programs(n: int) -> list[Program]:
    return [Program(code=GOOD.format(a=i, b=i + 1)) for i in range(n)]


@pytest.fixture(scope="module")
def pool_runner(request):
    """8 sandboxes, shared across the module — spawning children is what costs, not running them."""
    r = SandboxProgramRunner(
        pool_size=8,
        limits=SandboxLimits(preimport=(), memory_mb=512, spawn_timeout_s=60.0),
    )
    request.addfinalizer(r.close)
    return r


# ------------------------------------------------------------------------------------------- #
# Correctness.
# ------------------------------------------------------------------------------------------- #
def test_evaluate_all_scores_every_program(pool_runner, engine_for):
    engine = engine_for(pool_runner, workers=8)
    executor = ParallelExecutor(engine, workers=8)

    outcomes = executor.evaluate_all(programs(24))

    assert len(outcomes) == 24
    assert all(isinstance(o, EvalOutcome) for o in outcomes)
    assert all(o.verdict.valid for o in outcomes), "every GOOD program should verify"
    # program i returns [i, i+1] -> score 2i+1; the best of 0..23 is 2*23+1 = 47
    assert executor.stats.best_score == 47.0
    assert executor.stats.steps == 24
    assert executor.stats.valid == 24


def test_imap_streams_results_as_they_land(pool_runner, engine_for):
    engine = engine_for(pool_runner, workers=4)
    executor = ParallelExecutor(engine, workers=4)

    seen = 0
    for outcome in executor.imap(programs(12)):
        assert isinstance(outcome, EvalOutcome)
        seen += 1
    assert seen == 12


def test_imap_consumes_an_unbounded_source_lazily_and_stops_on_demand(pool_runner, engine_for):
    """A real run is driven by an endless stream of mutations. It must stream, not materialize —
    and `stop()` must actually end it."""
    engine = engine_for(pool_runner, workers=4)
    executor = ParallelExecutor(engine, workers=4)

    endless = (Program(code=GOOD.format(a=i, b=1)) for i in itertools.count())

    seen = 0
    for _ in executor.imap(endless):
        seen += 1
        if seen == 20:
            executor.stop()
    assert seen >= 20
    assert executor.stopped


# ------------------------------------------------------------------------------------------- #
# Resilience: nothing in the zoo may end the run.
# ------------------------------------------------------------------------------------------- #
def test_a_mix_of_crashing_hanging_and_good_programs_completes(pool_runner, engine_for):
    engine = engine_for(pool_runner, workers=8, timeout_s=1.0)
    executor = ParallelExecutor(engine, workers=8)

    batch = [
        *programs(8),
        Program(code=CRASH),
        Program(code=HANG),
        Program(code="def build():\n    return None\n"),
        Program(code="import os\ndef build():\n    return [1]\n"),
        Program(code="def build():\n    return [0] * (10**9)\n"),
        *programs(4),
    ]
    outcomes = executor.evaluate_all(batch)
    stats = executor.stats

    assert len(outcomes) == len(batch), "the run must not lose evaluations to bad programs"
    assert stats.steps == len(batch)
    assert stats.valid == 12, "all 12 good programs still scored"
    assert stats.timed_out >= 1, "the hang was counted as a timeout"
    assert stats.crashed >= 3, "the crashers were counted"
    assert stats.faults == 0, "a bad *program* is not an executor fault"


def test_a_verifier_that_breaks_its_contract_does_not_kill_the_run(
    pool_runner, engine_for, exploding_problem
):
    """`Problem.verify` promises never to raise. This one raises anyway. The run must survive."""
    engine = engine_for(pool_runner, workers=4, problem=exploding_problem)
    executor = ParallelExecutor(engine, workers=4)

    outcomes = executor.evaluate_all(programs(8))

    assert len(outcomes) == 8, "the run continued despite a verifier that raises every time"
    assert executor.stats.faults == 8
    assert all(o.error and "verifier exploded" in o.error for o in outcomes)
    assert executor.stats.valid == 0


def test_stop_is_graceful(pool_runner, engine_for):
    engine = engine_for(pool_runner, workers=4)
    executor = ParallelExecutor(engine, workers=4)

    def stop_soon() -> None:
        time.sleep(0.3)
        executor.stop()

    threading.Thread(target=stop_soon, daemon=True).start()
    outcomes = executor.evaluate_all(
        Program(code=GOOD.format(a=i, b=1)) for i in range(100_000)
    )
    assert executor.stopped
    assert len(outcomes) < 100_000, "stop() did not actually stop the run"
    assert all(isinstance(o, EvalOutcome) for o in outcomes)


# ------------------------------------------------------------------------------------------- #
# Observability + throughput.
# ------------------------------------------------------------------------------------------- #
def test_progress_is_streamed_so_a_long_run_is_never_a_silent_hang(pool_runner, engine_for):
    engine = engine_for(pool_runner, workers=4)
    ticks: list[ExecutorStats] = []
    executor = ParallelExecutor(
        engine, workers=4, on_progress=ticks.append, progress_interval_s=0.05
    )

    executor.evaluate_all(programs(40))

    assert ticks, "no progress was ever emitted"
    assert ticks[-1].steps == 40
    assert "eval/s" in ticks[-1].format()


def test_throughput_counter_is_live(pool_runner, engine_for):
    engine = engine_for(pool_runner, workers=8)
    executor = ParallelExecutor(engine, workers=8)
    executor.evaluate_all(programs(40))

    stats = executor.stats
    assert stats.steps == 40
    assert stats.elapsed > 0
    assert stats.evals_per_sec > 0
    assert stats.snapshot()["evals_per_sec"] == pytest.approx(stats.evals_per_sec, rel=0.1)


def test_concurrency_is_real_not_serialized_behind_the_gil(pool_runner, engine_for):
    """The load-bearing claim of the whole design.

    Each program burns real CPU *inside its own process*, so N parent threads really do drive N cores
    — the parent thread is parked on a pipe read, holding no GIL. Same batch, 1 worker vs 8: if the
    speedup isn't there, the parallelism is a lie and the method's throughput ceiling collapses.
    """
    busy = (
        "def build():\n"
        "    total = 0\n"
        "    for i in range(3_000_000):\n"
        "        total += i * i\n"
        "    return [total % 1000]\n"
    )
    batch = [Program(code=busy)] * 8
    engine = engine_for(pool_runner, workers=8, timeout_s=120.0)

    ParallelExecutor(engine, workers=8).evaluate_all(batch)      # warm every sandbox in the pool

    t0 = time.perf_counter()
    ParallelExecutor(engine, workers=1).evaluate_all(batch)
    serial_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ParallelExecutor(engine, workers=8).evaluate_all(batch)
    parallel_s = time.perf_counter() - t0

    speedup = serial_s / parallel_s
    # A quiet 8-core box gives ~6x. Assert well below that: this must catch "no parallelism at all",
    # not police the scheduler on a loaded CI machine.
    assert speedup > 2.5, (
        f"no real parallelism: 8 evals took {serial_s:.2f}s at 1 worker and {parallel_s:.2f}s at "
        f"8 workers ({speedup:.2f}x)"
    )


# ------------------------------------------------------------------------------------------- #
# Wiring to the Engine.
# ------------------------------------------------------------------------------------------- #
def test_workers_default_to_the_engine_config(pool_runner, engine_for):
    engine = engine_for(pool_runner, workers=64)
    assert ParallelExecutor(engine).workers == 64


def test_run_steps_says_so_plainly_when_ws1_has_not_landed(pool_runner, engine_for):
    """Engine.step is WS-1's. Until it exists, run_steps must fail loudly — silently counting a
    million NotImplementedErrors as 'faults' would look like a working run that finds nothing."""
    engine = engine_for(pool_runner, workers=2)
    executor = ParallelExecutor(engine, workers=2)

    with pytest.raises(RuntimeError, match="Engine.step"):
        executor.run_steps(max_steps=10)


def test_run_steps_drives_a_supplied_step_fn(pool_runner, engine_for):
    """The production loop, with WS-1's step stubbed in."""
    engine = engine_for(pool_runner, workers=4)
    executor = ParallelExecutor(engine, workers=4)
    counter = itertools.count()

    def step():
        i = next(counter)
        verdict, _ = engine.evaluate(Program(code=GOOD.format(a=i, b=1)))
        return verdict

    stats = executor.run_steps(max_steps=20, step_fn=step)

    assert stats.steps == 20
    assert stats.valid == 20
    assert stats.evals_per_sec > 0


def test_a_step_fn_that_raises_never_kills_the_run(pool_runner, engine_for):
    engine = engine_for(pool_runner, workers=4)
    executor = ParallelExecutor(engine, workers=4)
    counter = itertools.count()

    def flaky():
        i = next(counter)
        if i % 3 == 0:
            raise RuntimeError("flaky step")
        verdict, _ = engine.evaluate(Program(code=GOOD.format(a=i, b=1)))
        return verdict

    stats = executor.run_steps(max_steps=30, step_fn=flaky)

    assert stats.faults >= 1, "the raising steps were counted"
    assert stats.steps >= 1, "the healthy steps still landed"
    assert stats.recent_errors and "flaky step" in stats.recent_errors[0]
