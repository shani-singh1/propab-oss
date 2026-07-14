"""WS-2: containment tests for the sandboxed ProgramRunner.

The governing assumption is that mutated code is ADVERSARIAL BY ACCIDENT. Every test here feeds the
runner a program an LLM plausibly writes by mistake and asserts two things:

  1. it is contained (killed, capped, or refused), and
  2. it comes back as ``ExecResult(ok=False, ...)`` — a dead program is a normal event, never an
     exception, because an exception here would take down the search.
"""
from __future__ import annotations

import sys
import threading
import time

import pytest

from propab.evolve.program import ExecResult, Program
from propab.evolve.runner import (
    DEFAULT_ALLOWED_IMPORTS,
    SandboxLimits,
    SandboxProgramRunner,
)


def prog(code: str) -> Program:
    return Program(code=code)


# ------------------------------------------------------------------------------------------- #
# Happy paths — including the two contract ambiguities the runner has to resolve.
# ------------------------------------------------------------------------------------------- #
def test_trivial_program_returns_candidates(runner):
    res = runner.run(prog("def build():\n    return [1, 2, 4]\n"), timeout_s=5)
    assert res.ok, res.error
    # A returned list is ambiguous ("a candidate, or a list of candidates"), so the runner emits
    # BOTH readings and lets the exact verifier arbitrate. The whole list must be among them.
    assert [1, 2, 4] in res.candidates
    assert res.seconds > 0


def test_generator_is_an_unambiguous_batch(runner):
    res = runner.run(prog("def build():\n    for i in range(3):\n        yield [i, i]\n"), timeout_s=5)
    assert res.ok, res.error
    assert res.candidates == [[0, 0], [1, 1], [2, 2]]


def test_scalar_return_is_one_candidate(runner):
    res = runner.run(prog("def build():\n    return {'a': 1}\n"), timeout_s=5)
    assert res.ok, res.error
    assert res.candidates == [{"a": 1}]


def test_allowed_imports_work_and_numpy_crosses_as_json():
    # numpy must survive the process boundary. It crosses as JSON (never pickle -- unpickling data
    # from untrusted code would hand it the parent), so an ndarray arrives as nested lists.
    limits = SandboxLimits(preimport=("numpy",), memory_mb=1024)
    with SandboxProgramRunner(pool_size=1, limits=limits) as r:
        res = r.run(prog(
            "import numpy as np\n"
            "import math\n"
            "def build():\n"
            "    return np.array([[1, 2], [3, 4]], dtype=np.int64) * int(math.sqrt(4))\n"
        ), timeout_s=30)
    assert res.ok, res.error
    assert [[2, 4], [6, 8]] in res.candidates


def test_stdout_is_captured_and_bounded(runner, fast_limits):
    res = runner.run(prog(
        "def build():\n"
        "    for _ in range(50000):\n"
        "        print('x' * 100)\n"
        "    return [7]\n"
    ), timeout_s=20)
    assert res.ok, res.error
    assert [7] in res.candidates
    # ~5 MB of prints must not become 5 MB of parent memory.
    assert 0 < len(res.stdout) <= fast_limits.max_stdout_bytes


def test_runs_are_deterministic_even_when_the_program_forgets_to_seed(runner):
    # Workers are reused, so unseeded `random` would otherwise depend on which programs ran before
    # this one on this worker. The runner resets the RNG per run.
    code = "import random\ndef build():\n    return [random.randint(0, 10**6) for _ in range(5)]\n"
    first = runner.run(prog(code), timeout_s=5)
    second = runner.run(prog(code), timeout_s=5)
    assert first.ok and second.ok
    assert first.candidates == second.candidates


# ------------------------------------------------------------------------------------------- #
# Containment: wall clock.
# ------------------------------------------------------------------------------------------- #
def _warm(runner) -> None:
    """Force a live sandbox into the idle pool, so a timing assertion measures the TIMEOUT and not
    the cost of spawning a child (~0.5s here, and much more on a loaded box)."""
    assert runner.run(prog("def build():\n    return [1]\n"), timeout_s=30).ok


@pytest.mark.parametrize("code, kind", [
    ("def build():\n    while True:\n        pass\n", "a tight python loop"),
    # A blocking C call cannot be interrupted from inside the interpreter at all. Only killing the
    # process contains this one -- which is exactly why the PARENT owns the deadline.
    ("import time\ndef build():\n    time.sleep(600)\n", "a blocking C call"),
])
def test_a_hang_is_killed_on_the_wall_clock(runner, code, kind):
    _warm(runner)

    started = time.perf_counter()
    res = runner.run(prog(code), timeout_s=1.0)
    elapsed = time.perf_counter() - started

    assert res.ok is False, kind
    assert "timeout" in (res.error or ""), kind
    # The containment budget itself: the program was cut off at the deadline.
    assert res.seconds < 3.0, f"{kind}: ran {res.seconds:.1f}s past a 1s deadline"
    # And run() returned promptly -- killing the corpse happens on a reaper thread, not here.
    assert elapsed < 5.0, f"{kind}: a hung worker stalled its caller for {elapsed:.1f}s"


def test_the_pool_self_heals_after_a_timeout(runner):
    """A killed worker must not permanently remove capacity, and must never leak state forward."""
    runner.run(prog("def build():\n    while True:\n        pass\n"), timeout_s=1.0)
    res = runner.run(prog("def build():\n    return [42]\n"), timeout_s=5)
    assert res.ok, res.error
    assert [42] in res.candidates


# ------------------------------------------------------------------------------------------- #
# Containment: memory.
#
# The cap is proved by a CONTROL: the same allocation must succeed under a large cap and fail under
# a small one. Without the control, a MemoryError might just mean the machine ran out of RAM.
# ------------------------------------------------------------------------------------------- #
_ALLOC_600MB = (
    "def build():\n"
    "    blob = bytearray(600 * 1024 * 1024)\n"
    "    return [len(blob)]\n"
)


def test_allocation_succeeds_under_a_generous_memory_cap():
    with SandboxProgramRunner(
        pool_size=1, limits=SandboxLimits(preimport=(), memory_mb=2048)
    ) as r:
        res = r.run(prog(_ALLOC_600MB), timeout_s=60)
    assert res.ok, f"control failed — 600MB should fit under a 2GB cap: {res.error}"


def test_the_same_allocation_is_refused_under_a_tight_memory_cap():
    with SandboxProgramRunner(
        pool_size=1, limits=SandboxLimits(preimport=(), memory_mb=256)
    ) as r:
        res = r.run(prog(_ALLOC_600MB), timeout_s=60)
    assert res.ok is False, "600MB was allowed through a 256MB cap — the memory cap is not enforced"
    # Either the allocation is refused in-process (MemoryError) or the OS kills the child. Both are
    # containment; both come back as a normal failed ExecResult.
    assert "MemoryError" in (res.error or "") or "died" in (res.error or "")


def test_allocating_ten_gb_never_takes_down_the_parent():
    with SandboxProgramRunner(
        pool_size=1, limits=SandboxLimits(preimport=(), memory_mb=256)
    ) as r:
        res = r.run(prog("def build():\n    return [0] * (10**9)\n"), timeout_s=30)
        assert res.ok is False
        # ...and the runner is still usable afterwards.
        assert r.run(prog("def build():\n    return [1]\n"), timeout_s=10).ok


def test_runaway_recursion_does_not_crash_the_child(runner):
    # Unbounded recursion blows the C stack and hard-crashes CPython (no traceback) unless the
    # recursion limit is capped low enough to raise RecursionError first.
    res = runner.run(prog("def build():\n    def f(n):\n        return f(n + 1)\n    return f(0)\n"),
                     timeout_s=10)
    assert res.ok is False
    assert "RecursionError" in (res.error or "")


# ------------------------------------------------------------------------------------------- #
# Containment: imports, filesystem, network, processes.
# ------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", ["os", "sys", "socket", "subprocess", "shutil", "pathlib",
                                    "urllib", "requests", "ctypes", "importlib", "pickle"])
def test_dangerous_imports_are_refused(runner, module):
    res = runner.run(prog(f"import {module}\ndef build():\n    return [1]\n"), timeout_s=5)
    assert res.ok is False
    assert "not allowed" in (res.error or "")


def test_from_import_is_refused_too(runner):
    res = runner.run(prog("from os import listdir\ndef build():\n    return [1]\n"), timeout_s=5)
    assert res.ok is False
    assert "not allowed" in (res.error or "")


def test_the_allowlist_is_default_deny():
    assert "os" not in DEFAULT_ALLOWED_IMPORTS
    assert "socket" not in DEFAULT_ALLOWED_IMPORTS
    assert "subprocess" not in DEFAULT_ALLOWED_IMPORTS
    assert {"numpy", "math", "itertools", "random"} <= DEFAULT_ALLOWED_IMPORTS


@pytest.mark.parametrize("code", [
    "def build():\n    return open('pwned.txt', 'w').write('x')\n",
    "def build():\n    return eval('1+1')\n",
    "def build():\n    return exec('x=1')\n",
    "def build():\n    return __import__('os').listdir('.')\n",
    "def build():\n    return ().__class__.__bases__[0].__subclasses__()\n",
    "def build():\n    return getattr((), '__class__')\n",
    "def build():\n    return globals()\n",
])
def test_escape_attempts_are_refused(runner, code):
    res = runner.run(prog(code), timeout_s=5)
    assert res.ok is False
    assert "not allowed" in (res.error or ""), res.error


def test_no_file_io(runner, tmp_path):
    target = tmp_path / "should_not_exist.txt"
    res = runner.run(prog(
        f"def build():\n"
        f"    f = open({str(target)!r}, 'w')\n"
        f"    f.write('pwned')\n"
        f"    return [1]\n"
    ), timeout_s=5)
    assert res.ok is False
    assert not target.exists(), "the sandbox wrote to the filesystem"


def test_fork_bomb_is_contained_even_if_it_gets_past_the_static_screen():
    """The AST screen already refuses `import subprocess`. This proves the layer BELOW it.

    We hand the program `subprocess` explicitly, so the screen lets it through, and check that the OS
    boundary still holds. On Windows that is the Job Object's ActiveProcessLimit=1 (the child simply
    cannot create a process). On POSIX there is no cheap equivalent, so containment there is the
    wall-clock kill of the whole process *group* — either way, bounded and non-fatal.
    """
    limits = SandboxLimits(
        preimport=(),
        memory_mb=512,
        allowed_imports=DEFAULT_ALLOWED_IMPORTS | {"subprocess", "sys"},
    )
    bomb = (
        "import subprocess, sys\n"
        "def build():\n"
        "    kids = []\n"
        "    for _ in range(20):\n"
        "        kids.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']))\n"
        "    return [len(kids)]\n"
    )
    started = time.perf_counter()
    with SandboxProgramRunner(pool_size=1, limits=limits) as r:
        res = r.run(prog(bomb), timeout_s=5)
    elapsed = time.perf_counter() - started

    assert elapsed < 30, "the fork bomb was not contained in bounded time"
    if sys.platform == "win32":
        assert res.ok is False, "Job Object ActiveProcessLimit did not block process creation"
    # The parent survived and reported a normal result either way.
    assert isinstance(res, ExecResult)


# ------------------------------------------------------------------------------------------- #
# The cardinal rule: a broken program is data, never an exception.
# ------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("code", [
    "def build():\n    raise RuntimeError('boom')\n",
    "def build():\n    return None\n",
    "def build():\n    return lambda x: x\n",                 # not serializable
    "def build():\n    return [1/0]\n",
    "def build():\n    return undefined_name\n",
    "def nope():\n    return [1]\n",                          # no entrypoint
    "def build(:\n",                                          # syntax error
    "",                                                       # empty
    "def build():\n    return []\n",                          # empty list
    "def build():\n    import os\n    return [1]\n",          # import inside the function
    "def build():\n    return [float('nan'), float('inf')]\n",
])
def test_a_broken_program_is_a_normal_failed_result_not_an_exception(runner, code):
    res = runner.run(prog(code), timeout_s=5)
    assert isinstance(res, ExecResult)
    if not res.ok:
        assert res.error, "a failed run must say why — the mutator feeds on this string"


def test_the_runner_never_raises_on_anything(runner):
    """One pass over the whole adversarial zoo. Nothing may escape as an exception."""
    zoo = [
        "def build():\n    while True:\n        pass\n",
        "def build():\n    return [0] * (10**9)\n",
        "def build():\n    raise SystemExit(1)\n",
        "def build():\n    raise BaseException('nasty')\n",
        "def build():\n    return None\n",
        "import os\ndef build():\n    return [1]\n",
        "def build():\n    return ().__class__\n",
        "def build():\n    return {1: {2: {3: object()}}}\n",
        "\x00\x01\x02",
        "def build():\n    return [[[[[[[[[[1]]]]]]]]]]\n",
    ]
    for code in zoo:
        res = runner.run(prog(code), timeout_s=1.0)     # must not raise
        assert isinstance(res, ExecResult)
    # ...and the pool is still healthy.
    assert runner.run(prog("def build():\n    return [5]\n"), timeout_s=5).ok


# ------------------------------------------------------------------------------------------- #
# Concurrency.
# ------------------------------------------------------------------------------------------- #
def test_the_runner_is_thread_safe_under_hammering(fast_limits):
    """One runner, shared by every executor worker — so `run` must be safe from many threads."""
    n_threads, per_thread = 16, 4
    results: list[ExecResult] = []
    lock = threading.Lock()

    with SandboxProgramRunner(pool_size=8, limits=fast_limits) as r:
        def hammer(worker_id: int) -> None:
            for i in range(per_thread):
                # mix good programs with a crasher and a hang
                if i == 1:
                    code = "def build():\n    raise ValueError('x')\n"
                elif i == 2 and worker_id % 4 == 0:
                    code = "def build():\n    while True:\n        pass\n"
                else:
                    code = f"def build():\n    return [{worker_id}, {i}]\n"
                res = r.run(prog(code), timeout_s=2.0)
                with lock:
                    results.append(res)

        threads = [threading.Thread(target=hammer, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        assert not any(t.is_alive() for t in threads), "a worker thread wedged"

    assert len(results) == n_threads * per_thread
    assert all(isinstance(x, ExecResult) for x in results)
    assert any(x.ok for x in results), "nothing succeeded — the pool is broken, not just contended"


def test_stats_are_recorded(runner):
    before = runner.stats.snapshot()["runs"]
    runner.run(prog("def build():\n    return [1]\n"), timeout_s=5)
    runner.run(prog("def build():\n    raise ValueError('x')\n"), timeout_s=5)
    after = runner.stats.snapshot()
    assert after["runs"] == before + 2
    assert after["ok"] >= 1
    assert after["errors"] >= 1


def test_a_closed_runner_fails_soft(fast_limits):
    r = SandboxProgramRunner(pool_size=1, limits=fast_limits)
    r.close()
    res = r.run(prog("def build():\n    return [1]\n"), timeout_s=5)
    assert res.ok is False and "closed" in (res.error or "")
