"""Evolve — the sandboxed ProgramRunner (WS-2).

Mutated code is **adversarial by accident**: it will infinite-loop, allocate forever, raise, return
garbage, and (rarely, because an LLM pasted a training script) try to open a socket. A hung program
does not just lose one sample — it parks a worker forever and stalls the whole search. So the single
load-bearing property here is *containment*: every program run is bounded in wall-clock and in
memory, and a program that dies is a NORMAL event that returns ``ExecResult(ok=False, ...)``.

Design (and why it differs from ``math_combinatorics/discovery/sandbox_exec.py``, which it is
otherwise modelled on):

* ``sandbox_exec`` spawns a **fresh interpreter per construction**. That is right when you run a
  handful of constructions per LLM call. It is fatal here: we run programs *millions* of times, and
  a bare ``python -c pass`` costs ~50-450ms depending on the host. That is a hard ceiling of ~2-20
  evals/sec/worker before any useful work happens.
* So we keep a **pool of persistent sandbox children**. Each child boots once, imports numpy once,
  and then serves programs over a line-delimited JSON protocol on stdin/stdout. Steady-state cost of
  one program is an ``exec`` + a JSON round-trip, not a process spawn.

The containment layers (defence in depth — layers 1-2 are a *screen*, layers 3-5 are the real
boundary):

1. **Static AST screen** — rejects non-allowlisted imports, dunder attribute access (the classic
   ``().__class__.__bases__`` escape) and the dangerous builtins, before a single opcode runs.
2. **Restricted builtins + guarded ``__import__``** — ``open``/``eval``/``exec``/``compile``/
   ``getattr`` are simply absent (``NameError``), and imports are default-deny against an allowlist
   (numpy/math/itertools/random and friends). ``socket`` is additionally neutered inside the child.
3. **Separate process** — the program runs out-of-process, so a hard crash or an OOM cannot take the
   search down. The trusted verifier always runs in the PARENT, on returned data, never in the
   sandbox.
4. **Hard memory cap** — POSIX: ``RLIMIT_AS``/``RLIMIT_DATA`` (plus ``RLIMIT_FSIZE=0``, so the child
   cannot write files at all). Windows: a **Job Object** with ``ProcessMemoryLimit`` and
   ``ActiveProcessLimit=1``, which also contains fork bombs and kills the child if we die.
   Because the pool is persistent, the limit is applied at spawn time — long before any untrusted
   code arrives — so there is no assign-vs-allocate race.
5. **Hard wall-clock timeout, enforced by the parent** — the parent owns the deadline and *kills*
   the child on overrun (process group on POSIX, job object on Windows). This is the universal
   backstop: it catches the infinite loop, the 10GB allocation the OS was too generous to refuse,
   and anything the layers above missed. A killed worker is discarded and respawned; it is never
   reused, so a timed-out run can never contaminate the next one.

Candidates cross the process boundary as **JSON**, never pickle — unpickling data produced by
untrusted code would hand it the parent. This matches the Ledger contract, which already requires a
candidate to be JSON-serializable. Consequence to know: tuples/sets/ndarrays arrive as lists.

Documented limits: this is a robust guard against accidental damage and the common escapes, not a
hostile-multi-tenant boundary. Allowing ``numpy`` transitively allows ``numpy.load``/``numpy.save``;
on POSIX ``RLIMIT_FSIZE=0`` blocks the write, on Windows it does not. If we ever run genuinely
hostile code, put it in a container.
"""
from __future__ import annotations

import ast
import contextlib
import itertools
import json
import os
import queue
import random
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

# --------------------------------------------------------------------------------------------- #
# Wire protocol + policy. Shared by parent and child, so it must not depend on anything but stdlib.
# --------------------------------------------------------------------------------------------- #
_PROTO_PREFIX = "@@EVOLVE@@"        # tags the one response line, so stray child output is ignorable
_READY = "@@EVOLVE-READY@@"         # child -> parent: limits applied, imports warm, send work
_CHILD_FLAG = "--evolve-sandbox-child"

# Duplicated (not imported) from .program so the child never imports propab: see _child_main().
# The parent asserts these agree at import time, so the contract cannot drift silently.
_ENTRYPOINT = "build"

#: Default-deny import allowlist. Roots only — ``numpy.linalg`` is allowed because ``numpy`` is.
DEFAULT_ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "array", "bisect", "cmath", "collections", "copy", "dataclasses", "decimal", "enum",
    "fractions", "functools", "heapq", "itertools", "math", "numbers", "operator", "random",
    "re", "statistics", "string", "time", "typing",
    "numpy",
})

#: Names the static screen refuses outright (they are also absent from the sandbox builtins).
_FORBIDDEN_NAMES: frozenset[str] = frozenset({
    "__import__", "eval", "exec", "compile", "open", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr", "memoryview", "breakpoint", "help", "exit",
    "quit", "object", "type", "super", "classmethod", "staticmethod", "property",
})

_MAX_DEPTH = 40          # candidate nesting depth (kills self-referential structures)
_MAX_NODES = 5_000_000   # candidate node budget (bounds serialization cost)


# --------------------------------------------------------------------------------------------- #
# CHILD SIDE.
#
# Everything from here to _child_main() runs INSIDE the sandbox process. It must import nothing
# from propab: the child is launched as a bare script (see the __package__ guard below), which keeps
# its startup at bare-interpreter cost and keeps propab's modules out of reach of untrusted code.
# --------------------------------------------------------------------------------------------- #
class _BoundedWriter:
    """stdout sink for the program. Bounded, so ``while True: print(x)`` cannot eat the heap."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._parts: list[str] = []
        self._size = 0

    def write(self, s: str) -> int:
        if self._size < self._limit:
            self._parts.append(s[: self._limit - self._size])
            self._size += len(s)
        return len(s)

    def flush(self) -> None:  # pragma: no cover - stdout protocol
        return None

    def value(self) -> str:
        return "".join(self._parts)


def _safe_builtins(allowed_imports: frozenset[str]) -> dict[str, Any]:
    """The only builtins a program can see. Default-deny: anything absent raises NameError."""
    import builtins as _b

    allow = (
        # data + numerics
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr",
        "complex", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset", "hash",
        "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
        "next", "oct", "ord", "pow", "print", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "zip",
        # exceptions: LLM-written code uses try/except constantly. Without these, `except ValueError`
        # is itself a NameError -- which would show up as a mystery failure rather than a real one.
        "ArithmeticError", "AssertionError", "AttributeError", "BaseException", "Exception",
        "FloatingPointError", "IndexError", "KeyError", "LookupError", "MemoryError", "NameError",
        "NotImplementedError", "OverflowError", "RecursionError", "RuntimeError", "StopIteration",
        "TypeError", "ValueError", "ZeroDivisionError",
    )
    safe: dict[str, Any] = {n: getattr(_b, n) for n in allow if hasattr(_b, n)}
    safe.update({"True": True, "False": False, "None": None})
    # `class` statements compile to a __build_class__ call; without it no program may define a class.
    safe["__build_class__"] = _b.__build_class__
    safe["__import__"] = _make_guarded_import(allowed_imports)
    return safe


def _make_guarded_import(allowed_imports: frozenset[str]) -> Callable[..., Any]:
    """`import x` compiles to a __import__ call, so the allowlist is enforced here as well as in the
    AST screen. Belt and braces: the screen catches the static case, this catches everything else."""
    import builtins as _b

    real_import = _b.__import__

    def _guarded(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if level:
            raise ImportError("relative imports are not allowed in the evolve sandbox")
        root = str(name).split(".")[0]
        if root not in allowed_imports:
            raise ImportError(
                f"import of {name!r} is not allowed in the evolve sandbox "
                f"(allowed: {', '.join(sorted(allowed_imports))})"
            )
        return real_import(name, globals, locals, fromlist, level)

    return _guarded


def _screen_source(code: str, allowed_imports: frozenset[str]) -> None:
    """Static AST screen. Raises ``ValueError`` with a structural reason if the source is unsafe.

    Runs before a single opcode does. Blocks non-allowlisted imports, dunder attribute access, dunder
    names, and the forbidden builtins.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed_imports:
                    raise ValueError(f"import of '{alias.name}' is not allowed in the sandbox")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("relative imports are not allowed in the sandbox")
            root = (node.module or "").split(".")[0]
            if root not in allowed_imports:
                raise ValueError(f"import of '{node.module}' is not allowed in the sandbox")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError(f"dunder attribute access '{node.attr}' is not allowed")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id.endswith("__"):
                raise ValueError(f"dunder name '{node.id}' is not allowed")
            if node.id in _FORBIDDEN_NAMES:
                raise ValueError(f"use of '{node.id}' is not allowed in the sandbox")


def _to_jsonable(obj: Any, budget: list[int], depth: int = 0) -> Any:
    """Coerce a candidate into JSON-native data, in the CHILD.

    JSON, never pickle: the parent must never deserialize code-bearing data produced by untrusted
    code. numpy arrays/scalars go through ``.tolist()``; tuples and sets become lists.
    """
    if depth > _MAX_DEPTH:
        raise ValueError(f"candidate nested deeper than {_MAX_DEPTH} levels")
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError(f"candidate exceeds the {_MAX_NODES} node budget")

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    # numpy ndarray and numpy scalars both expose .tolist() -> pure-python nesting.
    tolist = getattr(type(obj), "tolist", None)
    if callable(tolist):
        return _to_jsonable(tolist(obj), budget, depth + 1)
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(x, budget, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, budget, depth + 1) for k, v in obj.items()}
    raise TypeError(f"candidate holds a non-JSON-serializable value of type '{type(obj).__name__}'")


def _expand_candidates(ret: Any, *, max_expand: int, max_candidates: int) -> list[Any]:
    """Resolve the contract's "return a candidate, **or a list of candidates**" ambiguity.

    A bare list is genuinely ambiguous: is ``[[1,0],[0,1]]`` one generator matrix, or two candidates?
    Guessing wrong silently throws away real results. So we do not guess — when the return is a list
    we emit *both* readings (the whole list as one candidate, then each element as a candidate) and
    let the verifier arbitrate. This is safe precisely because ``Problem.verify`` is contractually
    exact and safe on garbage: the wrong reading scores invalid, and ``Engine.evaluate`` takes the
    max. Cost is one extra (cheap) verify per program.

    A **generator** is unambiguous — it is always a batch. Prefer ``yield`` in seed programs.
    """
    if ret is None:
        raise ValueError(f"{_ENTRYPOINT}() returned None")

    if hasattr(ret, "__next__"):                       # generator/iterator == an explicit batch
        batch = list(itertools.islice(ret, max_candidates + 1))
        if len(batch) > max_candidates:
            raise ValueError(f"{_ENTRYPOINT}() yielded more than {max_candidates} candidates")
        if not batch:
            raise ValueError(f"{_ENTRYPOINT}() yielded no candidates")
        return batch

    if not isinstance(ret, (list, tuple)):             # ndarray, dict, scalar... -> one candidate
        return [ret]

    seq = list(ret)
    out: list[Any] = [ret]                             # reading A: the whole thing is ONE candidate
    if 0 < len(seq) <= max_expand:                     # reading B: it is a LIST of candidates
        out.extend(seq)
    return out[:max_candidates]


def _execute_program(
    code: str,
    *,
    allowed_imports: frozenset[str],
    seed: int = 0,
    max_expand: int = 512,
    max_candidates: int = 4096,
    max_stdout_bytes: int = 4096,
    max_payload_bytes: int = 8_000_000,
) -> dict[str, Any]:
    """Screen, exec, call ``build()``, and serialize the candidates. Never raises.

    Returns the child's wire response. ``fatal`` means the process state is suspect (MemoryError,
    RecursionError) and the parent must retire the worker rather than reuse it.
    """
    t0 = time.perf_counter()
    sink = _BoundedWriter(max_stdout_bytes)

    def _fail(exc: BaseException) -> dict[str, Any]:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "stdout": sink.value(),
            "seconds": time.perf_counter() - t0,
            "fatal": isinstance(exc, (MemoryError, RecursionError)),
        }

    try:
        _screen_source(code, allowed_imports)
    except Exception as exc:  # noqa: BLE001 - a rejected program is a normal event
        return _fail(exc)

    # Fresh globals every run: no state leaks between programs sharing this worker.
    sandbox_globals: dict[str, Any] = {
        "__builtins__": _safe_builtins(allowed_imports),
        "__name__": "__evolve_program__",
    }
    try:
        # Determinism: a worker is reused, so RNG state must be reset per run or a program that
        # forgets to seed would depend on which programs ran before it on this worker.
        random.seed(seed)
        np = sys.modules.get("numpy")
        if np is not None:
            np.random.seed(seed & 0xFFFF_FFFF)

        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(code, "<evolve-program>", "exec"), sandbox_globals)  # noqa: S102
            fn = sandbox_globals.get(_ENTRYPOINT)
            if not callable(fn):
                raise ValueError(f"program does not define a callable '{_ENTRYPOINT}()'")
            ret = fn()
        candidates = _expand_candidates(
            ret, max_expand=max_expand, max_candidates=max_candidates
        )
        payload = [_to_jsonable(c, [_MAX_NODES]) for c in candidates]
        blob = json.dumps(payload)
    except BaseException as exc:  # noqa: BLE001 - ANY failure is data, never an exception
        return _fail(exc)

    if len(blob) > max_payload_bytes:
        return _fail(ValueError(f"candidates exceed the {max_payload_bytes}-byte payload cap"))
    return {
        "ok": True,
        "candidates": payload,
        "stdout": sink.value(),
        "seconds": time.perf_counter() - t0,
        "fatal": False,
    }


def _apply_posix_limits(memory_mb: int) -> None:
    """POSIX memory + filesystem caps. No-op on Windows (the Job Object covers it there).

    Called AFTER the warm imports: ``RLIMIT_FSIZE=0`` would otherwise make CPython's .pyc writing
    fail with SIGXFSZ while importing numpy.
    """
    try:
        import resource
    except ImportError:      # Windows
        return
    nbytes = int(memory_mb) * 1024 * 1024
    for name in ("RLIMIT_AS", "RLIMIT_DATA"):
        limit = getattr(resource, name, None)
        if limit is not None:
            with contextlib.suppress(Exception):
                resource.setrlimit(limit, (nbytes, nbytes))
    for name in ("RLIMIT_FSIZE", "RLIMIT_CORE"):     # no file writes, no core dumps
        limit = getattr(resource, name, None)
        if limit is not None:
            with contextlib.suppress(Exception):
                resource.setrlimit(limit, (0, 0))


def _start_orphan_watchdog() -> None:
    """POSIX: exit if the parent dies. No-op on Windows (the Job Object's KILL_ON_JOB_CLOSE does it).

    The child is put in its own session so we can killpg its whole tree — but that also means it
    SURVIVES the parent. Normally it notices via EOF on stdin; a child stuck in a runaway program is
    not reading stdin and would spin forever as an orphan. So poll for reparenting instead.
    """
    if sys.platform == "win32" or not hasattr(os, "getppid"):
        return
    original = os.getppid()

    def _watch() -> None:
        while True:
            time.sleep(1.0)
            if os.getppid() != original:      # reparented to init => the search is gone
                os._exit(3)                   # not sys.exit(): we may be inside untrusted code

    threading.Thread(target=_watch, daemon=True).start()


def _disable_network() -> None:
    """Belt to the allowlist's braces: even a transitively-reachable socket cannot connect."""
    try:
        import socket
    except Exception:  # noqa: BLE001 - pragma: no cover
        return

    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("network access is disabled in the evolve sandbox")

    for attr in ("socket", "create_connection", "socketpair", "getaddrinfo", "create_server"):
        with contextlib.suppress(Exception):
            setattr(socket, attr, _blocked)


def _child_main(argv: list[str] | None = None) -> None:
    """Serve programs over line-delimited JSON on stdin/stdout until stdin closes."""
    argv = list(sys.argv[1:] if argv is None else argv)
    raw_cfg = argv[argv.index("--config") + 1] if "--config" in argv else "{}"
    cfg = json.loads(raw_cfg)

    allowed = frozenset(cfg.get("allowed_imports") or DEFAULT_ALLOWED_IMPORTS)
    limits = {
        "seed": int(cfg.get("seed", 0)),
        "max_expand": int(cfg.get("max_expand", 512)),
        "max_candidates": int(cfg.get("max_candidates", 4096)),
        "max_stdout_bytes": int(cfg.get("max_stdout_bytes", 4096)),
        "max_payload_bytes": int(cfg.get("max_payload_bytes", 8_000_000)),
    }

    # A runaway recursion would blow the C stack and hard-crash the child (no traceback, no error
    # string). Cap it low enough that CPython raises RecursionError we can report instead.
    sys.setrecursionlimit(int(cfg.get("recursion_limit", 3000)))

    for name in cfg.get("preimport") or ():         # pay numpy's ~1.5s import once per worker
        with contextlib.suppress(Exception):
            __import__(str(name))

    _apply_posix_limits(int(cfg.get("memory_mb", 2048)))
    _disable_network()
    _start_orphan_watchdog()

    out = sys.stdout
    out.write(_READY + "\n")
    out.flush()

    while True:
        line = sys.stdin.readline()
        if not line:                                 # parent closed stdin -> shut down
            return
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = _execute_program(
                str(req.get("code", "")),
                allowed_imports=allowed,
                **{**limits, "seed": int(req.get("seed", limits["seed"]))},
            )
        except BaseException as exc:  # noqa: BLE001 - the loop must survive anything
            resp = {"ok": False, "error": f"sandbox fault: {type(exc).__name__}: {exc}",
                    "stdout": "", "seconds": 0.0, "fatal": True}
        out.write(_PROTO_PREFIX + json.dumps(resp) + "\n")
        out.flush()


# --------------------------------------------------------------------------------------------- #
# Child bootstrap.
#
# The child is launched as a BARE SCRIPT (`python -I runner.py --evolve-sandbox-child`), so it has
# no package and cannot do the relative import below. Dispatching here -- before that import -- is
# what lets the child code live in this file (real, linted, directly unit-testable) while still
# costing only a bare interpreter to start (~460ms here vs ~840ms if it imported propab) and while
# keeping every propab module out of the untrusted process's reach.
# --------------------------------------------------------------------------------------------- #
if __name__ == "__main__" and __package__ in (None, "") and _CHILD_FLAG in sys.argv:
    _child_main()
    raise SystemExit(0)


from .program import ENTRYPOINT, ExecResult, Program, ProgramRunner  # noqa: E402

# The child hardcodes the entrypoint (it cannot import the contract). Fail loudly if it ever drifts.
assert _ENTRYPOINT == ENTRYPOINT, f"entrypoint drift: runner={_ENTRYPOINT!r} contract={ENTRYPOINT!r}"


# --------------------------------------------------------------------------------------------- #
# PARENT SIDE.
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SandboxLimits:
    """Per-run containment budget. The defaults are deliberately tight — a program that needs more
    than this is not a candidate generator, it is a bug."""

    memory_mb: int = 2048
    max_candidates: int = 4096          # hard cap on candidates emitted by one program
    max_expand: int = 512               # only expand a returned list this long into candidates
    max_payload_bytes: int = 8_000_000  # cap on the serialized candidate payload
    max_stdout_bytes: int = 4096        # cap on captured program stdout
    recursion_limit: int = 3000
    seed: int = 0                       # RNG reset before every run -> reproducible programs
    spawn_timeout_s: float = 120.0      # generous: a cold numpy import costs seconds on some hosts
    preimport: tuple[str, ...] = ("numpy",)
    allowed_imports: frozenset[str] = DEFAULT_ALLOWED_IMPORTS

    def child_config(self) -> str:
        return json.dumps({
            "memory_mb": self.memory_mb,
            "max_candidates": self.max_candidates,
            "max_expand": self.max_expand,
            "max_payload_bytes": self.max_payload_bytes,
            "max_stdout_bytes": self.max_stdout_bytes,
            "recursion_limit": self.recursion_limit,
            "seed": self.seed,
            "preimport": list(self.preimport),
            "allowed_imports": sorted(self.allowed_imports),
        })


@dataclass
class RunnerStats:
    """Observability for the runner. Cheap counters — the search lives or dies on these."""

    runs: int = 0
    ok: int = 0
    errors: int = 0
    timeouts: int = 0
    deaths: int = 0        # child killed by the OS (OOM) or crashed hard
    spawns: int = 0
    seconds: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "runs": self.runs, "ok": self.ok, "errors": self.errors, "timeouts": self.timeouts,
            "deaths": self.deaths, "spawns": self.spawns,
            "mean_seconds": (self.seconds / self.runs) if self.runs else 0.0,
        }


class _SandboxTimeout(Exception):
    """The child blew the wall-clock deadline. It is killed, never reused."""


class _SandboxDead(Exception):
    """The child exited/crashed (OOM kill, segfault, hard exit)."""


def _apply_windows_job_limits(pid: int, memory_mb: int) -> int | None:
    """Put the child in a Job Object: memory-capped, cannot spawn processes, dies if we die.

    Returns the job handle (the caller must keep it alive — closing it kills the child, which is
    exactly how we guarantee no orphaned sandboxes), or None if the platform refuses.
    """
    import ctypes
    from ctypes import wintypes as wt

    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x0000_0008
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x0000_0100
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x0000_0200
    JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x0000_0400
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x0000_2000
    PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001

    class _IOCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wt.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wt.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wt.DWORD),
            ("SchedulingClass", wt.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IOCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wt.HANDLE
    k32.OpenProcess.restype = wt.HANDLE
    k32.AssignProcessToJobObject.restype = wt.BOOL
    k32.SetInformationJobObject.restype = wt.BOOL

    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None

    info = _ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = (
        JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | JOB_OBJECT_LIMIT_JOB_MEMORY
        | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
    )
    info.BasicLimitInformation.ActiveProcessLimit = 1     # no child processes -> no fork bomb
    nbytes = int(memory_mb) * 1024 * 1024
    info.ProcessMemoryLimit = nbytes
    info.JobMemoryLimit = nbytes
    if not k32.SetInformationJobObject(
        job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
    ):
        k32.CloseHandle(job)
        return None

    handle = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, int(pid))
    if not handle:
        k32.CloseHandle(job)
        return None
    ok = k32.AssignProcessToJobObject(job, handle)
    k32.CloseHandle(handle)
    if not ok:
        k32.CloseHandle(job)
        return None
    return int(job)


class _SandboxWorker:
    """One persistent sandbox child + the parent-side deadline machinery.

    The response is read by a dedicated thread onto a queue, so the parent can wait with a deadline.
    (You cannot ``select()`` a pipe on Windows, and a blocking ``readline()`` cannot be interrupted —
    a reader thread is the portable way to own the timeout.)
    """

    def __init__(self, limits: SandboxLimits, python: str | None = None) -> None:
        self.limits = limits
        self.runs = 0
        self._job: int | None = None
        self._closed = False
        self._kill_lock = threading.Lock()

        env = dict(os.environ)
        # Without this, every numpy child opens a full BLAS thread pool and 64 workers oversubscribe
        # the box into the ground. One thread per sandbox; the parallelism is the pool.
        env.update({
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x0800_0000               # CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True                  # own process group -> killpg the tree

        self.proc = subprocess.Popen(
            # -I: isolated (no PYTHONPATH, no user site, no env overrides).
            # -B: never write .pyc. Must be a FLAG, not PYTHONDONTWRITEBYTECODE, because -I implies
            #     -E and would ignore the env var -- and a .pyc write under RLIMIT_FSIZE=0 is SIGXFSZ.
            [
                python or sys.executable, "-I", "-B", os.path.abspath(__file__),
                _CHILD_FLAG, "--config", limits.child_config(),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            cwd=tempfile.gettempdir(),      # never let a stray relative path touch the repo
            env=env,
            **kwargs,
        )

        if sys.platform == "win32":
            with contextlib.suppress(Exception):
                self._job = _apply_windows_job_limits(self.proc.pid, limits.memory_mb)

        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=25)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._await_ready()

    # -- plumbing ------------------------------------------------------------------------------ #
    def _read_stdout(self) -> None:
        try:
            for line in self.proc.stdout:                   # type: ignore[union-attr]
                if line.startswith(_PROTO_PREFIX):
                    try:
                        self._responses.put(json.loads(line[len(_PROTO_PREFIX):]))
                    except Exception:  # noqa: BLE001, S112 - a mangled frame is a dead child
                        break
                elif line.startswith(_READY):
                    self._responses.put({"_ready": True})
                # anything else is stray output; ignore it rather than desync the protocol
        except Exception:  # noqa: BLE001 - the pipe died under us; fall through to the EOF sentinel
            pass
        self._responses.put(None)                           # EOF sentinel == the child is gone

    def _read_stderr(self) -> None:
        with contextlib.suppress(Exception):
            for line in self.proc.stderr:                   # type: ignore[union-attr]
                self._stderr.append(line.rstrip())

    def _await_ready(self) -> None:
        try:
            first = self._responses.get(timeout=self.limits.spawn_timeout_s)
        except queue.Empty:
            tail = self.stderr_tail()          # read BEFORE kill(): kill() closes the stderr pipe
            self.kill()
            raise _SandboxDead(
                f"sandbox did not start within {self.limits.spawn_timeout_s:g}s: {tail}"
            ) from None
        if not (isinstance(first, dict) and first.get("_ready")):
            time.sleep(0.05)                   # let the stderr drain thread catch the traceback
            tail = self.stderr_tail()
            self.kill()
            raise _SandboxDead(f"sandbox failed to start: {tail}")

    def stderr_tail(self) -> str:
        return " | ".join(self._stderr)[-400:] or "<no stderr>"

    @property
    def alive(self) -> bool:
        return not self._closed and self.proc.poll() is None

    # -- the one operation that matters -------------------------------------------------------- #
    def request(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """Send one program, wait at most ``timeout_s`` for its result.

        Raises _SandboxTimeout / _SandboxDead. The caller retires the worker in both cases: a worker
        that blew its deadline is still running the runaway program, so it can never be reused.
        """
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")   # type: ignore[union-attr]
            self.proc.stdin.flush()                             # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - broken pipe == the child already died
            raise _SandboxDead(f"could not reach sandbox: {exc}; {self.stderr_tail()}") from exc

        try:
            resp = self._responses.get(timeout=max(0.01, float(timeout_s)))
        except queue.Empty:
            raise _SandboxTimeout(
                f"timeout: exceeded {timeout_s:g}s wall-clock budget"
            ) from None
        if resp is None:
            raise _SandboxDead(
                "sandbox process died (OOM kill or hard crash): " + self.stderr_tail()
            )
        self.runs += 1
        return resp

    def kill(self) -> None:
        """Hard-kill the child and its whole tree. Idempotent, thread-safe, never raises."""
        with self._kill_lock:
            if self._closed:
                return
            self._closed = True
        with contextlib.suppress(Exception):
            if sys.platform != "win32" and self.proc.poll() is None:
                import signal
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)   # the tree, not just the child
        with contextlib.suppress(Exception):
            self.proc.kill()
        if self._job is not None:
            # KILL_ON_JOB_CLOSE: closing the handle terminates every process in the job.
            with contextlib.suppress(Exception):
                import ctypes
                ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(self._job)
            self._job = None
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            with contextlib.suppress(Exception):
                pipe.close()                                            # type: ignore[union-attr]
        with contextlib.suppress(Exception):
            self.proc.wait(timeout=5)


class _SandboxPool:
    """A fixed-capacity pool of persistent sandbox children. Thread-safe; workers spawn lazily.

    Capacity is a semaphore, idle workers are a LIFO stack (keep hot workers hot). A worker is
    retired — killed, not returned — after a timeout, an OOM, a fatal error, or ``max_runs``; a fresh
    one is spawned on the next acquire. That is what makes the pool self-healing: no single bad
    program can permanently remove capacity.
    """

    def __init__(self, size: int, limits: SandboxLimits, max_runs: int, stats: RunnerStats) -> None:
        self.size = max(1, int(size))
        self.limits = limits
        self.max_runs = max_runs
        self.stats = stats
        self._slots = threading.Semaphore(self.size)
        self._idle: queue.LifoQueue[_SandboxWorker] = queue.LifoQueue()
        self._live: set[_SandboxWorker] = set()
        self._lock = threading.Lock()
        self._closed = False
        # Reaping a killed sandbox (TerminateProcess + wait + pipe teardown) costs real milliseconds
        # and occasionally much more under load. It must NOT sit on the search's hot path: a timeout
        # is already the most expensive thing that can happen to a worker, and making the caller pay
        # teardown on top of it turns a 1s timeout into a multi-second stall. So we hand dead workers
        # to a background reaper and free the slot immediately.
        self._graveyard: queue.Queue[_SandboxWorker | None] = queue.Queue()
        self._reaper = threading.Thread(target=self._reap_loop, name="evolve-reaper", daemon=True)
        self._reaper.start()

    def _reap_loop(self) -> None:
        while True:
            worker = self._graveyard.get()
            if worker is None:
                return
            with contextlib.suppress(Exception):
                worker.kill()

    def acquire(self) -> _SandboxWorker:
        if self._closed:
            raise _SandboxDead("sandbox pool is closed")
        self._slots.acquire()
        try:
            while True:
                try:
                    worker = self._idle.get_nowait()
                except queue.Empty:
                    break
                if worker.alive:
                    return worker
                self._forget(worker)          # died while idle; drop it and try the next
            worker = _SandboxWorker(self.limits)
            with self._lock:
                self._live.add(worker)
                self.stats.spawns += 1
            return worker
        except BaseException:
            self._slots.release()             # never leak a slot on a failed spawn
            raise

    def release(self, worker: _SandboxWorker, *, retire: bool = False) -> None:
        """Return a worker to the pool (or retire it). Always frees exactly one slot."""
        try:
            if retire or self._closed or not worker.alive or worker.runs >= self.max_runs:
                self._forget(worker)
            else:
                self._idle.put(worker)
        finally:
            self._slots.release()

    def _forget(self, worker: _SandboxWorker) -> None:
        with self._lock:
            self._live.discard(worker)
        self._graveyard.put(worker)          # killed off the hot path by the reaper

    def close(self) -> None:
        """Synchronous teardown: no orphaned sandboxes when the search ends."""
        self._closed = True
        with self._lock:
            live = list(self._live)           # idle workers are still in _live, so this covers them
            self._live.clear()
        for worker in live:
            worker.kill()
        with contextlib.suppress(queue.Empty):
            while True:
                self._idle.get_nowait()
        self._graveyard.put(None)
        self._reaper.join(timeout=10)


class SandboxProgramRunner(ProgramRunner):
    """The WS-2 ProgramRunner: pool-backed, thread-safe, and it never raises.

    One instance is shared by the whole search (``Engine.runner``) and called concurrently by every
    executor worker; ``pool_size`` is what actually bounds sandbox parallelism, so it should match
    (or exceed) ``EngineConfig.workers``. Threads that find no free sandbox block — natural
    backpressure, no unbounded process fan-out.
    """

    def __init__(
        self,
        *,
        pool_size: int = 8,
        limits: SandboxLimits | None = None,
        max_runs_per_worker: int = 256,
    ) -> None:
        self.limits = limits or SandboxLimits()
        self.stats = RunnerStats()
        # Recycle workers periodically: CPython does not return freed arenas to the OS, so a worker
        # that once served a 1.5GB program keeps that footprint. Recycling bounds long-run RSS.
        self._pool = _SandboxPool(pool_size, self.limits, max_runs_per_worker, self.stats)
        self._lock = threading.Lock()
        self._closed = False

    # -- the contract --------------------------------------------------------------------------- #
    def run(self, program: Program, *, timeout_s: float = 10.0) -> ExecResult:
        """Execute one program under the full containment stack. NEVER raises: a crash, a timeout,
        an OOM and a garbage return value are all normal events that come back as ok=False."""
        t0 = time.perf_counter()
        if self._closed:
            return ExecResult(ok=False, error="runner is closed")

        try:
            worker = self._pool.acquire()
        except BaseException as exc:  # noqa: BLE001 - even "no sandbox available" is just a bad eval
            return self._record(ExecResult(
                ok=False, error=f"sandbox unavailable: {type(exc).__name__}: {exc}",
                seconds=time.perf_counter() - t0,
            ))

        retire = False
        try:
            resp = worker.request({"code": program.code, "seed": self.limits.seed}, timeout_s)
        except _SandboxTimeout as exc:
            retire = True
            with self._lock:
                self.stats.timeouts += 1
            return self._record(ExecResult(
                ok=False, error=str(exc), seconds=time.perf_counter() - t0,
            ))
        except _SandboxDead as exc:
            retire = True
            with self._lock:
                self.stats.deaths += 1
            return self._record(ExecResult(
                ok=False, error=str(exc), seconds=time.perf_counter() - t0,
            ))
        except BaseException as exc:  # noqa: BLE001 - defensive: the search must not die here
            retire = True
            return self._record(ExecResult(
                ok=False, error=f"sandbox fault: {type(exc).__name__}: {exc}",
                seconds=time.perf_counter() - t0,
            ))
        else:
            retire = bool(resp.get("fatal"))    # MemoryError/RecursionError: process state suspect
            return self._record(ExecResult(
                ok=bool(resp.get("ok")),
                candidates=list(resp.get("candidates") or ()),
                error=resp.get("error"),
                stdout=str(resp.get("stdout") or ""),
                seconds=time.perf_counter() - t0,
            ))
        finally:
            self._pool.release(worker, retire=retire)

    def _record(self, result: ExecResult) -> ExecResult:
        with self._lock:
            self.stats.runs += 1
            self.stats.seconds += result.seconds
            if result.ok:
                self.stats.ok += 1
            else:
                self.stats.errors += 1
        return result

    # -- lifecycle ------------------------------------------------------------------------------ #
    def close(self) -> None:
        self._closed = True
        self._pool.close()

    def __enter__(self) -> SandboxProgramRunner:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort; no orphaned sandboxes
        with contextlib.suppress(Exception):
            self.close()


__all__ = [
    "DEFAULT_ALLOWED_IMPORTS",
    "RunnerStats",
    "SandboxLimits",
    "SandboxProgramRunner",
]
