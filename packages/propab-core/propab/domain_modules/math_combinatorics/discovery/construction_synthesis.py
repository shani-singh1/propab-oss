"""
Construction synthesis: a closed generate -> execute -> verify -> refine loop
over *programs* the model writes, gated by a hard exact oracle it cannot bluff.

Motivation
----------
Every finder in this package (ILS, DLS, exact CP-SAT, branch-and-bound) tunes a
*fixed* search routine; the model never writes the construction itself. Both exact
CP-SAT and ILS proved insufficient to cross the a(7) wall. Construction synthesis
moves the creative act to the model: it WRITES a ``construct(n)`` function, we
EXECUTE that code in a locked-down sandbox, the emitted object is checked by the
independent exact VERIFIER (``is_B3`` for B_3), and the concrete STRUCTURAL FAILURE
(which threefold sum collided, which vector left the cube) is fed back to refine
the next program. The model cannot fake a witness: the oracle re-derives every
constraint from scratch.

What this module is (and is not)
--------------------------------
This builds the LOOP and its safety/verification guarantees. Whether a real model
in a real run CROSSES a published bound is an empirical question this code does not
answer and never asserts. The sole record gate remains ``certify_b3_record``; the
loop returns a verified witness + trace and lets that certifier -- run elsewhere --
decide whether anything is a record.

Sandbox safety model (defense in depth, NOT an adversarial security boundary)
-----------------------------------------------------------------------------
Model-generated code is untrusted-by-default and executed under four layers:

1. **Static AST screen** (``_screen_source``) rejects, before any execution:
   ``import``/``from``-imports, dunder attribute access (``x.__class__`` -- the
   classic ``().__class__.__bases__`` escape), and dunder *names*. Syntax errors
   are rejected here too.
2. **Restricted globals** (``_SAFE_BUILTINS``): only a hand-picked allowlist of
   pure builtins is exposed. ``__import__``, ``open``, ``eval``, ``exec``,
   ``compile``, ``globals``, ``getattr``/``setattr``, ``input`` etc. are simply
   absent -> ``NameError``. A few pure helper modules (``math``, ``itertools``,
   ``random``) are pre-bound as names so constructions can use them without an
   ``import`` statement.
3. **Separate process** (``sandbox_exec`` launched as a bare, dependency-free
   child interpreter): the code runs out-of-process, so an OOM/crash cannot take
   down the caller, and the trusted verifier runs in the PARENT on the returned
   data -- never inside the sandbox. The child imports no ``propab`` code, so it
   starts in ~100ms and the budget measures the construction, not import warmup.
4. **Hard wall-clock timeout**: ``subprocess.run(..., timeout=exec_budget_s)`` kills
   the child if it overruns. An infinite loop or runaway construction is killed, not
   merely abandoned. Output size is capped (``_MAX_POINTS``) before it is returned,
   bounding memory/serialization cost.

Documented limits: this is a robust guard against *accidental* damage and common
escapes from non-adversarial, model-written code. It is NOT a hardened jail against
a determined adversary crafting native-code exploits -- the child runs with the
same OS privileges as the parent. For untrusted third-party code you would add
OS-level isolation (containers/seccomp/rlimits). Treat the model as a careless
author, not a malicious one.
"""
from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
from typing import Any, Callable, Optional, Protocol, Sequence

from propab.domain_modules.math_combinatorics.discovery.verifier import is_B3
from propab.domain_modules.math_combinatorics.discovery import sandbox_exec
# Re-exported so callers/tests share the child's exact static screen and limits.
from propab.domain_modules.math_combinatorics.discovery.sandbox_exec import (  # noqa: F401
    _MAX_POINTS,
    _SAFE_BUILTINS,
    _execute_construct,
    _screen_source,
)

Vector = tuple[int, ...]


# ---------------------------------------------------------------------------
# Sandbox: run model-written construct() in an isolated child with a hard timeout.
# ---------------------------------------------------------------------------
def _run_in_sandbox(
    code: str, n: int, budget_s: float
) -> tuple[bool, Optional[list[Vector]], Optional[str]]:
    """Execute ``code`` in a dependency-free child interpreter with a hard timeout.

    Launches ``sandbox_exec`` as a bare script (no propab imports -> fast startup),
    passing ``{code, n}`` on stdin and reading the result on stdout. On timeout the
    child is killed by ``subprocess.run`` and ``ok`` is False. The trusted verifier
    runs in the PARENT on the returned data, never inside the sandbox. Returns
    ``(ok, points, error)`` and never raises for a sandbox fault.
    """
    payload = json.dumps({"code": code, "n": int(n)})
    try:
        proc = subprocess.run(
            [sys.executable, sandbox_exec.__file__],
            input=payload,
            capture_output=True,
            text=True,
            timeout=max(0.05, float(budget_s)),
        )
    except subprocess.TimeoutExpired:
        return (False, None, f"timeout: exceeded {budget_s:g}s wall-clock budget")
    except Exception as exc:  # noqa: BLE001 - launching the sandbox must never crash the loop
        return (False, None, f"sandbox launch failed: {type(exc).__name__}: {exc}")

    out = (proc.stdout or "").strip()
    if not out:
        err = (proc.stderr or "").strip() or f"sandbox exited with code {proc.returncode}"
        return (False, None, err[:500])
    try:
        res = json.loads(out)
    except json.JSONDecodeError:
        return (False, None, f"sandbox produced unparseable output: {out[:200]}")
    if res.get("status") == "ok":
        pts = [tuple(int(x) for x in v) for v in res.get("points", [])]
        return (True, pts, None)
    return (False, None, str(res.get("error", "unknown sandbox error")))


# ---------------------------------------------------------------------------
# Verification result + spec (the loop is domain-general; the spec carries the
# domain-specific oracle and prompt).
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class VerificationResult:
    """Outcome of running the exact oracle on an emitted object."""

    valid: bool
    size: int
    failure: Optional[str] = None  # concrete STRUCTURAL failure when not valid


class _LLMLike(Protocol):
    async def call(self, *, prompt: str, purpose: str, session_id: str, **kw: Any) -> str: ...


@dataclasses.dataclass
class ConstructionSpec:
    """Domain payload for the generic synthesis loop.

    ``verify`` is the exact oracle: it takes the emitted points and returns a
    ``VerificationResult`` (valid + size, or a concrete structural failure). The
    loop trusts nothing else. ``build_prompt`` renders the next instruction from
    the best-so-far and the last failure.
    """

    name: str
    n: int
    verify: Callable[[Sequence[Sequence[int]]], VerificationResult]
    object_description: str = "a large combinatorial object"
    published_best: Optional[int] = None
    extra_guidance: str = ""

    def build_prompt(
        self,
        best: Optional[dict[str, Any]],
        last_failure: Optional[str],
        *,
        iteration: int,
    ) -> str:
        lines = [
            "You are writing an executable Python construction for a hard "
            "combinatorial optimisation problem.",
            "",
            f"TARGET: {self.object_description} for n = {self.n}.",
        ]
        if self.published_best is not None:
            lines.append(f"Published best-known size to beat: {self.published_best}.")
        best_size = best.get("size") if best else 0
        lines += [
            f"Current best VERIFIED size so far: {best_size}. Emit a STRICTLY LARGER "
            "valid object.",
            "",
            "Write a single pure function with this exact signature:",
            "    def construct(n):",
            "        # return a list of tuples, each an n-length integer vector",
            "        ...",
            "",
            "SANDBOX RULES (violating any of these fails your program):",
            " - No import statements. `math`, `itertools`, `random` are already "
            "available as names.",
            " - No file, OS, network, or dunder (`__...__`) access.",
            " - Must be deterministic-friendly and finish within the time budget.",
            " - Return value must be a list of n-length integer tuples.",
        ]
        if self.extra_guidance:
            lines += ["", self.extra_guidance]
        if last_failure:
            lines += [
                "",
                "The PREVIOUS attempt failed. Concrete structural failure to fix:",
                f"    {last_failure}",
                "Diagnose the cause and change the construction accordingly.",
            ]
        lines += [
            "",
            "Respond with ONLY a ```python fenced code block containing construct(n).",
        ]
        return "\n".join(lines)


_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str) -> Optional[str]:
    """Pull the construct() source out of an LLM response (fenced or bare)."""
    if not text:
        return None
    blocks = _CODE_FENCE.findall(text)
    for block in blocks:
        if "def construct" in block:
            return block.strip()
    if blocks:
        return blocks[0].strip()
    if "def construct" in text:
        return text.strip()
    return None


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------
async def synthesize_construction(
    spec: ConstructionSpec,
    *,
    llm: _LLMLike,
    max_iters: int = 6,
    exec_budget_s: float = 5.0,
    session_id: str = "construction-synthesis",
    best_so_far: Optional[Sequence[Sequence[int]]] = None,
    sandbox_runner: Optional[Callable[[str, int, float], tuple[bool, Optional[list[Vector]], Optional[str]]]] = None,
) -> dict[str, Any]:
    """Closed generate -> execute -> verify -> refine loop over model-written code.

    The model writes ``construct(n)``; each program is executed in the locked-down
    sandbox (see module docstring), the emitted object is checked by ``spec.verify``
    (the exact oracle), and the structural failure feeds the next prompt. Tracks the
    best VERIFIED witness across up to ``max_iters`` iterations.

    Returns ``{"best": <witness dict or None>, "trace": [...], "spec": name,
    "iterations": k}``. The loop NEVER asserts a record -- ``certify_b3_record``
    (run by the caller) is the sole record gate.
    """
    runner = sandbox_runner or _run_in_sandbox
    trace: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    last_failure: Optional[str] = None

    # Seed the incumbent with a provided warm-start witness, if it actually verifies.
    if best_so_far:
        seed_vr = spec.verify(best_so_far)
        if seed_vr.valid and seed_vr.size > 0:
            best = {"n": spec.n, "size": seed_vr.size, "set": [list(v) for v in best_so_far]}

    for it in range(max_iters):
        prompt = spec.build_prompt(best, last_failure, iteration=it)
        step: dict[str, Any] = {"iteration": it}
        try:
            raw = await llm.call(
                prompt=prompt, purpose="construction_synthesis", session_id=session_id
            )
        except Exception as exc:  # noqa: BLE001 - an LLM fault must not kill the loop
            step.update(stage="llm", ok=False, failure=f"llm_error: {type(exc).__name__}: {exc}")
            last_failure = step["failure"]
            trace.append(step)
            continue

        code = _extract_code(raw or "")
        if not code:
            step.update(stage="extract", ok=False, failure="no construct() code found in response")
            last_failure = step["failure"]
            trace.append(step)
            continue
        step["code"] = code

        ok, points, err = runner(code, spec.n, exec_budget_s)
        if not ok:
            step.update(stage="execute", ok=False, failure=err or "sandbox failure")
            last_failure = step["failure"]
            trace.append(step)
            continue

        vr = spec.verify(points or [])
        step["size"] = vr.size
        if vr.valid:
            step.update(stage="verify", ok=True, failure=None)
            if best is None or vr.size > best["size"]:
                best = {"n": spec.n, "size": vr.size, "set": [list(v) for v in (points or [])]}
                step["new_best"] = True
            # Even a valid-but-not-larger result is not a "failure"; nudge for more.
            last_failure = (
                None
                if step.get("new_best")
                else f"valid but size {vr.size} did not beat best {best['size']}; go larger"
            )
        else:
            step.update(stage="verify", ok=False, failure=vr.failure or "failed verification")
            last_failure = step["failure"]
        trace.append(step)

    return {
        "best": best,
        "trace": trace,
        "spec": spec.name,
        "iterations": len(trace),
    }


# ---------------------------------------------------------------------------
# Concrete B_3 spec: exact oracle (is_B3) + structural failure extraction.
# ---------------------------------------------------------------------------
def _b3_structural_failure(S: list[Vector], n: int) -> Optional[str]:
    """Return the first concrete reason S is not a valid B_3 set in {0,1}^n, or None.

    Localises the failure the way the model needs to fix it: a mis-shaped vector, a
    non-binary entry, a duplicate, or the *specific* pair of unordered triples whose
    threefold sums collide.
    """
    if not S:
        return None
    for v in S:
        if len(v) != n:
            return f"vector {v} has length {len(v)}, expected n={n}"
        for x in v:
            if x not in (0, 1):
                return f"vector {v} has non-binary entry {x} (entries must be 0 or 1)"
    seen_vecs: set[Vector] = set()
    for v in S:
        if v in seen_vecs:
            return f"duplicate vector {v} (all vectors must be distinct)"
        seen_vecs.add(v)
    # Find the first colliding pair of threefold multiset sums.
    sums: dict[Vector, tuple[Vector, Vector, Vector]] = {}
    m = len(S)
    for i in range(m):
        a = S[i]
        for j in range(i, m):
            b = S[j]
            ab = tuple(a[t] + b[t] for t in range(n))
            for k in range(j, m):
                c = S[k]
                key = tuple(ab[t] + c[t] for t in range(n))
                prev = sums.get(key)
                if prev is not None:
                    return (
                        f"threefold-sum collision: triples {prev} and "
                        f"{(a, b, c)} both sum to {key} -- these two multisets must "
                        "differ but produce the same sum"
                    )
                sums[key] = (a, b, c)
    return None


def _b3_verify_factory(n: int) -> Callable[[Sequence[Sequence[int]]], VerificationResult]:
    def verify(obj: Sequence[Sequence[int]]) -> VerificationResult:
        S = [tuple(int(x) for x in v) for v in obj]
        failure = _b3_structural_failure(S, n)
        if failure is None:
            # Independent paranoid re-check (never trust the structural pass alone).
            if is_B3(S):
                return VerificationResult(valid=True, size=len(S), failure=None)
            return VerificationResult(valid=False, size=len(S), failure="failed is_B3 exact check")
        return VerificationResult(valid=False, size=len(S), failure=failure)

    return verify


def b3_construction_spec(n: int, *, published_best: Optional[int] = None) -> ConstructionSpec:
    """Build a ConstructionSpec for B_3 sets in the binary cube {0,1}^n (A396704)."""
    return ConstructionSpec(
        name=f"b3_binary_cube_n{n}",
        n=n,
        verify=_b3_verify_factory(n),
        object_description=(
            "a B_3 set in the binary cube {0,1}^n: a set of 0/1 vectors of length n "
            "such that all threefold multiset sums a+b+c (over the integers) are "
            "distinct, i.e. a+b+c=d+e+f forces the multisets {a,b,c}={d,e,f}"
        ),
        published_best=published_best,
        extra_guidance=(
            "Hint: linear/algebraic structure (e.g. rows of a generator matrix over "
            "GF(2), Sidon-style difference constructions, or B_h designs) tends to "
            "avoid threefold-sum collisions far better than random 0/1 vectors."
        ),
    )
