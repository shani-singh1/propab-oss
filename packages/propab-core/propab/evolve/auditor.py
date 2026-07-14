"""Evolve — the Adversarial Auditor (WS-6).

The exact verifier checks the MATH. Nothing checked the CLAIM. This is that layer.

This project has twice reported a positive result that dissolved under scrutiny. Both times the
math was fine and the *comparison* was garbage: once the delta was pure LLM-judge noise, once the
"baseline" was a model that had been asked about a CSV it was never given. Neither failure is
detectable by `verify()` — a verifier that says `valid=True, score=7` is telling the truth about the
object and nothing at all about whether 7 is a discovery.

So: enumerate every way to fool yourself, then actively try to fool yourself, and accept only what
survives. `audit()` runs a fixed KILL-LIST and a Problem's own extra checks. A discovery is recorded
only if it passes.

DEFAULT TO REJECT. If the auditor cannot *positively confirm* something, that is a kill, not a pass.
Absence of evidence is a kill here — that is the entire point of the layer. In particular, a target
that does not supply provenance for its baseline, an independent class-membership check, and
positive/negative verifier controls CANNOT produce a publishable result through this auditor. The
burden of proof is on the target, not on the auditor.

Domain-general by construction (see the standing domain-independence rule): this file knows only the
`Problem` contract. Every domain-specific kill (e.g. `coding_theory.trivial_rediscovery`) is injected
by the target through the optional hooks in `AuditableProblem` below.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import pickle
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .problem import Problem, Verdict
from .program import Program

AUDITOR_VERSION = "ws6.1"

KILL = "kill"
WARN = "warn"

# How thin a float margin has to be before we call it noise rather than a discovery.
FLOAT_ATOL = 1e-9
FLOAT_RTOL = 1e-9
# A margin thinner than this (relative) is not yet a kill, but it is not a headline either.
THIN_MARGIN_RTOL = 1e-6

# Baselines rot. A record sourced years ago may have been beaten since.
DEFAULT_MAX_BASELINE_AGE_S = 365 * 24 * 3600.0

# Provenance kinds that are NOT evidence of anything. These are the ways we lied to ourselves before.
FORBIDDEN_PROVENANCE_KINDS = {
    "",
    "model_memory",
    "model",
    "llm",
    "memory",
    "guess",
    "hardcoded",
    "hardcode",
    "literal",
    "unknown",
    "assumed",
    "tbd",
}

# Witness keys that mean "this number came out of a table, not a computation".
_TABLE_SOURCE_VALUES = {
    "best_known_table",
    "table",
    "table_lookup",
    "tabulation",
    "lookup",
    "known_construction",
    "literature",
    "textbook",
}
_TABLE_METHOD_VALUES = {
    "table_lookup",
    "tabulation",
    "best_known_table",
    "lookup",
    "recall",
    "memory",
}

# Witness keys that reveal the score depends on a slop parameter someone chose.
_TOLERANCE_KEYS = ("tol", "tolerance", "atol", "rtol", "eps", "epsilon", "threshold", "cutoff")

# Candidate fields that smell like a program grading its own homework.
_SELF_REPORT_RE = re.compile(
    r"^(score|fitness|objective|value|quality|min_?distance|distance|d|weight|"
    r"best|result|improvement|margin|rank|size|count)$",
    re.IGNORECASE,
)
_SELF_REPORT_SENTINEL = 987654.321

_UNSEEDED_RNG_RE = re.compile(r"(?<!\.)\b(random\.|np\.random\.|numpy\.random\.)", re.IGNORECASE)
_SEEDED_RE = re.compile(r"\bseed\s*\(|\bSeed\b|default_rng\s*\(\s*\d|\bRandomState\s*\(\s*\d")


# --------------------------------------------------------------------------- #
# Report types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CheckResult:
    """One named check. `passed=False` + `severity=KILL` sinks the discovery."""

    name: str
    passed: bool
    severity: str = KILL
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_kill(self) -> bool:
        return not self.passed and self.severity == KILL


@dataclass
class AuditReport:
    """The audit that cleared (or killed) a claimed discovery. Serializable; attaches to a Record."""

    problem: str
    passed: bool
    kills: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    verifier_fingerprint: str = ""
    auditor_version: str = AUDITOR_VERSION
    audited_at: float = field(default_factory=time.time)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def summary(self) -> str:
        if self.passed:
            n = len(self.warnings)
            tail = f" ({n} warning{'s' if n != 1 else ''})" if n else ""
            return f"PASS — {len(self.checks)} checks{tail}"
        lines = [f"KILL — {len(self.kills)} fatal finding(s):"]
        lines += [f"  [{i + 1}] {k}" for i, k in enumerate(self.kills)]
        return "\n".join(lines)

    def attach(self, record: Any) -> Any:
        """Attach this audit to a `Record` so every published result carries the audit that cleared it.

        `Record` (frozen contract, ledger.py) has no `audit` field, and `Record.to_json()` serializes
        only declared dataclass fields — so `notes` is the only channel that actually survives export.
        We write the audit there as structured JSON and *also* set `.audit_report` for in-memory
        consumers. WS-0 should promote `audit: dict` to a first-class Record field; until then this is
        the honest workaround, not a nice-to-have.
        """
        setattr(record, "audit_report", self)
        blob = json.dumps({"audit": self.to_dict()}, default=str)
        existing = (getattr(record, "notes", "") or "").strip()
        record.notes = f"{existing}\n{blob}".strip() if existing else blob
        return record


class AuditError(RuntimeError):
    """Raised only by `audit_or_raise`."""


# --------------------------------------------------------------------------- #
# Optional per-target hooks. A Problem opts in to being auditable by implementing these.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BestKnownSource:
    """Where `best_known()` actually came from. A float with no source is not a baseline."""

    value: float
    source: str            # "Grassl, Bounds on the minimum distance of linear codes, [24,12] cell"
    citation: str          # URL / DOI / table id — something a third party can open
    retrieved_at: float    # unix ts: when a human/pipeline last confirmed it against the source
    kind: str = "table"    # table | registry | paper | proof | computed — never model_memory

    def problems(self) -> list[str]:
        """Everything wrong with this attribution. Empty list == a real, citable baseline."""
        bad: list[str] = []
        kind = (self.kind or "").strip().lower()
        if kind in FORBIDDEN_PROVENANCE_KINDS:
            bad.append(
                f"best_known provenance kind={self.kind!r} is not a real source "
                "(a model's memory / a hardcoded literal is not a baseline)"
            )
        if not (self.source or "").strip():
            bad.append("best_known provenance has an empty `source` — nothing to attribute it to")
        if not (self.citation or "").strip():
            bad.append(
                "best_known provenance has an empty `citation` — a third party cannot open it"
            )
        if not isinstance(self.value, (int, float)) or not math.isfinite(float(self.value)):
            bad.append(f"best_known provenance value={self.value!r} is not a finite number")
        if not isinstance(self.retrieved_at, (int, float)) or self.retrieved_at <= 0:
            bad.append("best_known provenance has no `retrieved_at` — cannot tell if it is stale")
        return bad


@dataclass(frozen=True)
class Control:
    """A known-good or known-bad input. If the verifier can't score these, nothing it says counts."""

    name: str
    candidate: Any
    expect_valid: bool
    expect_score: float | None = None   # None == don't care about the exact number
    tol: float = 1e-9


@runtime_checkable
class AuditableProblem(Protocol):
    """Everything a `Problem` MAY expose to be auditable. All optional; absence of the load-bearing
    ones (provenance, independent_check, controls) is itself a kill — see the module docstring."""

    def best_known_provenance(self) -> BestKnownSource: ...
    def exact_best_known(self) -> Fraction | int: ...
    def independent_check(self, candidate: Any) -> tuple[bool, str]: ...
    def controls(self) -> Sequence[Control]: ...
    def exact_score(self, candidate: Any) -> Fraction | int | None: ...
    def is_degenerate(self, candidate: Any) -> str | None: ...
    def trivial_rediscovery(self, candidate: Any, verdict: Verdict) -> str | None: ...
    def check_witness(self, candidate: Any, witness: dict[str, Any]) -> tuple[bool, str]: ...
    def to_jsonable(self, candidate: Any) -> Any: ...
    def from_jsonable(self, obj: Any) -> Any: ...
    def audit_spec(self) -> dict[str, Any]: ...
    def audit_checks(self) -> Sequence[Callable[[AuditContext], CheckResult]]: ...


@dataclass
class AuditContext:
    """Handed to per-target extra checks."""

    problem: Problem
    verdict: Verdict
    candidate: Any
    program: Program | None
    best_known: float
    witness: dict[str, Any]


# --------------------------------------------------------------------------- #
# Verifier identity
# --------------------------------------------------------------------------- #
def verifier_fingerprint(problem: Problem) -> str:
    """Hash of the code that will check the claim. Pins WHAT verified it, so a later reader can tell
    whether the verifier has drifted since the record was written.

    CAVEAT, stated plainly because it matters: this hashes the Problem class and its `verify` method.
    Real targets are THIN ADAPTERS — `verify()` delegates to a domain module (coding_theory,
    graph_invariants, …) — and that helper's source is NOT covered here. The actual checking logic
    can therefore change while the fingerprint stays put. A target must declare its real verifier via
    `verifier_sources()` (module names or file paths) to be honestly pinned.
    """
    parts: list[str] = []
    cls = type(problem)
    for obj in (getattr(cls, "verify", None), cls):
        try:
            parts.append(inspect.getsource(obj))  # type: ignore[arg-type]
        except (OSError, TypeError):
            continue
    if not parts:
        parts.append(f"{cls.__module__}.{cls.__qualname__}")

    for source in _declared_verifier_sources(problem):
        parts.append(source)

    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _declared_verifier_sources(problem: Problem) -> list[str]:
    """Source of every module the target declares its verifier actually depends on."""
    hook = getattr(problem, "verifier_sources", None)
    if hook is None:
        return []
    out: list[str] = []
    try:
        declared = list(hook())
    except Exception:  # noqa: BLE001
        return []
    for item in declared:
        try:
            if isinstance(item, str) and not item.endswith(".py"):
                module = importlib.import_module(item)
                out.append(inspect.getsource(module))
            else:
                out.append(Path(str(item)).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — an unreadable declared source is caught by the repro check
            out.append(f"<unreadable:{item}>")
    return out


# --------------------------------------------------------------------------- #
# The child process: re-check the claim from the record alone, in a clean interpreter.
# --------------------------------------------------------------------------- #
_MARKER = "@@AUDIT@@"

_CHILD_SOURCE = r'''
import importlib, io, json, pickle, socket, sys

MARKER = "@@AUDIT@@"

def emit(obj):
    sys.stdout.write(MARKER + json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()

try:
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    sys.path[:0] = payload["sys_path"]

    # verify() MUST be cheap, exact and offline (problem.py contract). Enforce it, don't assume it:
    # an LLM-in-the-verifier is precisely how a "delta" turned out to be judge noise last time.
    class NetworkBlocked(RuntimeError):
        pass

    def _blocked(*a, **k):
        raise NetworkBlocked("verify() attempted network access during audit")

    socket.socket = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked

    # Rebuild the PROBLEM (the verifier) — never the program, never the search state.
    spec = payload.get("problem_spec")
    if spec:
        mod = importlib.import_module(spec["module"])
        factory = getattr(mod, spec["factory"])
        problem = factory(**spec.get("kwargs", {}))
    else:
        with open(payload["problem_pkl"], "rb") as fh:
            problem = pickle.load(fh)

    # Rebuild the CANDIDATE from its serialized form only. If the record's JSON cannot be
    # re-hydrated into something the verifier accepts, a third party cannot re-check us either.
    candidate = json.loads(payload["candidate_json"])
    if hasattr(problem, "from_jsonable"):
        candidate = problem.from_jsonable(candidate)

    buf = io.StringIO()
    real_stdout, sys.stdout = sys.stdout, buf
    try:
        verdict = problem.verify(candidate)
    finally:
        sys.stdout = real_stdout

    out = {
        "ok": True,
        "valid": bool(verdict.valid),
        "score": float(verdict.score),
        "detail": json.loads(json.dumps(dict(verdict.detail or {}), default=str)),
        "fingerprint": payload["expect_fingerprint"],
    }

    # Recompute the verifier fingerprint in the child: if the verifier that checked the claim is not
    # the verifier sitting on disk now, the record is unreproducible whatever the numbers say.
    try:
        import hashlib, inspect
        parts = []
        cls = type(problem)
        for obj in (getattr(cls, "verify", None), cls):
            try:
                parts.append(inspect.getsource(obj))
            except Exception:
                pass
        if not parts:
            parts.append(cls.__module__ + "." + cls.__qualname__)
        # Must mirror the parent's _declared_verifier_sources exactly, or every target that declares
        # its real verifier would trip a bogus "fingerprint drifted" kill.
        hook = getattr(problem, "verifier_sources", None)
        if hook is not None:
            for item in list(hook()):
                try:
                    if isinstance(item, str) and not item.endswith(".py"):
                        parts.append(inspect.getsource(importlib.import_module(item)))
                    else:
                        with open(str(item), "r", encoding="utf-8") as sf:
                            parts.append(sf.read())
                except Exception:
                    parts.append("<unreadable:%s>" % (item,))
        out["fingerprint"] = hashlib.sha256(
            "\n".join(parts).encode("utf-8")
        ).hexdigest()[:16]
    except Exception:
        pass

    # The recorded witness must itself be checkable, not just decorative.
    if hasattr(problem, "check_witness"):
        witness = json.loads(payload["witness_json"])
        try:
            ok, why = problem.check_witness(candidate, witness)
            out["witness_ok"] = bool(ok)
            out["witness_reason"] = str(why)
        except Exception as exc:
            out["witness_ok"] = False
            out["witness_reason"] = "check_witness raised: %r" % (exc,)

    emit(out)
except BaseException as exc:  # noqa: BLE001 — a crash in here is a KILL, not a traceback
    import traceback

    emit({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
          "traceback": traceback.format_exc()[-2000:]})
'''


@dataclass(frozen=True)
class ReproResult:
    ok: bool
    valid: bool = False
    score: float = float("-inf")
    detail: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    witness_ok: bool | None = None
    witness_reason: str = ""
    error: str = ""
    reconstruction: str = ""          # "audit_spec" (rebuilt from scratch) | "pickle" (state carried)
    carried_state: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The Auditor
# --------------------------------------------------------------------------- #
class Auditor:
    """Try to kill the discovery. Everything that survives is what we are willing to publish."""

    def __init__(
        self,
        *,
        repro_timeout_s: float = 120.0,
        max_baseline_age_s: float = DEFAULT_MAX_BASELINE_AGE_S,
        determinism_reps: int = 3,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.repro_timeout_s = repro_timeout_s
        self.max_baseline_age_s = max_baseline_age_s
        self.determinism_reps = determinism_reps
        self._now = now

    # -- entry point -------------------------------------------------------- #
    def audit(
        self,
        problem: Problem,
        verdict: Verdict,
        candidate: Any,
        program: Program | None = None,
    ) -> AuditReport:
        """Run the full kill-list. Passing means: we tried to break this and could not."""
        started = self._now()
        report = AuditReport(
            problem=getattr(problem, "name", type(problem).__name__),
            passed=False,
            verifier_fingerprint=_safe(lambda: verifier_fingerprint(problem), ""),
        )

        checks: list[CheckResult] = []
        best_known = _safe(problem.best_known, None)

        # Ordering matters: a broken verifier (5) invalidates every other check's evidence, so run it
        # first and short-circuit — no output from a broken engine can be trusted, including ours.
        checks.extend(self._check_verifier_sanity(problem))
        if any(c.is_kill for c in checks):
            checks.append(
                CheckResult(
                    "audit_short_circuit",
                    False,
                    KILL,
                    "verifier failed its own controls — every downstream check is meaningless, "
                    "so nothing from this engine run can be trusted",
                )
            )
            return self._finish(report, checks, started)

        checks.append(self._check_claim_coherence(problem, verdict, best_known))
        checks.append(self._check_witness_present(verdict))
        checks.append(self._check_candidate_serializable(problem, candidate))
        checks.extend(self._check_score_provenance(problem, verdict, candidate))
        checks.append(self._check_verifier_determinism(problem, candidate, verdict))
        checks.append(self._check_degenerate(problem, candidate))
        checks.append(self._check_class_membership(problem, candidate))
        checks.extend(self._check_baseline_realness(problem, best_known))
        checks.append(self._check_trivial_rediscovery(problem, candidate, verdict))
        checks.extend(self._check_numerical_artifact(problem, verdict, candidate, best_known))
        checks.extend(self._check_reproduce_from_witness(problem, verdict, candidate, report))
        checks.append(self._check_program_hygiene(program))
        checks.extend(self._run_target_checks(problem, verdict, candidate, program, best_known))

        return self._finish(report, checks, started)

    def audit_or_raise(
        self,
        problem: Problem,
        verdict: Verdict,
        candidate: Any,
        program: Program | None = None,
    ) -> AuditReport:
        report = self.audit(problem, verdict, candidate, program)
        if not report.passed:
            raise AuditError(report.summary())
        return report

    def _finish(self, report: AuditReport, checks: list[CheckResult], started: float) -> AuditReport:
        report.checks = checks
        report.kills = [f"{c.name}: {c.reason}" for c in checks if c.is_kill]
        report.warnings = [
            f"{c.name}: {c.reason}" for c in checks if not c.passed and c.severity == WARN
        ]
        report.passed = not report.kills
        report.seconds = round(self._now() - started, 4)
        return report

    # ------------------------------------------------------------------ #
    # KILL 5 — Verifier sanity (positive/negative controls). Runs FIRST.
    # ------------------------------------------------------------------ #
    def _check_verifier_sanity(self, problem: Problem) -> list[CheckResult]:
        name = "verifier_sanity"
        controls_fn = getattr(problem, "controls", None)
        if controls_fn is None:
            return [
                CheckResult(
                    name,
                    False,
                    KILL,
                    f"target {getattr(problem, 'name', '?')!r} exposes no controls() — the verifier "
                    "has never been shown to score a KNOWN result correctly, so we cannot "
                    "distinguish a discovery from a verifier bug. Supply >=1 known-good and "
                    ">=1 known-bad candidate.",
                )
            ]
        try:
            controls = list(controls_fn())
        except Exception as exc:  # noqa: BLE001
            return [CheckResult(name, False, KILL, f"controls() raised: {exc!r}")]

        results: list[CheckResult] = []
        pos = [c for c in controls if c.expect_valid]
        neg = [c for c in controls if not c.expect_valid]
        if not pos or not neg:
            results.append(
                CheckResult(
                    name,
                    False,
                    KILL,
                    f"controls() supplies {len(pos)} positive and {len(neg)} negative control(s); "
                    "both are required — a verifier that has only ever seen good inputs has not "
                    "been shown to REJECT anything",
                )
            )

        for ctl in controls:
            try:
                v = problem.verify(ctl.candidate)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    CheckResult(
                        f"{name}[{ctl.name}]",
                        False,
                        KILL,
                        f"verify() RAISED on control {ctl.name!r} ({exc!r}) — the contract says it "
                        "must return Verdict(valid=False) on any input, never raise",
                    )
                )
                continue
            if bool(v.valid) != bool(ctl.expect_valid):
                results.append(
                    CheckResult(
                        f"{name}[{ctl.name}]",
                        False,
                        KILL,
                        f"verifier scored control {ctl.name!r} as valid={v.valid}, expected "
                        f"valid={ctl.expect_valid}. THE ENGINE IS BROKEN — no output from this run "
                        "can be trusted, including the claimed discovery",
                        {"score": _jsonable_number(v.score)},
                    )
                )
                continue
            if ctl.expect_score is not None and not _close(v.score, ctl.expect_score, ctl.tol):
                results.append(
                    CheckResult(
                        f"{name}[{ctl.name}]",
                        False,
                        KILL,
                        f"verifier scored known result {ctl.name!r} as {v.score!r}, expected "
                        f"{ctl.expect_score!r} (tol={ctl.tol}). THE ENGINE IS BROKEN — a verifier "
                        "that cannot reproduce a KNOWN value cannot be trusted on an unknown one, "
                        "so nothing from this run is evidence of anything",
                    )
                )

        # Controls we synthesize ourselves: a mutated program emits garbage constantly, and the
        # contract says verify() must survive it. If it raises, the engine's "invalid" signal is
        # actually "crashed", and a crash is not evidence of anything.
        for junk_name, junk in (
            ("none", None),
            ("empty_list", []),
            ("empty_dict", {}),
            ("empty_str", ""),
            ("garbage_str", "not a candidate"),
            ("nested_junk", [[None, "x"], {"a": 1}]),
        ):
            try:
                problem.verify(junk)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    CheckResult(
                        f"{name}[garbage:{junk_name}]",
                        False,
                        KILL,
                        f"verify() RAISED on garbage input {junk_name!r} ({type(exc).__name__}: "
                        f"{exc}) — contract requires Verdict(valid=False, score=-inf) instead",
                    )
                )

        if not results:
            results.append(
                CheckResult(
                    name,
                    True,
                    KILL,
                    "",
                    {
                        "controls": len(controls),
                        "positive": len(pos),
                        "negative": len(neg),
                    },
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Claim coherence — is this even a claimed improvement?
    # ------------------------------------------------------------------ #
    def _check_claim_coherence(
        self, problem: Problem, verdict: Verdict, best_known: float | None
    ) -> CheckResult:
        name = "claim_coherence"
        if not verdict.valid:
            return CheckResult(name, False, KILL, "verdict.valid is False — there is no claim here")
        score = _as_float(verdict.score)
        if score is None or not math.isfinite(score):
            return CheckResult(
                name, False, KILL, f"verdict.score={verdict.score!r} is not a finite number"
            )
        if best_known is None:
            return CheckResult(
                name, False, KILL, "best_known() raised or returned None — nothing to compare against"
            )
        bk = _as_float(best_known)
        if bk is None or not math.isfinite(bk):
            return CheckResult(
                name, False, KILL, f"best_known()={best_known!r} is not a finite number"
            )
        try:
            claims_improvement = bool(problem.is_improvement(verdict))
        except Exception as exc:  # noqa: BLE001
            return CheckResult(name, False, KILL, f"is_improvement() raised: {exc!r}")
        if not claims_improvement:
            return CheckResult(
                name,
                False,
                KILL,
                "the target's own is_improvement() says this is NOT an improvement — "
                "the auditor will not overrule the target in the optimistic direction",
            )
        if score <= bk:
            return CheckResult(
                name,
                False,
                KILL,
                f"is_improvement() returned True but score={score!r} does not beat "
                f"best_known()={bk!r}. The target's improvement rule is incoherent — this is exactly "
                "the shape of the bug that manufactured our last two 'wins'",
            )
        return CheckResult(name, True, KILL, "", {"score": score, "best_known": bk})

    # ------------------------------------------------------------------ #
    # KILL 1 (part a) — no witness, no result.
    # ------------------------------------------------------------------ #
    def _check_witness_present(self, verdict: Verdict) -> CheckResult:
        name = "witness_present"
        witness = verdict.detail or {}
        if not isinstance(witness, dict) or not witness:
            return CheckResult(
                name,
                False,
                KILL,
                "verdict carries no witness (detail is empty) — a score with no evidence a third "
                "party can re-check is not a result (problem.py: 'No witness => not a result')",
            )
        try:
            json.dumps(witness, default=str)
        except (TypeError, ValueError) as exc:
            return CheckResult(
                name, False, KILL, f"witness is not JSON-serializable ({exc}) — it cannot be published"
            )
        return CheckResult(name, True, KILL, "", {"witness_keys": sorted(witness)[:20]})

    # ------------------------------------------------------------------ #
    # Candidate must survive the round-trip that publication will put it through.
    # ------------------------------------------------------------------ #
    def _check_candidate_serializable(self, problem: Problem, candidate: Any) -> CheckResult:
        name = "candidate_serializable"
        try:
            blob = _candidate_json(problem, candidate)
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                name,
                False,
                KILL,
                f"candidate is not JSON-serializable ({type(exc).__name__}: {exc}) — it cannot be "
                "written to the Ledger or re-checked by anyone else. Supply to_jsonable().",
            )
        return CheckResult(name, True, KILL, "", {"bytes": len(blob)})

    # ------------------------------------------------------------------ #
    # KILL 8 — Score provenance. Never trust the candidate's self-report.
    # ------------------------------------------------------------------ #
    def _check_score_provenance(
        self, problem: Problem, verdict: Verdict, candidate: Any
    ) -> list[CheckResult]:
        name = "score_provenance"
        out: list[CheckResult] = []

        # (a) The score in the verdict must be the score the VERIFIER computes, right now, here.
        try:
            recomputed = problem.verify(candidate)
        except Exception as exc:  # noqa: BLE001
            return [
                CheckResult(name, False, KILL, f"verify() raised while re-scoring the candidate: {exc!r}")
            ]
        if not recomputed.valid:
            return [
                CheckResult(
                    name,
                    False,
                    KILL,
                    "re-running the verifier on the recorded candidate says valid=False, but the "
                    "claimed verdict says valid=True — the verdict did not come from the verifier",
                )
            ]
        if not _close(recomputed.score, verdict.score, 1e-12):
            return [
                CheckResult(
                    name,
                    False,
                    KILL,
                    f"claimed score={verdict.score!r} but the verifier recomputes "
                    f"{recomputed.score!r} on the same candidate. The score was NOT produced by the "
                    "verifier — something (the program, a cache, the search state) supplied it",
                )
            ]
        out.append(CheckResult(name, True, KILL, "", {"recomputed": _jsonable_number(recomputed.score)}))

        # (b) Actively try to fool the verifier: if the candidate carries a self-reported number and
        # the verifier READS it, then a mutated program can simply claim a world record.
        for key, sentinel_score in self._probe_self_report(problem, candidate, verdict):
            if sentinel_score is None:
                continue
            if _close(sentinel_score, _SELF_REPORT_SENTINEL, 1e-6):
                out.append(
                    CheckResult(
                        f"{name}[self_report:{key}]",
                        False,
                        KILL,
                        f"the verifier ECHOES the candidate's self-reported field {key!r}: setting it "
                        f"to {_SELF_REPORT_SENTINEL} makes verify() return that same number. The "
                        "program is grading its own homework — any 'record' from this target is "
                        "whatever the mutated code decided to write down",
                    )
                )
            else:
                out.append(
                    CheckResult(
                        f"{name}[self_report:{key}]",
                        False,
                        WARN,
                        f"verify()'s score is sensitive to the candidate's self-reported field "
                        f"{key!r} (perturbing it changed the score to {sentinel_score!r}). If that "
                        "field is structural this is fine; if it is the program's own claim, it is a "
                        "hole. Confirm which.",
                    )
                )
        return out

    def _probe_self_report(
        self, problem: Problem, candidate: Any, verdict: Verdict
    ) -> list[tuple[str, float | None]]:
        """Perturb every score-shaped field in the candidate and see if the verdict follows."""
        if not isinstance(candidate, dict):
            return []
        probes: list[tuple[str, float | None]] = []
        for key, value in candidate.items():
            if not isinstance(key, str) or not _SELF_REPORT_RE.match(key):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            tampered = dict(candidate)
            tampered[key] = _SELF_REPORT_SENTINEL
            try:
                v = problem.verify(tampered)
            except Exception:  # noqa: BLE001 — a raise here is caught by the garbage controls
                probes.append((key, None))
                continue
            if not v.valid:
                # Rejecting the tampered candidate is the CORRECT behaviour: the verifier noticed.
                continue
            if _close(v.score, verdict.score, 1e-12):
                # Score unchanged => the verifier ignored the self-report. Good.
                continue
            probes.append((key, _as_float(v.score)))
        return probes

    # ------------------------------------------------------------------ #
    # Determinism — the failure that produced "win" #1 (a delta that was judge noise).
    # ------------------------------------------------------------------ #
    def _check_verifier_determinism(
        self, problem: Problem, candidate: Any, verdict: Verdict
    ) -> CheckResult:
        name = "verifier_determinism"
        seen: set[tuple[bool, str]] = set()
        for _ in range(max(2, self.determinism_reps)):
            try:
                v = problem.verify(candidate)
            except Exception as exc:  # noqa: BLE001
                return CheckResult(name, False, KILL, f"verify() raised on a repeat call: {exc!r}")
            seen.add((bool(v.valid), repr(_jsonable_number(v.score))))
        if len(seen) > 1:
            return CheckResult(
                name,
                False,
                KILL,
                f"verify() is NONDETERMINISTIC: {self.determinism_reps} runs on the identical "
                f"candidate produced {len(seen)} different verdicts {sorted(seen)!r}. A stochastic "
                "scorer means the 'improvement' is a draw from a distribution — and the engine takes "
                "the max over millions of draws, so it will ALWAYS beat the baseline eventually. "
                "This is exactly how a noise delta becomes a headline",
            )
        return CheckResult(name, True, KILL, "", {"reps": self.determinism_reps})

    # ------------------------------------------------------------------ #
    # KILL 7 — Degenerate / trivial candidate.
    # ------------------------------------------------------------------ #
    def _check_degenerate(self, problem: Problem, candidate: Any) -> CheckResult:
        name = "degenerate_candidate"
        generic = _generic_degeneracy(candidate)
        if generic:
            return CheckResult(
                name,
                False,
                KILL,
                f"{generic} — a search that 'wins' with a degenerate object has found a hole in the "
                "objective, not a discovery",
            )
        hook = getattr(problem, "is_degenerate", None)
        if hook is None:
            return CheckResult(
                name,
                True,
                KILL,
                "",
                {
                    "note": "generic degeneracy checks only; target exposes no is_degenerate()",
                },
            )
        try:
            why = hook(candidate)
        except Exception as exc:  # noqa: BLE001
            return CheckResult(name, False, KILL, f"is_degenerate() raised: {exc!r}")
        if why:
            return CheckResult(name, False, KILL, f"target rejects the candidate as degenerate: {why}")
        return CheckResult(name, True, KILL)

    # ------------------------------------------------------------------ #
    # KILL 4 — Class membership, re-checked INDEPENDENTLY of verify().
    # ------------------------------------------------------------------ #
    def _check_class_membership(self, problem: Problem, candidate: Any) -> CheckResult:
        name = "class_membership"
        hook = getattr(problem, "independent_check", None)
        if hook is None:
            return CheckResult(
                name,
                False,
                KILL,
                f"target {getattr(problem, 'name', '?')!r} exposes no independent_check() — the "
                "claim that the candidate satisfies the hard constraints rests entirely on the SAME "
                "verify() that produced the score. A verifier bug that accepts illegal candidates is "
                "the highest-severity failure mode there is, and one implementation cannot detect it. "
                "Supply a second, deliberately separate implementation.",
            )
        try:
            ok, why = hook(candidate)
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                name,
                False,
                KILL,
                f"independent_check() raised ({exc!r}) — cannot confirm the candidate is even a legal "
                "member of the problem class",
            )
        if not ok:
            return CheckResult(
                name,
                False,
                KILL,
                f"INDEPENDENT re-check says the candidate does NOT satisfy the problem's hard "
                f"constraints: {why}. verify() accepted it anyway — trust the second implementation "
                "and treat verify() as broken until proven otherwise",
            )
        return self._check_independent_check_is_not_vacuous(problem, str(why))

    def _check_independent_check_is_not_vacuous(self, problem: Problem, why: str) -> CheckResult:
        """Who checks the checker?

        `independent_check` is the highest-severity check in this file — it is the only thing standing
        between a verifier bug and a published illegal object. But the auditor cannot tell a real
        second implementation from `return True, "looks fine"`. A lazy or stubbed-out hook silently
        converts the strongest kill into a no-op, and the audit still says PASS. That is the same
        vacuous-truth failure the auditor exists to catch, one level up.

        So make it prove it can REJECT: run it against the target's own known-bad controls and against
        garbage. An independent check that has never said "no" to anything has not been shown to be a
        check at all.
        """
        name = "class_membership"
        rejected: list[str] = []
        accepted_bad: list[str] = []

        negatives: list[tuple[str, Any]] = [
            ("garbage:none", None),
            ("garbage:empty_list", []),
            ("garbage:string", "not a candidate"),
        ]
        controls_fn = getattr(problem, "controls", None)
        if controls_fn is not None:
            try:
                negatives += [
                    (f"control:{c.name}", c.candidate) for c in controls_fn() if not c.expect_valid
                ]
            except Exception:  # noqa: BLE001 — controls already audited upstream
                pass

        for label, bad in negatives:
            try:
                ok, _ = problem.independent_check(bad)   # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 — raising on garbage counts as rejecting it
                rejected.append(label)
                continue
            (accepted_bad if ok else rejected).append(label)

        if accepted_bad:
            return CheckResult(
                name,
                False,
                KILL,
                f"independent_check() ACCEPTS known-bad inputs {accepted_bad!r} — it is vacuous. The "
                "one check standing between a verifier bug and a published illegal object does not "
                "actually reject anything, so its 'confirmation' of the candidate means nothing",
                {"accepted_bad": accepted_bad},
            )
        if not rejected:
            return CheckResult(
                name,
                False,
                KILL,
                "independent_check() has not been shown to reject ANYTHING — no negative control and "
                "no garbage input was refused, so it cannot be distinguished from `return True`",
            )
        return CheckResult(
            name,
            True,
            KILL,
            "",
            {"independent_check": why[:200], "rejects": rejected},
        )

    # ------------------------------------------------------------------ #
    # KILL 2 — Baseline realness. THIS is the one that dissolved our last two wins.
    # ------------------------------------------------------------------ #
    def _check_baseline_realness(
        self, problem: Problem, best_known: float | None
    ) -> list[CheckResult]:
        name = "baseline_realness"
        hook = getattr(problem, "best_known_provenance", None)
        if hook is None:
            return [
                CheckResult(
                    name,
                    False,
                    KILL,
                    f"best_known() for target {getattr(problem, 'name', '?')!r} has NO source "
                    "attribution: the target exposes no best_known_provenance(), so the number we are "
                    "'beating' is an unsourced literal in our own code. This is precisely the failure "
                    "that dissolved the last two positive results. Supply BestKnownSource(value, "
                    "source, citation, retrieved_at, kind).",
                )
            ]
        try:
            src = hook()
        except Exception as exc:  # noqa: BLE001
            return [CheckResult(name, False, KILL, f"best_known_provenance() raised: {exc!r}")]
        if not isinstance(src, BestKnownSource):
            return [
                CheckResult(
                    name,
                    False,
                    KILL,
                    f"best_known_provenance() returned {type(src).__name__}, expected BestKnownSource",
                )
            ]

        results: list[CheckResult] = []
        for problem_str in src.problems():
            results.append(CheckResult(name, False, KILL, problem_str))

        # The cited number and the number the code actually compares against must be the SAME number.
        bk = _as_float(best_known)
        if bk is not None and math.isfinite(bk) and not _close(bk, float(src.value), 1e-9):
            results.append(
                CheckResult(
                    f"{name}[drift]",
                    False,
                    KILL,
                    f"best_known() returns {bk!r} but the cited source ({src.citation}) says "
                    f"{src.value!r}. The code is not comparing against the record it claims to cite — "
                    "one of them is wrong, and either way the delta is meaningless",
                )
            )

        age = self._now() - float(src.retrieved_at or 0)
        if src.retrieved_at and age > self.max_baseline_age_s:
            results.append(
                CheckResult(
                    f"{name}[stale]",
                    False,
                    WARN,
                    f"the baseline was last confirmed against {src.citation} {age / 86400:.0f} days "
                    "ago. Records move; re-confirm before publishing or the 'beat' may already be old "
                    "news",
                )
            )

        if not any(c.is_kill for c in results):
            results.append(
                CheckResult(
                    name,
                    True,
                    KILL,
                    "",
                    {"source": src.source, "citation": src.citation, "kind": src.kind},
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # KILL 3 — Trivial rediscovery.
    # ------------------------------------------------------------------ #
    def _check_trivial_rediscovery(
        self, problem: Problem, candidate: Any, verdict: Verdict
    ) -> CheckResult:
        name = "trivial_rediscovery"
        witness = dict(verdict.detail or {})

        # Generic evidence sniff — the domain-general core of coding_theory.is_table_lookup_evidence.
        table = _looks_like_table_lookup(witness)
        if table:
            return CheckResult(
                name,
                False,
                KILL,
                f"{table} — reproducing a tabulated value is not a discovery, and a 'score' that came "
                "from a lookup is not a computation",
            )

        hook = getattr(problem, "trivial_rediscovery", None)
        if hook is None:
            return CheckResult(
                name,
                True,
                WARN,
                "target exposes no trivial_rediscovery() hook — only the generic table-lookup sniff "
                "ran. Domain guards (e.g. coding_theory.trivial_rediscovery) should be wired in here",
            )
        try:
            why = hook(candidate, verdict)
        except Exception as exc:  # noqa: BLE001
            return CheckResult(name, False, KILL, f"trivial_rediscovery() raised: {exc!r}")
        if why:
            return CheckResult(
                name,
                False,
                KILL,
                f"this is a REDISCOVERY, not a discovery: {why}",
            )
        return CheckResult(name, True, KILL)

    # ------------------------------------------------------------------ #
    # KILL 6 — Numerical / floating-point artifact.
    # ------------------------------------------------------------------ #
    def _check_numerical_artifact(
        self, problem: Problem, verdict: Verdict, candidate: Any, best_known: float | None
    ) -> list[CheckResult]:
        name = "float_artifact"
        results: list[CheckResult] = []
        score = _as_float(verdict.score)
        bk = _as_float(best_known)
        if score is None or bk is None or not math.isfinite(score) or not math.isfinite(bk):
            return [CheckResult(name, True, KILL, "", {"skipped": "non-finite handled upstream"})]

        margin = score - bk
        tol = max(FLOAT_ATOL, FLOAT_RTOL * abs(bk))
        if margin <= tol:
            results.append(
                CheckResult(
                    name,
                    False,
                    KILL,
                    f"the 'improvement' is {margin!r} over a baseline of {bk!r} — that is within "
                    f"floating-point tolerance ({tol:.3g}). A margin this thin is arithmetic noise, "
                    "not a discovery",
                    {"margin": margin, "tolerance": tol},
                )
            )
        elif abs(bk) > 0 and margin / abs(bk) < THIN_MARGIN_RTOL:
            results.append(
                CheckResult(
                    name,
                    False,
                    WARN,
                    f"the margin is real but very thin: {margin:.3g} on a baseline of {bk!r} "
                    f"({margin / abs(bk):.2e} relative). Re-check in exact arithmetic before claiming "
                    "anything",
                    {"margin": margin},
                )
            )
        else:
            results.append(CheckResult(name, True, KILL, "", {"margin": margin}))

        # Does the win depend on a slop parameter someone picked?
        witness = dict(verdict.detail or {})
        for key in _TOLERANCE_KEYS:
            if key not in witness:
                continue
            tol_val = _as_float(witness.get(key))
            if tol_val is None:
                continue
            if margin <= abs(tol_val):
                results.append(
                    CheckResult(
                        f"{name}[tolerance:{key}]",
                        False,
                        KILL,
                        f"the claimed margin ({margin:.6g}) is smaller than the verifier's own "
                        f"{key}={tol_val:.6g}. The 'improvement' lives entirely inside the slop the "
                        "verifier was told to allow — change the tolerance and the discovery vanishes",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        f"{name}[tolerance:{key}]",
                        False,
                        WARN,
                        f"the verdict depends on a tolerance choice ({key}={tol_val!r}). State it in "
                        "the publication and show the result survives a stricter one",
                    )
                )

        # The real test: does the margin survive exact arithmetic?
        exact_fn = getattr(problem, "exact_score", None)
        if exact_fn is None:
            results.append(
                CheckResult(
                    name + "[exact]",
                    True,
                    WARN,
                    "target exposes no exact_score() — the margin was only ever checked in floating "
                    "point. Where the objective is integral or rational, re-check it exactly",
                )
            )
            return results
        try:
            exact = exact_fn(candidate)
        except Exception as exc:  # noqa: BLE001
            return results + [CheckResult(name + "[exact]", False, KILL, f"exact_score() raised: {exc!r}")]
        if exact is None:
            results.append(
                CheckResult(
                    name + "[exact]", True, WARN, "exact_score() returned None for this candidate"
                )
            )
            return results

        exact_bk = _exact_best_known(problem, bk)
        exact_margin = Fraction(exact) - exact_bk
        if exact_margin <= 0:
            results.append(
                CheckResult(
                    name + "[exact]",
                    False,
                    KILL,
                    f"under EXACT arithmetic the candidate scores {exact} against a baseline of "
                    f"{exact_bk} — margin {exact_margin}. The float margin of {margin:.6g} was an "
                    "artifact. A margin that vanishes under exact arithmetic is not a discovery",
                    {"exact_score": str(exact), "exact_best_known": str(exact_bk)},
                )
            )
        else:
            results.append(
                CheckResult(
                    name + "[exact]",
                    True,
                    KILL,
                    "",
                    {"exact_margin": str(exact_margin)},
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # KILL 1 — Reproduce from the record alone, in a FRESH process.
    # ------------------------------------------------------------------ #
    def _check_reproduce_from_witness(
        self, problem: Problem, verdict: Verdict, candidate: Any, report: AuditReport
    ) -> list[CheckResult]:
        name = "reproduce_from_witness"
        repro = self._reproduce(problem, verdict, candidate)
        if not repro.ok:
            return [
                CheckResult(
                    name,
                    False,
                    KILL,
                    f"a FRESH process could not re-check the claim from the record alone: "
                    f"{repro.error}. If a clean clone cannot re-verify the witness in one command, we "
                    "do not have a result — we have a number that only exists inside our own process",
                    {"error": repro.error},
                )
            ]
        if not repro.valid:
            return [
                CheckResult(
                    name,
                    False,
                    KILL,
                    "in a fresh process, verifying the SERIALIZED candidate returns valid=False. The "
                    "record does not reproduce — the in-memory object and the recorded one are not "
                    "the same thing",
                )
            ]
        if not _close(repro.score, verdict.score, 1e-9):
            return [
                CheckResult(
                    name,
                    False,
                    KILL,
                    f"the recorded claim ({verdict.score!r}) does not reproduce from the record: a "
                    f"fresh process re-verifying the serialized candidate gets {repro.score!r}. "
                    "Something in the live process (a cache, the program, search state) was carrying "
                    "the result",
                    {
                        "claimed": _jsonable_number(verdict.score),
                        "reproduced": _jsonable_number(repro.score),
                    },
                )
            ]
        if repro.witness_ok is False:
            return [
                CheckResult(
                    name,
                    False,
                    KILL,
                    f"the recorded WITNESS does not check out against the candidate: "
                    f"{repro.witness_reason}. The evidence we would publish does not support the "
                    "claim we would publish",
                )
            ]
        if repro.fingerprint and report.verifier_fingerprint and (
            repro.fingerprint != report.verifier_fingerprint
        ):
            return [
                CheckResult(
                    name,
                    False,
                    KILL,
                    f"verifier fingerprint drifted between the claim ({report.verifier_fingerprint}) "
                    f"and the fresh re-check ({repro.fingerprint}) — the code that checked this claim "
                    "is not the code on disk. The record is unreproducible whatever the numbers say",
                )
            ]

        results = [
            CheckResult(
                name,
                True,
                KILL,
                "",
                {
                    "reproduced_score": _jsonable_number(repro.score),
                    "reconstruction": repro.reconstruction,
                },
            )
        ]
        if repro.witness_ok is None:
            results.append(
                CheckResult(
                    f"{name}[witness_unchecked]",
                    False,
                    WARN,
                    "target exposes no check_witness(): the CANDIDATE reproduced, but the witness we "
                    "would publish as the evidence was never itself validated. Nobody has confirmed "
                    "the evidence supports the claim",
                )
            )
        if repro.reconstruction == "pickle" and repro.carried_state:
            results.append(
                CheckResult(
                    f"{name}[state_carried]",
                    False,
                    WARN,
                    "the re-check process could only be given the verifier by UNPICKLING it, which "
                    f"carries the live instance's state across with it ({', '.join(repro.carried_state)}). "
                    "The process is fresh; the object is not. An instance-level memo cache or a "
                    "pre-loaded baseline would survive the trip and reproduce its own wrong answer. "
                    "Implement audit_spec() so the child rebuilds the verifier from scratch",
                    {"carried": repro.carried_state},
                )
            )
        return results

    def _reproduce(self, problem: Problem, verdict: Verdict, candidate: Any) -> ReproResult:
        """Re-verify in a clean interpreter given ONLY the serialized candidate + witness + verifier.

        Deliberately does NOT receive: the program, the island/search state, or any live object.
        """
        try:
            candidate_json = _candidate_json(problem, candidate)
        except Exception as exc:  # noqa: BLE001
            return ReproResult(False, error=f"candidate is not serializable: {exc!r}")

        with tempfile.TemporaryDirectory(prefix="propab-audit-") as tmp:
            tmpdir = Path(tmp)
            spec = None
            pkl_path = None
            reconstruction = "audit_spec"
            carried: list[str] = []

            spec_fn = getattr(problem, "audit_spec", None)
            if spec_fn is not None:
                try:
                    spec = spec_fn()
                except Exception as exc:  # noqa: BLE001
                    return ReproResult(False, error=f"audit_spec() raised: {exc!r}")

            if spec is None:
                reconstruction = "pickle"
                carried = _mutable_instance_state(problem)
                pkl_path = tmpdir / "problem.pkl"
                try:
                    pkl_path.write_bytes(pickle.dumps(problem))
                except Exception as exc:  # noqa: BLE001
                    return ReproResult(
                        False,
                        error=(
                            f"the Problem cannot be reconstructed in a fresh process "
                            f"(pickle failed: {type(exc).__name__}: {exc}) and it exposes no "
                            "audit_spec(). A verifier that only exists inside this process cannot "
                            "re-check anything for anyone else"
                        ),
                    )

            payload = {
                "sys_path": _child_sys_path(),
                "problem_pkl": str(pkl_path) if pkl_path else None,
                "problem_spec": spec,
                "candidate_json": candidate_json,
                "witness_json": json.dumps(dict(verdict.detail or {}), default=str),
                "expect_fingerprint": "",
            }
            payload_path = tmpdir / "payload.json"
            payload_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
            script_path = tmpdir / "recheck.py"
            script_path.write_text(_CHILD_SOURCE, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path), str(payload_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.repro_timeout_s,
                    cwd=str(tmpdir),   # so nothing in the repo cwd can be implicitly imported
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return ReproResult(
                    False,
                    error=f"the fresh re-check did not finish within {self.repro_timeout_s}s",
                )

            line = next(
                (
                    ln[len(_MARKER):]
                    for ln in reversed((proc.stdout or "").splitlines())
                    if ln.startswith(_MARKER)
                ),
                None,
            )
            if line is None:
                err = (proc.stderr or "").strip()[-800:] or "(no output)"
                return ReproResult(
                    False, error=f"the fresh re-check produced no verdict (exit={proc.returncode}): {err}"
                )
            try:
                data = json.loads(line)
            except ValueError as exc:
                return ReproResult(False, error=f"unparseable re-check output: {exc}")

        if not data.get("ok"):
            return ReproResult(
                False,
                error=str(data.get("error") or "unknown failure"),
                reconstruction=reconstruction,
                carried_state=carried,
            )
        return ReproResult(
            ok=True,
            valid=bool(data.get("valid")),
            score=_as_float(data.get("score")) or float("-inf"),
            detail=data.get("detail") or {},
            fingerprint=str(data.get("fingerprint") or ""),
            witness_ok=data.get("witness_ok"),
            witness_reason=str(data.get("witness_reason") or ""),
            reconstruction=reconstruction,
            carried_state=carried,
        )

    # ------------------------------------------------------------------ #
    # Program hygiene (advisory — the auditor never runs the program).
    # ------------------------------------------------------------------ #
    def _check_program_hygiene(self, program: Program | None) -> CheckResult:
        name = "program_hygiene"
        if program is None:
            return CheckResult(name, True, WARN, "", {"note": "no program supplied"})
        code = program.code or ""
        if _UNSEEDED_RNG_RE.search(code) and not _SEEDED_RE.search(code):
            return CheckResult(
                name,
                False,
                WARN,
                "the generating program uses randomness with no visible seed — the candidate itself "
                "reproduces (that is what the witness is for), but nobody can regenerate it from the "
                "program. Seed the RNG before publishing the construction",
            )
        return CheckResult(name, True, WARN)

    # ------------------------------------------------------------------ #
    # Per-target extra kill checks.
    # ------------------------------------------------------------------ #
    def _run_target_checks(
        self,
        problem: Problem,
        verdict: Verdict,
        candidate: Any,
        program: Program | None,
        best_known: float | None,
    ) -> list[CheckResult]:
        hook = getattr(problem, "audit_checks", None)
        if hook is None:
            return []
        ctx = AuditContext(
            problem=problem,
            verdict=verdict,
            candidate=candidate,
            program=program,
            best_known=_as_float(best_known) or float("-inf"),
            witness=dict(verdict.detail or {}),
        )
        out: list[CheckResult] = []
        try:
            extra = list(hook())
        except Exception as exc:  # noqa: BLE001
            return [CheckResult("target_checks", False, KILL, f"audit_checks() raised: {exc!r}")]
        for fn in extra:
            label = getattr(fn, "__name__", repr(fn))
            try:
                res = fn(ctx)
            except Exception as exc:  # noqa: BLE001
                out.append(
                    CheckResult(
                        f"target[{label}]",
                        False,
                        KILL,
                        f"target check {label!r} raised ({exc!r}) — an audit check that cannot run is "
                        "a kill, not a pass",
                    )
                )
                continue
            if isinstance(res, CheckResult):
                out.append(res)
            elif isinstance(res, Sequence):
                out.extend(r for r in res if isinstance(r, CheckResult))
            else:
                out.append(
                    CheckResult(
                        f"target[{label}]",
                        False,
                        KILL,
                        f"target check {label!r} returned {type(res).__name__}, expected CheckResult",
                    )
                )
        return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mutable_instance_state(problem: Problem) -> list[str]:
    """Instance attributes that pickle would carry into the 'fresh' process.

    A dict/set/list living on `self` is exactly where a memo cache hides. Pickling the Problem takes
    it along, so the child would faithfully reproduce the cached answer — a fresh process holding a
    stale object. Name them so the warning is actionable.
    """
    state = getattr(problem, "__dict__", None)
    if not isinstance(state, dict):
        return []
    return sorted(
        f"{key}: {type(value).__name__}"
        for key, value in state.items()
        if isinstance(value, (dict, set, list, bytearray)) and value
    )


def _child_sys_path() -> list[str]:
    """The parent's import path, made absolute, for the re-check process.

    The child deliberately runs in a scratch cwd (so nothing is implicitly imported from wherever the
    engine happened to be launched). That makes RELATIVE sys.path entries — including "" , which is
    how `python -m ...` spells "the current directory" — meaningless there: they would silently
    resolve against the scratch dir, the verifier would fail to import, and a perfectly good
    discovery would be killed for a reason that has nothing to do with the discovery.

    So resolve them against the PARENT's cwd. Imports still have to come from an explicit path entry;
    they just point where they pointed before.
    """
    cwd = Path.cwd()
    out: list[str] = []
    seen: set[str] = set()
    for entry in sys.path:
        path = Path(entry) if entry not in ("", ".") else cwd
        resolved = str(path if path.is_absolute() else (cwd / path))
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _safe(fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonable_number(value: Any) -> Any:
    f = _as_float(value)
    if f is None:
        return str(value)
    return f if math.isfinite(f) else str(f)


def _close(a: Any, b: Any, tol: float) -> bool:
    fa, fb = _as_float(a), _as_float(b)
    if fa is None or fb is None:
        return False
    if math.isinf(fa) or math.isinf(fb):
        return fa == fb
    if math.isnan(fa) or math.isnan(fb):
        return False
    return abs(fa - fb) <= max(tol, tol * max(abs(fa), abs(fb)))


def _exact_best_known(problem: Problem, fallback: float) -> Fraction:
    hook = getattr(problem, "exact_best_known", None)
    if hook is not None:
        try:
            return Fraction(hook())
        except Exception:  # noqa: BLE001
            pass
    return Fraction(fallback)


def _candidate_json(problem: Problem, candidate: Any) -> str:
    hook = getattr(problem, "to_jsonable", None)
    obj = hook(candidate) if hook is not None else candidate
    return json.dumps(obj, default=_json_default)


def _json_default(obj: Any) -> Any:
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001
            pass
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable")


def _looks_like_table_lookup(witness: dict[str, Any]) -> str | None:
    """The domain-general core of coding_theory.is_table_lookup_evidence."""
    src = str(witness.get("construction_source") or "").strip().lower()
    if src in _TABLE_SOURCE_VALUES:
        return f"the witness declares construction_source={src!r}"
    method = str(
        witness.get("verification_method") or witness.get("method") or ""
    ).strip().lower()
    if method in _TABLE_METHOD_VALUES:
        return f"the witness declares method={method!r}, i.e. the score came from a table"
    if witness.get("from_table") is True or witness.get("table_lookup") is True:
        return "the witness is flagged as a table lookup"
    return None


def _generic_degeneracy(candidate: Any) -> str | None:
    """The classic ways a search 'wins' by exploiting a hole in the objective."""
    if candidate is None:
        return "the candidate is None"

    tolist = getattr(candidate, "tolist", None)
    shape = getattr(candidate, "shape", None)
    if callable(tolist) and shape is not None:   # numpy-like, without importing numpy
        size = getattr(candidate, "size", None)
        if size == 0:
            return "the candidate array is empty (size 0)"
        try:
            if not any(bool(x) for x in _flatten(candidate.tolist())):
                return "the candidate array is all-zero"
        except Exception:  # noqa: BLE001
            return None
        return None

    if isinstance(candidate, (str, bytes)):
        return "the candidate is empty" if len(candidate) == 0 else None

    if isinstance(candidate, dict):
        if not candidate:
            return "the candidate is an empty dict"
        return None

    if isinstance(candidate, (list, tuple, set, frozenset)):
        if len(candidate) == 0:
            return "the candidate is empty (size 0)"
        flat = _flatten(candidate)
        if flat and all(isinstance(x, (int, float)) and x == 0 for x in flat):
            return "the candidate is all-zero"
        if all(isinstance(row, (list, tuple)) and len(row) == 0 for row in candidate):
            return "every row of the candidate is empty"
        return None

    return None


def _flatten(obj: Any, depth: int = 0) -> list[Any]:
    if depth > 8:
        return []
    if isinstance(obj, (list, tuple, set, frozenset)):
        out: list[Any] = []
        for item in obj:
            out.extend(_flatten(item, depth + 1))
        return out
    return [obj]


__all__ = [
    "AUDITOR_VERSION",
    "Auditor",
    "AuditableProblem",
    "AuditContext",
    "AuditError",
    "AuditReport",
    "BestKnownSource",
    "CheckResult",
    "Control",
    "KILL",
    "WARN",
    "verifier_fingerprint",
]
