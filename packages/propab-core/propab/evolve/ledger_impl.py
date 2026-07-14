"""Evolve — the Ledger implementation (WS-5a).

`FileLedger` is the concrete `Ledger`: an append-only store of verified improvements plus the
publication export. `ledger.py` is the frozen contract; this file supplies the behaviour.

The credibility model (copied, deliberately, from OpenAI's CDC Lean release)
---------------------------------------------------------------------------
CDC did not ask anyone to believe them. They shipped a repository where

    lake build CDCLean

re-checks the theorem from scratch, plus an audit that scans for `sorry`/`native_decide`/`axiom` —
i.e. they shipped the *checker*, pinned the *toolchain*, and published the *exact command*. Our
equivalent, for an object rather than a proof, is:

    cd <bundle> && python verify.py        # exit 0 == PASS, exit 1 == FAIL

The bundle contains the construction, the witness, the **exact verifier source that judged it**, a
standalone runner, and a sha256 MANIFEST. It imports nothing from propab. A third party never has
to trust us, run our engine, or even read our code: they read `verifier.py` (which is a literal
transcription of the problem's constraints) and run one command.

**If a result is not independently re-checkable from the bundle alone, it is not a result.** That is
enforced here, not merely documented:

  * `record()` refuses a Record whose `verifier_fingerprint` does not hash the verifier source it
    ships with (a fingerprint that does not pin the code is worthless).
  * `record()` re-runs that verifier itself, in a fresh namespace, on the persisted candidate, and
    refuses the Record if the re-check disagrees with the claimed verdict or score.
  * `record()` refuses a Record that does not strictly beat its own `best_known_at_time`.
  * `export_publishable()` refuses to emit a bundle it cannot itself re-check by subprocess.

Every refusal is written to `rejected.jsonl` with a reason. Nothing is silently swallowed — this
project has twice reported a number that dissolved under scrutiny, and the ledger is the last place
that can catch it.

Sourced baselines: `math_combinatorics/discovery/record_registry.py` is the repo's table of REAL,
citable best-known values. The export cites it, so the bundle states *what* we claim to beat and
*where that number came from* — never a model's memory.

The standalone-verifier contract
--------------------------------
A target that wants publishable records must supply a stdlib-only Python module source defining::

    VERIFIER_ID = "<stable id>"

    def verify(candidate) -> dict:
        # {"valid": bool, "score": float, "detail": {...}}

`verify` must never raise and must be importable with no third-party dependencies. The target hands
that source to the ledger via `Record.witness["verifier_source"]` (or `register_verifier()`), and
`Record.verifier_fingerprint` must equal `verifier_fingerprint(source)`.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from propab.domain_modules.math_combinatorics.discovery import record_registry

from .ledger import Ledger, Record

__all__ = [
    "FileLedger",
    "canonical_json",
    "content_hash",
    "verifier_fingerprint",
    "STANDALONE_VERIFIER_CONTRACT",
    "SCORE_RECHECK_TOL",
]

# Relative tolerance when comparing a claimed score against the ledger's own re-check.
# Scores are floats (objectives routinely involve log/exp); validity is never float-dependent —
# that is the target's job to make exact.
SCORE_RECHECK_TOL = 1e-9

STANDALONE_VERIFIER_CONTRACT = """\
Standalone verifier source must be a stdlib-only Python module defining:

    VERIFIER_ID = "<stable id>"

    def verify(candidate) -> dict:
        \"\"\"Return {"valid": bool, "score": float, "detail": {...}}. MUST NOT raise.\"\"\"

It must not import propab, must not touch the network, and must be readable as a literal
transcription of the problem's constraints — an auditor has to believe it by reading it.
"""


# --------------------------------------------------------------------------- #
# Content addressing
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    """Deterministic JSON for hashing. Dict keys sorted; list order preserved (it is data).

    Targets whose candidate is an unordered *set* must emit it in a canonical order (e.g. sorted)
    so that two spellings of the same object collide here — that is what makes dedupe work.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(problem: str, candidate: Any) -> str:
    """Identity of a discovery: the problem it answers + the object itself."""
    payload = canonical_json({"problem": problem, "candidate": candidate})
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verifier_fingerprint(source: str) -> str:
    """Pin the exact code that judged a claim. Byte-for-byte: whitespace is part of the pin."""
    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(problem: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in problem]
    return "".join(keep)[:120] or "unnamed"


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #
class FileLedger(Ledger):
    """Append-only, file-backed `Ledger`.

    Layout under ``root``::

        <problem-slug>/records.jsonl     append-only; one accepted Record per line
        <problem-slug>/rejected.jsonl    append-only; every refusal, with its reason
        verifiers/<fingerprint>.py       content-addressed verifier blobs

    Append-only means append-only: a line is never rewritten or deleted. Superseded records stay —
    the history of what we believed and when is part of the evidence.
    """

    RECORDS = "records.jsonl"
    REJECTED = "rejected.jsonl"
    VERIFIERS = "verifiers"

    def __init__(self, root: str | Path = "artifacts/evolve") -> None:
        super().__init__(root)
        self._lock = threading.Lock()  # the engine runs many workers; appends must not interleave
        (self.root / self.VERIFIERS).mkdir(parents=True, exist_ok=True)

    # ---------------- verifier blob store ---------------- #
    def register_verifier(self, source: str) -> str:
        """Store a standalone verifier source, content-addressed. Returns its fingerprint."""
        fp = verifier_fingerprint(source)
        path = self.root / self.VERIFIERS / f"{fp.split(':', 1)[1]}.py"
        if not path.exists():
            # write_bytes, never write_text: the fingerprint is over exact bytes, and write_text
            # would translate "\n" to "\r\n" on Windows and silently break the pin.
            path.write_bytes(source.encode("utf-8"))
        return fp

    def verifier_source(self, fingerprint: str) -> str | None:
        """The exact source behind a fingerprint, or None if we never stored it."""
        if not fingerprint or ":" not in fingerprint:
            return None
        path = self.root / self.VERIFIERS / f"{fingerprint.split(':', 1)[1]}.py"
        if not path.exists():
            return None
        source = path.read_text(encoding="utf-8")
        # Paranoia: the blob store is content-addressed, so a mismatch means the file was edited.
        if verifier_fingerprint(source) != fingerprint:
            return None
        return source

    # ---------------- paths ---------------- #
    def _dir(self, problem: str) -> Path:
        d = self.root / _slug(problem)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _append(self, path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def _reject(self, rec: Record, reason: str, **extra: Any) -> bool:
        self._append(
            self._dir(rec.problem) / self.REJECTED,
            {
                "rejected_at": time.time(),
                "reason": reason,
                "problem": rec.problem,
                "score": rec.score,
                "best_known_at_time": rec.best_known_at_time,
                "program_id": rec.program_id,
                "verifier_fingerprint": rec.verifier_fingerprint,
                "content_hash": content_hash(rec.problem, rec.candidate),
                **extra,
            },
        )
        return False

    # ---------------- read ---------------- #
    def _rows(self, problem: str) -> list[dict[str, Any]]:
        path = self._dir(problem) / self.RECORDS
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def records(self, problem: str) -> list[Record]:
        """Every accepted Record for `problem`, in the order it was written."""
        return [Record(**row["record"]) for row in self._rows(problem)]

    def rejections(self, problem: str) -> list[dict[str, Any]]:
        """Every refusal for `problem`, with its reason. Read this before believing a null result."""
        path = self._dir(problem) / self.REJECTED
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def best(self, problem: str) -> Record | None:
        """The highest-scoring accepted Record. Ties go to the one recorded first (priority)."""
        best: Record | None = None
        for rec in self.records(problem):
            if best is None or rec.score > best.score:
                best = rec
        return best

    # ---------------- write ---------------- #
    def record(self, rec: Record) -> bool:
        """Persist a verified improvement.

        Returns False — never raises — when the Record is a duplicate/rediscovery, or when it fails
        any of the ledger's own independent checks. Every False is logged to `rejected.jsonl`.

        The checks, in order:
          1. dedupe on (problem, candidate content hash);
          2. the shipped verifier source must hash to `verifier_fingerprint`;
          3. the ledger re-runs that verifier on the candidate and must reproduce `valid` and `score`;
          4. the score must strictly beat `best_known_at_time`.
        """
        with self._lock:
            chash = content_hash(rec.problem, rec.candidate)

            # 1. duplicate / rediscovery
            if any(row.get("content_hash") == chash for row in self._rows(rec.problem)):
                return self._reject(rec, "duplicate_candidate")

            # 2. the fingerprint must actually pin the verifier we were handed
            witness = dict(rec.witness or {})
            source = witness.pop("verifier_source", None)
            if source is None:
                source = self.verifier_source(rec.verifier_fingerprint)
            if not source:
                # No verifier source => nobody can re-check this => it is not a result.
                return self._reject(rec, "no_verifier_source")
            if verifier_fingerprint(source) != rec.verifier_fingerprint:
                return self._reject(
                    rec,
                    "verifier_fingerprint_mismatch",
                    actual_fingerprint=verifier_fingerprint(source),
                )

            # 3. independent re-check: we do not take the caller's verdict on trust
            try:
                observed = self._recheck(source, rec.candidate)
            except Exception as exc:  # a verifier that crashes cannot certify anything
                return self._reject(rec, "verifier_crashed", error=f"{type(exc).__name__}: {exc}")
            if not observed.get("valid"):
                return self._reject(rec, "recheck_invalid", observed=observed.get("detail"))
            obs_score = float(observed.get("score", float("-inf")))
            if abs(obs_score - rec.score) > SCORE_RECHECK_TOL * max(1.0, abs(rec.score)):
                return self._reject(rec, "recheck_score_mismatch", observed_score=obs_score)

            # 4. it must actually beat the sourced baseline it claims to beat
            if not rec.score > rec.best_known_at_time:
                return self._reject(rec, "not_an_improvement")

            fp = self.register_verifier(source)
            witness["verifier_blob"] = f"{self.VERIFIERS}/{fp.split(':', 1)[1]}.py"
            stored = Record(**{**asdict(rec), "witness": witness})
            self._append(
                self._dir(rec.problem) / self.RECORDS,
                {"content_hash": chash, "record": asdict(stored)},
            )
            return True

    @staticmethod
    def _recheck(source: str, candidate: Any) -> dict[str, Any]:
        """Run a standalone verifier in a fresh namespace against the *persisted* candidate.

        The candidate is round-tripped through JSON first: a Record that only verifies as a live
        Python object, and not as the bytes we would actually publish, is not publishable.
        """
        ns: dict[str, Any] = {"__name__": "propab_ledger_recheck"}
        exec(compile(source, "<verifier>", "exec"), ns)  # noqa: S102 — content-addressed source
        fn = ns.get("verify")
        if not callable(fn):
            raise TypeError("verifier source defines no callable verify()")
        return fn(json.loads(canonical_json(candidate)))

    # ---------------- publish ---------------- #
    def export_publishable(self, problem: str, dest: str | Path) -> Path:
        """Emit a self-contained bundle for the best Record on `problem`.

        The bundle::

            README.md        the claim, the LIMITATIONS, and the one command to re-check it
            verify.py        the runnable checker  ->  `python verify.py`, exit 0 == PASS
            verifier.py      the EXACT verifier source that judged the claim (stdlib only)
            candidate.json   the construction
            witness.json     the verifier's evidence (margin, violating pair, …)
            record.json      the full Record (score, baseline, program id, fingerprint)
            program.py       the generator that found it (provenance; `verify.py --regenerate`)
            MANIFEST.json    sha256 of every file above

        Before returning, this method *runs* `verify.py` in a subprocess. If the bundle does not
        re-check, it is not shipped: `RuntimeError`. A bundle that we cannot re-check ourselves is
        one we have no business handing to anyone else.
        """
        rec = self.best(problem)
        if rec is None:
            raise LookupError(f"ledger has no record for problem {problem!r}")

        source = self.verifier_source(rec.verifier_fingerprint)
        if not source:
            raise RuntimeError(
                f"no verifier source stored for fingerprint {rec.verifier_fingerprint!r}: "
                "the result cannot be independently re-checked, so it cannot be published"
            )

        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        provenance = self._baseline_provenance(problem, rec)
        # Exact bytes: verify.py re-hashes this file and compares it to `verifier_fingerprint`.
        # write_text() would newline-translate on Windows and break the pin on the one file whose
        # bytes are load-bearing.
        (dest / "verifier.py").write_bytes(source.encode("utf-8"))
        (dest / "candidate.json").write_text(
            json.dumps(rec.candidate, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        (dest / "witness.json").write_text(
            json.dumps(rec.witness, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        (dest / "record.json").write_text(rec.to_json(), encoding="utf-8")
        (dest / "program.py").write_text(rec.program_code or "# (no program recorded)\n", encoding="utf-8")
        (dest / "verify.py").write_text(_VERIFY_RUNNER, encoding="utf-8")
        (dest / "README.md").write_text(self._readme(rec, provenance), encoding="utf-8")

        manifest = {
            "bundle_format": "propab-evolve-bundle/1",
            "problem": rec.problem,
            "verifier_fingerprint": rec.verifier_fingerprint,
            "exported_at": time.time(),
            "files": {
                p.name: _sha256_file(p) for p in sorted(dest.iterdir()) if p.name != "MANIFEST.json"
            },
        }
        (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # The bundle must re-check itself, by the same one command we tell the reader to run.
        # PYTHONDONTWRITEBYTECODE so the self-check does not leave a __pycache__ in what we ship.
        proc = subprocess.run(
            [sys.executable, "verify.py"],
            cwd=dest,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        shutil.rmtree(dest / "__pycache__", ignore_errors=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "export self-check FAILED — refusing to publish a bundle that does not re-check.\n"
                f"exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
            )
        return dest

    # ---------------- baseline provenance ---------------- #
    @staticmethod
    def _baseline_provenance(problem: str, rec: Record) -> dict[str, Any]:
        """Where `best_known_at_time` came from. An unsourced baseline is stated as unsourced.

        Two sources, in order:
          1. the target says so itself (`witness["best_known_source"]`);
          2. `record_registry` — the repo's sourced OEIS table — if the problem names one of its
             sequences (e.g. "A396704-n7").
        """
        declared = (rec.witness or {}).get("best_known_source")
        if isinstance(declared, dict) and declared:
            return {"kind": "target-declared", **declared}
        for oeis_id in record_registry.RECORDS:
            if oeis_id in problem:
                entry = record_registry.get_record(oeis_id)
                if entry:
                    return {
                        "kind": "record_registry",
                        "oeis_id": entry["oeis_id"],
                        "url": entry["url"],
                        "title": entry["title"],
                        "source_note": entry["source_note"],
                    }
        return {
            "kind": "UNSOURCED",
            "warning": (
                "The baseline this result claims to beat carries NO citation. Treat the claim as "
                "unverified: 'better than a number we made up' is not a result."
            ),
        }

    @staticmethod
    def _readme(rec: Record, provenance: dict[str, Any]) -> str:
        limitations = (rec.witness or {}).get("limitations") or rec.notes or ""
        limitations_block = (
            f"## READ THIS FIRST — limitations of this claim\n\n{limitations.strip()}\n\n"
            if limitations.strip()
            else ""
        )
        prov_lines = "\n".join(f"- **{k}**: {v}" for k, v in provenance.items())
        delta = rec.score - rec.best_known_at_time
        return f"""\
# {rec.problem} — verified result bundle

{limitations_block}\
## The claim

| | |
|---|---|
| problem | `{rec.problem}` |
| score | **{rec.score!r}** |
| baseline it beats (`best_known_at_time`) | {rec.best_known_at_time!r} |
| margin | {delta!r} |
| generating program | `{rec.program_id}` (see `program.py`) |
| verifier that judged it | `{rec.verifier_fingerprint}` (see `verifier.py`) |
| generation | {rec.generation} |
| recorded at (unix) | {rec.created_at!r} |
| exported with | Python {platform.python_version()} on {platform.system()} |

{rec.notes.strip()}

## Re-check it yourself — one command

From inside this directory:

```bash
python verify.py
```

Exit code `0` means PASS. Exit code `1` means the claim does not hold — in which case the claim is
wrong and we want to know. Nothing here imports `propab`; nothing here touches the network. Python
3.11+ standard library only.

`verify.py` performs five independent checks:

1. **Integrity** — every file's sha256 matches `MANIFEST.json` (nothing was edited after export).
2. **Verifier pin** — `sha256(verifier.py)` equals the `verifier_fingerprint` in `record.json`, so
   the code you are about to read is provably the code that judged the claim. Not a re-implementation
   of it, not a later version of it: *that* code.
3. **Validity** — `verifier.verify(candidate)` re-derives `valid == True` from `candidate.json`.
4. **Score** — the re-derived score equals the claimed `score` in `record.json`.
5. **Improvement** — the re-derived score strictly beats `best_known_at_time`.

Optionally, `python verify.py --regenerate` also re-runs `program.py` and checks that its `build()`
still emits exactly this candidate (provenance; not needed to check the claim itself).

## Do not trust us — read `verifier.py`

The whole claim rests on `verifier.py` being a faithful transcription of the problem's constraints.
It is short and deliberately dependency-free so that you can read it in a few minutes and decide for
yourself. If `verifier.py` is wrong, the result is wrong, and no amount of process on our side
changes that. This is the same posture as a machine-checked proof: credibility lives in the checker,
never in the claimant.

## Where the baseline comes from

{prov_lines}

## Files

- `verify.py` — the runnable checker (this is the whole product)
- `verifier.py` — the exact verifier source, sha256-pinned
- `candidate.json` — the construction
- `witness.json` — the verifier's evidence
- `record.json` — the full ledger record
- `program.py` — the generator that found it
- `MANIFEST.json` — sha256 of every file
"""


# --------------------------------------------------------------------------- #
# The runnable checker shipped inside every bundle. stdlib only; never imports propab.
# --------------------------------------------------------------------------- #
_VERIFY_RUNNER = '''\
#!/usr/bin/env python3
"""Independently re-check the result in this bundle.

    python verify.py                # check the claim         (exit 0 = PASS, 1 = FAIL)
    python verify.py --regenerate   # also re-run program.py and confirm it rebuilds the candidate

Standard library only. Imports nothing from propab. Touches no network. If this exits 0, the claim
in README.md holds *on your machine, checked by code you can read* — you never had to trust us.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORE_TOL = 1e-9

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    record = json.loads((HERE / "record.json").read_text(encoding="utf-8"))
    candidate = json.loads((HERE / "candidate.json").read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))

    print(f"problem : {record['problem']}")
    print(f"claim   : score = {record['score']!r} > best_known {record['best_known_at_time']!r}")
    print(f"verifier: {record['verifier_fingerprint']}")
    print()

    # 1. integrity — nothing was edited after export
    bad = [
        name
        for name, digest in manifest["files"].items()
        if not (HERE / name).exists() or sha256_file(HERE / name) != digest
    ]
    check("integrity: files match MANIFEST.json", not bad, f"tampered/missing: {bad}" if bad else "")

    # 2. the fingerprint pins the verifier we are about to run
    actual_fp = sha256_file(HERE / "verifier.py")
    check(
        "verifier pin: sha256(verifier.py) == record.verifier_fingerprint",
        actual_fp == record["verifier_fingerprint"],
        f"got {actual_fp}",
    )

    # 3./4. re-derive the verdict from the candidate, with the pinned verifier
    sys.path.insert(0, str(HERE))
    import verifier  # noqa: E402 — deliberately imported after the pin check above

    out = verifier.verify(candidate)
    check("validity: verifier.verify(candidate).valid is True", bool(out.get("valid")), str(out.get("detail")))

    score = float(out.get("score", float("-inf")))
    claimed = float(record["score"])
    check(
        "score: re-derived score == claimed score",
        abs(score - claimed) <= SCORE_TOL * max(1.0, abs(claimed)),
        f"re-derived {score!r} vs claimed {claimed!r}",
    )

    # 5. it is actually an improvement over the stated baseline
    baseline = float(record["best_known_at_time"])
    check("improvement: score > best_known_at_time", score > baseline, f"{score!r} vs {baseline!r}")

    # optional provenance check
    if "--regenerate" in argv:
        ns: dict[str, object] = {"__name__": "bundled_program"}
        exec(compile((HERE / "program.py").read_text(encoding="utf-8"), "program.py", "exec"), ns)
        build = ns.get("build")
        if not callable(build):
            check("provenance: program.py defines build()", False)
        else:
            def canon(obj):
                return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)

            emitted = build()
            # build() may return ONE candidate or a LIST of candidates -- and for some problems a
            # candidate is itself a list, so both readings must be tried.
            options = [emitted]
            if isinstance(emitted, list):
                options.extend(emitted)
            want = canon(candidate)
            check(
                "provenance: program.py rebuilds this candidate",
                any(canon(c) == want for c in options),
            )
    else:
        print("[SKIP] provenance: pass --regenerate to also re-run program.py")

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        print("The claim does NOT hold. Please tell us: this is exactly what the bundle is for.")
        return 1
    print("PASS - the claim re-checks. Verified by verifier.py, not by our say-so.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''


def open_ledger(root: str | Path = "artifacts/evolve") -> FileLedger:
    """The concrete Ledger. `Ledger` itself is an abstract contract (`ledger.py`, frozen)."""
    return FileLedger(root)
