"""The Ledger — the store, and the publishable bundle.

The bundle is the product. The load-bearing assertion in this file is
`test_exported_bundle_rechecks_in_one_command`: a third party, with nothing but the bundle
directory, runs `python verify.py` and gets exit 0. If that ever fails, we have no result.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from propab.evolve.ledger import Record
from propab.evolve.ledger_impl import FileLedger, content_hash, verifier_fingerprint
from propab.evolve.program import Program
from propab.evolve.targets.erdos143 import HONESTY_STATEMENT, Erdos143Problem

# A toy standalone verifier: valid iff the candidate is a list of positive ints; score = sum.
TOY_VERIFIER = '''\
VERIFIER_ID = "toy/1"


def verify(candidate):
    try:
        if not isinstance(candidate, list) or not candidate:
            return {"valid": False, "score": float("-inf"), "detail": {"reason": "not a list"}}
        for v in candidate:
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                return {"valid": False, "score": float("-inf"), "detail": {"reason": "bad element"}}
        return {"valid": True, "score": float(sum(candidate)), "detail": {"witness": {"n": len(candidate)}}}
    except Exception as exc:
        return {"valid": False, "score": float("-inf"), "detail": {"reason": str(exc)}}
'''

TOY_PROGRAM = "def build():\n    return [10, 20, 30]\n"


#: A Record only reaches the ledger if the adversarial layer cleared it. These tests are about the
#: ledger's OWN checks, so they hand it a record that has already passed audit; the two tests below
#: cover the audit gate itself.
PASSING_AUDIT = {"passed": True, "kills": [], "warnings": [], "checks": []}


def toy_record(**over: object) -> Record:
    base = dict(
        problem="toy",
        score=60.0,
        best_known_at_time=50.0,
        candidate=[10, 20, 30],
        witness={"verifier_source": TOY_VERIFIER, "witness": {"n": 3}},
        program_code=TOY_PROGRAM,
        program_id="toyprog",
        verifier_fingerprint=verifier_fingerprint(TOY_VERIFIER),
        audit=dict(PASSING_AUDIT),
    )
    base.update(over)
    return Record(**base)  # type: ignore[arg-type]


def test_a_record_with_no_audit_is_refused(tmp_path):
    """The auditor is mandatory. An unenforced check is indistinguishable from a passed one — which
    is precisely how our earlier 'wins' survived long enough to be reported."""
    ledger = FileLedger(root=tmp_path)
    assert ledger.record(toy_record(audit=None)) is False
    assert ledger.best("toy") is None


def test_a_record_whose_audit_failed_is_refused(tmp_path):
    ledger = FileLedger(root=tmp_path)
    killed = {"passed": False, "kills": ["baseline_realness: best_known() has no source"]}
    assert ledger.record(toy_record(audit=killed)) is False
    assert ledger.best("toy") is None


@pytest.fixture()
def ledger(tmp_path: Path) -> FileLedger:
    return FileLedger(tmp_path / "evolve")


# --------------------------------------------------------------------------- #
# record / best
# --------------------------------------------------------------------------- #
def test_record_persists_and_best_returns_it(ledger: FileLedger) -> None:
    assert ledger.record(toy_record()) is True
    best = ledger.best("toy")
    assert best is not None
    assert best.score == 60.0
    assert best.candidate == [10, 20, 30]
    # the verifier source is moved to the content-addressed blob store, not kept inline
    assert "verifier_source" not in best.witness
    assert ledger.verifier_source(best.verifier_fingerprint) == TOY_VERIFIER


def test_best_of_an_unknown_problem_is_none(ledger: FileLedger) -> None:
    assert ledger.best("nothing-here") is None


def test_best_tracks_the_highest_score(ledger: FileLedger) -> None:
    assert ledger.record(toy_record()) is True
    assert ledger.record(toy_record(candidate=[10, 20, 40], score=70.0)) is True
    assert ledger.record(toy_record(candidate=[1, 2, 3], score=6.0, best_known_at_time=5.0)) is True
    assert ledger.best("toy").score == 70.0
    assert len(ledger.records("toy")) == 3  # append-only: nothing is overwritten


def test_store_is_append_only(ledger: FileLedger) -> None:
    ledger.record(toy_record())
    ledger.record(toy_record(candidate=[10, 20, 40], score=70.0))
    lines = (ledger.root / "toy" / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["record"]["score"] == 60.0  # the first line never changed


# --------------------------------------------------------------------------- #
# The refusals. Every one of these is the ledger catching a claim that would not survive scrutiny.
# --------------------------------------------------------------------------- #
def test_duplicate_candidate_is_refused(ledger: FileLedger) -> None:
    assert ledger.record(toy_record()) is True
    assert ledger.record(toy_record()) is False  # same candidate, same problem
    assert ledger.record(toy_record(score=61.0, program_id="other")) is False  # still the same object
    assert len(ledger.records("toy")) == 1
    assert ledger.rejections("toy")[0]["reason"] == "duplicate_candidate"


def test_same_candidate_under_a_different_problem_is_not_a_duplicate(ledger: FileLedger) -> None:
    assert ledger.record(toy_record()) is True
    assert ledger.record(toy_record(problem="toy2")) is True


def test_fingerprint_that_does_not_pin_the_verifier_is_refused(ledger: FileLedger) -> None:
    """A fingerprint that does not hash the shipped source is worthless — it pins nothing."""
    rec = toy_record(verifier_fingerprint="sha256:" + "0" * 64)
    assert ledger.record(rec) is False
    assert ledger.rejections("toy")[0]["reason"] == "verifier_fingerprint_mismatch"


def test_record_without_a_verifier_is_refused(ledger: FileLedger) -> None:
    """No verifier source => nobody can re-check it => it is not a result."""
    assert ledger.record(toy_record(witness={})) is False
    assert ledger.rejections("toy")[0]["reason"] == "no_verifier_source"


def test_ledger_reruns_the_verifier_and_refuses_a_false_claim(ledger: FileLedger) -> None:
    """The caller's verdict is not taken on trust: the toy verifier rejects negative elements."""
    assert ledger.record(toy_record(candidate=[10, -20, 30], score=60.0)) is False
    assert ledger.rejections("toy")[0]["reason"] == "recheck_invalid"


def test_a_lied_about_score_is_refused(ledger: FileLedger) -> None:
    assert ledger.record(toy_record(score=999.0)) is False
    rej = ledger.rejections("toy")[0]
    assert rej["reason"] == "recheck_score_mismatch"
    assert rej["observed_score"] == 60.0


def test_a_non_improvement_is_refused(ledger: FileLedger) -> None:
    assert ledger.record(toy_record(best_known_at_time=60.0)) is False  # ties are not wins
    assert ledger.record(toy_record(best_known_at_time=99.0)) is False
    assert ledger.rejections("toy")[0]["reason"] == "not_an_improvement"


def test_a_crashing_verifier_certifies_nothing(ledger: FileLedger) -> None:
    src = "def verify(candidate):\n    raise RuntimeError('boom')\n"
    rec = toy_record(
        witness={"verifier_source": src}, verifier_fingerprint=verifier_fingerprint(src)
    )
    assert ledger.record(rec) is False
    assert ledger.rejections("toy")[0]["reason"] == "verifier_crashed"


def test_record_never_raises_and_always_explains_itself(ledger: FileLedger) -> None:
    for rec in (
        toy_record(witness={}),
        toy_record(score=1e9),
        toy_record(candidate="not a list", score=1.0),
        toy_record(verifier_fingerprint=""),
    ):
        assert ledger.record(rec) is False
    reasons = [r["reason"] for r in ledger.rejections("toy")]
    assert len(reasons) == 4
    assert all(r for r in reasons)


# --------------------------------------------------------------------------- #
# The bundle — this is the product
# --------------------------------------------------------------------------- #
def _run_verify(bundle: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "verify.py", *args],
        cwd=bundle,
        capture_output=True,
        text=True,
        check=False,
    )


def test_exported_bundle_rechecks_in_one_command(ledger: FileLedger, tmp_path: Path) -> None:
    """THE test. Nothing but the bundle directory, one command, exit 0."""
    assert ledger.record(toy_record()) is True
    bundle = ledger.export_publishable("toy", tmp_path / "bundle")

    for name in (
        "README.md", "verify.py", "verifier.py", "candidate.json",
        "witness.json", "record.json", "program.py", "MANIFEST.json",
    ):
        assert (bundle / name).exists(), name

    proc = _run_verify(bundle)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout
    assert "FAILED" not in proc.stdout

    # the bundle is self-contained: no propab import statement anywhere in it
    # (prose mentioning propab is fine; an `import propab` is not)
    for name in ("verify.py", "verifier.py"):
        src = (bundle / name).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(?:import|from)\s+propab\b", src, re.MULTILINE), name

    # and the shipped verifier is byte-for-byte the one that judged the claim
    assert (bundle / "verifier.py").read_text(encoding="utf-8") == TOY_VERIFIER
    rec = json.loads((bundle / "record.json").read_text(encoding="utf-8"))
    assert verifier_fingerprint((bundle / "verifier.py").read_text(encoding="utf-8")) == rec["verifier_fingerprint"]


def test_the_pin_is_over_exact_bytes_on_disk(ledger: FileLedger, tmp_path: Path) -> None:
    """Regression: `write_text` newline-translates on Windows, which silently breaks the sha256 pin.

    The fingerprint is only worth something if it hashes the bytes a third party actually reads.
    """
    ledger.record(toy_record())
    bundle = ledger.export_publishable("toy", tmp_path / "bundle")
    raw = (bundle / "verifier.py").read_bytes()
    assert b"\r\n" not in raw
    assert "sha256:" + hashlib.sha256(raw).hexdigest() == verifier_fingerprint(TOY_VERIFIER)


def test_bundle_regenerate_flag_reruns_the_program(ledger: FileLedger, tmp_path: Path) -> None:
    ledger.record(toy_record())
    bundle = ledger.export_publishable("toy", tmp_path / "bundle")
    proc = _run_verify(bundle, "--regenerate")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "program.py rebuilds this candidate" in proc.stdout


def test_a_tampered_candidate_fails_the_recheck(ledger: FileLedger, tmp_path: Path) -> None:
    """Someone edits the construction after export: verify.py must catch it and exit 1."""
    ledger.record(toy_record())
    bundle = ledger.export_publishable("toy", tmp_path / "bundle")
    (bundle / "candidate.json").write_text("[10, 20, 999]")
    proc = _run_verify(bundle)
    assert proc.returncode == 1
    assert "FAILED" in proc.stdout
    assert "integrity" in proc.stdout


def test_a_swapped_verifier_fails_the_pin(ledger: FileLedger, tmp_path: Path) -> None:
    """Someone swaps in a permissive verifier: the sha256 pin no longer matches. Exit 1."""
    ledger.record(toy_record())
    bundle = ledger.export_publishable("toy", tmp_path / "bundle")
    (bundle / "verifier.py").write_text(
        'VERIFIER_ID = "cheat"\n\n\ndef verify(candidate):\n'
        '    return {"valid": True, "score": 60.0, "detail": {}}\n'
    )
    proc = _run_verify(bundle)
    assert proc.returncode == 1
    assert "verifier pin" in proc.stdout


def test_export_refuses_when_there_is_nothing_to_publish(ledger: FileLedger, tmp_path: Path) -> None:
    with pytest.raises(LookupError):
        ledger.export_publishable("toy", tmp_path / "bundle")


def test_export_refuses_a_result_it_cannot_recheck(ledger: FileLedger, tmp_path: Path) -> None:
    """If the verifier blob is gone, the claim is unverifiable — so it is not publishable."""
    ledger.record(toy_record())
    for blob in (ledger.root / "verifiers").glob("*.py"):
        blob.unlink()
    with pytest.raises(RuntimeError, match="cannot be independently re-checked"):
        ledger.export_publishable("toy", tmp_path / "bundle")


def test_readme_states_the_claim_the_command_and_the_provenance(
    ledger: FileLedger, tmp_path: Path
) -> None:
    ledger.record(toy_record())
    bundle = ledger.export_publishable("toy", tmp_path / "bundle")
    readme = (bundle / "README.md").read_text(encoding="utf-8")
    assert "python verify.py" in readme
    assert "Do not trust us" in readme
    # an unsourced baseline is stated as unsourced, loudly
    assert "UNSOURCED" in readme


def test_content_hash_is_stable_and_problem_scoped() -> None:
    assert content_hash("p", [1, 2]) == content_hash("p", [1, 2])
    assert content_hash("p", [1, 2]) != content_hash("q", [1, 2])
    assert content_hash("p", [1, 2]) != content_hash("p", [2, 1])


# --------------------------------------------------------------------------- #
# End to end: the real Erdős #143 verifier through the real ledger into a real bundle
# --------------------------------------------------------------------------- #
def test_erdos143_record_exports_a_bundle_that_rechecks(ledger: FileLedger, tmp_path: Path) -> None:
    """The whole pipeline with the real target.

    NOTE the baseline used here. The primes trivially beat the (N/2, N] construction, so this is a
    BUNDLE-FORMAT fixture, not a research claim — and the target itself says so: `is_improvement()`
    correctly refuses to call it a discovery (it is a known construction). The ledger stores what it
    is told and re-checks it; the target decides what counts as a find. Different jobs.
    """
    p = Erdos143Problem(n_max=200)
    ns: dict = {"__name__": "seed"}
    exec(compile(p.seed_programs()[0], "<primes>", "exec"), ns)  # the primes seed
    cand = ns["build"]()

    verdict = p.verify(cand)
    assert verdict.valid
    assert p.is_improvement(verdict) is False  # rediscovery — the target is not fooled

    half_interval_baseline = p.verify(list(range(101, 201))).score
    rec = p.make_record(
        cand,
        verdict,
        Program(code=p.seed_programs()[0]),
        best_known=half_interval_baseline,
        notes="TEST FIXTURE: baseline is the weak (N/2, N] construction, not the primes baseline.",
    )
    rec.audit = dict(PASSING_AUDIT)  # the adversarial layer is mandatory; stand in for it here
    assert ledger.record(rec) is True

    bundle = ledger.export_publishable(p.name, tmp_path / "e143")
    proc = _run_verify(bundle, "--regenerate")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    readme = (bundle / "README.md").read_text(encoding="utf-8")
    # the caveat travels with the claim, at the very top, by construction
    assert "READ THIS FIRST" in readme
    assert "NOT A SOLUTION TO ERD" in readme.upper()
    assert "no finite search can be one" in readme.lower()
    assert HONESTY_STATEMENT.strip().splitlines()[0] in readme
    # and the baseline is cited, not asserted
    assert "Lichtman" in readme

    witness = json.loads((bundle / "witness.json").read_text(encoding="utf-8"))
    assert witness["limitations"].strip().startswith("THIS IS NOT A SOLUTION")
    assert "Annals of Mathematics" in witness["best_known_source"]["citation"]


def test_erdos143_bundle_verifier_is_standalone(ledger: FileLedger, tmp_path: Path) -> None:
    """Run the exported #143 verifier in a bare subprocess: stdlib only, no propab on sys.path."""
    p = Erdos143Problem(n_max=100)
    ns: dict = {"__name__": "seed"}
    exec(compile(p.seed_programs()[0], "<primes>", "exec"), ns)
    cand = ns["build"]()
    rec = p.make_record(cand, p.verify(cand), Program(code=p.seed_programs()[0]), best_known=0.0)
    rec.audit = dict(PASSING_AUDIT)  # the adversarial layer is mandatory; stand in for it here
    assert ledger.record(rec) is True
    bundle = ledger.export_publishable(p.name, tmp_path / "e143")

    proc = subprocess.run(
        [sys.executable, "-S", "-c",
         "import json,sys;sys.path.insert(0,'.');import verifier;"
         "c=json.load(open('candidate.json'));"
         "print(verifier.verify(c)['valid'])"],
        cwd=bundle,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True"
