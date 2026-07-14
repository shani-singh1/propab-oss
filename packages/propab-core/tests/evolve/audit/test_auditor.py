"""WS-6 — prove every kill fires, and prove a genuinely good result still PASSES.

An auditor that rejects everything is exactly as useless as one that accepts everything: the first
test here is the one that keeps the layer honest.

No network: the only socket in these tests is constructed and immediately closed, and the audit's
fresh process blocks sockets outright.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import audit_targets as T
import pytest

from propab.evolve.auditor import (
    Auditor,
    AuditError,
    AuditReport,
    BestKnownSource,
    CheckResult,
    verifier_fingerprint,
)
from propab.evolve.ledger import Record
from propab.evolve.program import Program

GOOD_PROGRAM = Program(code="def build():\n    return " + repr(T.HAMMING_7_4) + "\n")


@pytest.fixture
def auditor() -> Auditor:
    return Auditor(now=lambda: T.NOW + 1.0)   # freeze "now" so staleness is deterministic


def _claim(problem, candidate):
    """The engine's claim: whatever the verifier said about this candidate."""
    return problem.verify(candidate)


def _kill_names(report: AuditReport) -> set[str]:
    return {c.name for c in report.checks if c.is_kill}


def _assert_killed_by(report: AuditReport, prefix: str) -> str:
    assert not report.passed, f"expected a KILL, but the audit PASSED: {report.summary()}"
    hits = [c for c in report.checks if c.is_kill and c.name.startswith(prefix)]
    assert hits, f"expected a kill from {prefix!r}, got kills: {sorted(_kill_names(report))}"
    reason = hits[0].reason
    assert reason.strip(), f"{prefix} produced an empty kill reason — useless to a human"
    return reason


# --------------------------------------------------------------------------- #
# The one that matters: a genuinely good result must PASS.
# --------------------------------------------------------------------------- #
def test_genuinely_good_result_passes(auditor):
    problem = T.CodeProblem()                      # cited record d=2; we found the d=3 Hamming code
    verdict = _claim(problem, T.HAMMING_7_4)
    assert verdict.valid and verdict.score == 3.0

    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)

    assert report.passed, f"a real improvement was killed: {report.summary()}"
    assert report.kills == []
    assert report.verifier_fingerprint
    # It really did run the full battery, including the fresh-process re-check.
    names = {c.name for c in report.checks}
    for required in (
        "verifier_sanity",
        "claim_coherence",
        "witness_present",
        "score_provenance",
        "verifier_determinism",
        "degenerate_candidate",
        "class_membership",
        "baseline_realness",
        "trivial_rediscovery",
        "float_artifact",
        "reproduce_from_witness",
    ):
        assert required in names, f"{required} never ran"


# --------------------------------------------------------------------------- #
# KILL 1 — reproduce from the witness alone, in a fresh process.
# --------------------------------------------------------------------------- #
def test_witness_that_does_not_reproduce_is_killed(auditor):
    """The score lives in the verifier's in-memory memo cache, not in the candidate.

    Every in-process check agrees with every other: verify() returns 8.0 every single time,
    deterministically, and the verifier passes all of its own controls. Only the fresh process —
    which starts with an empty cache — sees that the candidate is actually worth 3.0.
    """
    problem = T.CachedScoreProblem()
    candidate = T.HAMMING_7_4_PERMUTED                    # a valid d=3 code, not one of the controls
    T.LIVE_CACHE[repr(candidate)] = 8.0                   # a stale number left in the cache
    try:
        verdict = _claim(problem, candidate)
        assert verdict.score == 8.0                       # the engine "found" an 8
        # In-process it is airtight — which is exactly why only a fresh process can catch it.
        assert problem.verify(candidate).score == 8.0
        assert problem.is_improvement(verdict)

        report = auditor.audit(problem, verdict, candidate, GOOD_PROGRAM)
        reason = _assert_killed_by(report, "reproduce_from_witness")
        assert "8.0" in reason and "3.0" in reason
        # Nothing else noticed. The controls passed, the score was deterministic, the maths was fine.
        assert _kill_names(report) == {"reproduce_from_witness"}
    finally:
        T.LIVE_CACHE.clear()


def test_verifier_that_touches_the_network_is_killed(auditor):
    """verify() must be offline. The fresh process blocks sockets, so an LLM cannot hide in there."""
    problem = T.NetworkVerifierProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "reproduce_from_witness")
    assert "network" in reason.lower()


# --------------------------------------------------------------------------- #
# KILL 2 — baseline realness. The failure that dissolved the last two "wins".
# --------------------------------------------------------------------------- #
def test_unsourced_best_known_is_killed(auditor):
    problem = T.NoProvenanceProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "baseline_realness")
    assert "best_known_provenance" in reason


def test_best_known_from_model_memory_is_killed(auditor):
    problem = T.ModelMemoryProvenanceProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "baseline_realness")
    assert "model_memory" in reason


def test_best_known_drifted_from_its_cited_source_is_killed(auditor):
    """The code compares against 2; the table it cites says 3. The delta is meaningless either way."""
    problem = T.DriftedProvenanceProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "baseline_realness[drift]")
    assert "codetables.de" in reason


def test_stale_baseline_warns_but_does_not_kill():
    auditor = Auditor(now=lambda: T.NOW + 10 * 365 * 24 * 3600.0)   # a decade later
    problem = T.CodeProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    assert report.passed
    assert any("stale" in w or "days" in w for w in report.warnings), report.warnings


# --------------------------------------------------------------------------- #
# KILL 3 — trivial rediscovery.
# --------------------------------------------------------------------------- #
def test_rediscovery_of_a_tabulated_construction_is_killed(auditor):
    problem = T.CodeProblem(known_constructions=[T.HAMMING_7_4])
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "trivial_rediscovery")
    assert "tabulated construction" in reason


def test_table_lookup_witness_is_killed(auditor):
    problem = T.TableLookupProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "trivial_rediscovery")
    assert "best_known_table" in reason


# --------------------------------------------------------------------------- #
# KILL 4 — class membership, re-checked independently of verify().
# --------------------------------------------------------------------------- #
def test_illegal_candidate_accepted_by_a_buggy_verifier_is_killed(auditor):
    """A rank-3 matrix scored as a legal [7,4] with d=3. The verifier's own controls pass."""
    problem = T.IllegalAcceptedProblem()
    verdict = _claim(problem, T.RANK_DEFICIENT_7_4)
    assert verdict.valid and verdict.score == 3.0        # it looks exactly like a real improvement
    assert problem.is_improvement(verdict)

    report = auditor.audit(problem, verdict, T.RANK_DEFICIENT_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "class_membership")
    assert "rank-deficient" in reason
    # And the controls did NOT catch it — only the second implementation did.
    assert "verifier_sanity" not in _kill_names(report)


def test_missing_independent_check_is_killed(auditor):
    problem = T.NoIndependentCheckProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "class_membership")
    assert "independent_check" in reason


def test_vacuous_independent_check_is_killed(auditor):
    """Who checks the checker? An independent_check that has never rejected anything is not a check.

    The candidate here is GENUINELY GOOD — the audit would otherwise pass. It is killed purely
    because the layer protecting us was stubbed out, and a stub that always says yes is exactly as
    dangerous as no check at all while looking exactly like a passing one.
    """
    problem = T.VacuousIndependentCheckProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    assert verdict.valid and verdict.score == 3.0        # a real d=3 code: nothing wrong with it

    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "class_membership")
    assert "vacuous" in reason
    assert "rank_deficient" in reason                    # it approved a known-illegal generator


# --------------------------------------------------------------------------- #
# KILL 5 — verifier sanity. A broken verifier invalidates EVERYTHING.
# --------------------------------------------------------------------------- #
def test_broken_verifier_kills_everything(auditor):
    problem = T.BrokenVerifierProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    assert verdict.score == 4.0                          # it cannot score the known Hamming code

    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "verifier_sanity[hamming_7_4_known_d3]")
    assert "ENGINE IS BROKEN" in reason
    # Short-circuited: no downstream check's evidence is worth anything now.
    assert "audit_short_circuit" in _kill_names(report)
    assert "reproduce_from_witness" not in {c.name for c in report.checks}


def test_verifier_that_raises_on_garbage_is_killed(auditor):
    problem = T.RaisesOnGarbageProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "verifier_sanity[garbage:none]")
    assert "RAISED" in reason


def test_missing_controls_is_killed(auditor):
    problem = T.NoControlsProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "verifier_sanity")
    assert "controls()" in reason


# --------------------------------------------------------------------------- #
# KILL 6 — numerical / floating-point artifact.
# --------------------------------------------------------------------------- #
def test_float_noise_improvement_is_killed(auditor):
    """1/10 + 2/10 == 0.30000000000000004 in floats, which "beats" a baseline of 0.3."""
    problem = T.RationalSumProblem()
    candidate = [[1, 10], [2, 10]]
    verdict = _claim(problem, candidate)
    assert verdict.score > 0.3 and problem.is_improvement(verdict)   # the engine calls this a win

    report = auditor.audit(problem, verdict, candidate, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "float_artifact")
    assert "floating-point tolerance" in reason


def test_margin_that_vanishes_under_exact_arithmetic_is_killed(auditor):
    """A ~3e-7 margin: too big to be float noise, and still not real. Only exact arithmetic knows."""
    problem = T.ApproxSolverProblem()
    candidate = [[1, 10], [2, 10]]
    verdict = _claim(problem, candidate)
    assert verdict.score - 0.3 > 1e-8                  # clears the float-noise floor comfortably

    report = auditor.audit(problem, verdict, candidate, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "float_artifact[exact]")
    assert "EXACT arithmetic" in reason
    assert "3/10" in reason
    # The plain float check was fooled — this is exactly what exact re-checking is for.
    assert "float_artifact" not in _kill_names(report)


def test_improvement_inside_the_verifiers_own_tolerance_is_killed(auditor):
    problem = T.ToleranceProblem()
    candidate = [[1, 10], [2, 10]]
    verdict = _claim(problem, candidate)
    report = auditor.audit(problem, verdict, candidate, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "float_artifact[tolerance:tol]")
    assert "slop" in reason


# --------------------------------------------------------------------------- #
# KILL 7 — degenerate candidate (the vacuous-truth hole).
# --------------------------------------------------------------------------- #
def test_degenerate_all_zero_candidate_is_killed(auditor):
    """The all-zero code has no nonzero codewords, so "min weight of a nonzero codeword" is a min
    over the empty set. The verifier reports the best possible score, d=7, and every constraint is
    'satisfied'. The check passed for the wrong reason."""
    problem = T.DegenerateAcceptedProblem()
    verdict = _claim(problem, T.ALL_ZERO_7_4)
    assert verdict.valid and verdict.score == 7.0        # a "perfect" score for an empty object
    assert problem.is_improvement(verdict)

    report = auditor.audit(problem, verdict, T.ALL_ZERO_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "degenerate_candidate")
    assert "all-zero" in reason
    # Belt and braces: the independent class check catches it too.
    assert "class_membership" in _kill_names(report)


# --------------------------------------------------------------------------- #
# KILL 8 — score provenance. Never trust the candidate's self-report.
# --------------------------------------------------------------------------- #
def test_verifier_that_echoes_the_programs_self_reported_score_is_killed(auditor):
    """A mutated program simply writes down a world record, and the verifier believes it."""
    problem = T.EchoScoreProblem()
    candidate = {"matrix": T.HAMMING_7_4, "min_distance": 9}   # the program claims d=9
    verdict = _claim(problem, candidate)
    assert verdict.score == 9.0 and problem.is_improvement(verdict)

    report = auditor.audit(problem, verdict, candidate, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "score_provenance[self_report:min_distance]")
    assert "ECHOES" in reason and "grading its own homework" in reason


def test_score_not_produced_by_the_verifier_is_killed(auditor):
    """A verdict whose number the verifier does not reproduce did not come from the verifier."""
    problem = T.CodeProblem()
    real = _claim(problem, T.HAMMING_7_4)
    forged = type(real)(valid=True, score=99.0, detail=dict(real.detail))

    report = auditor.audit(problem, forged, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "score_provenance")
    assert "NOT produced by the verifier" in reason


def test_nondeterministic_verifier_is_killed(auditor):
    problem = T.NondeterministicProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "verifier_determinism")
    assert "NONDETERMINISTIC" in reason


# --------------------------------------------------------------------------- #
# Coherence / hygiene kills
# --------------------------------------------------------------------------- #
def test_is_improvement_that_does_not_beat_best_known_is_killed(auditor):
    problem = T.IncoherentImprovementProblem(best=5.0)     # d=3 does not beat a record of 5
    verdict = _claim(problem, T.HAMMING_7_4)
    assert problem.is_improvement(verdict)                 # …but the target says it does
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "claim_coherence")
    assert "incoherent" in reason


def test_missing_witness_is_killed(auditor):
    problem = T.NoWitnessProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "witness_present")
    assert "no witness" in reason


def test_unserializable_candidate_is_killed(auditor):
    problem = T.UnserializableProblem()
    candidate = {"matrix": T.HAMMING_7_4, "handle": object()}
    verdict = _claim(problem, candidate)
    assert verdict.valid
    report = auditor.audit(problem, verdict, candidate, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "candidate_serializable")
    assert "not JSON-serializable" in reason


def test_unseeded_program_randomness_warns_but_does_not_kill(auditor):
    problem = T.CodeProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    program = Program(code="import random\ndef build():\n    return random.choice(CODES)\n")
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, program)
    assert report.passed
    assert any("seed" in w for w in report.warnings), report.warnings


# --------------------------------------------------------------------------- #
# Per-target extension
# --------------------------------------------------------------------------- #
def test_target_specific_check_can_kill(auditor):
    problem = T.ExtraKillProblem()
    verdict = _claim(problem, T.HAMMING_7_4_PERMUTED)     # valid, d=3, but not systematic form
    assert verdict.valid and verdict.score == 3.0

    report = auditor.audit(problem, verdict, T.HAMMING_7_4_PERMUTED, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "target[systematic_form]")
    assert "systematic form" in reason
    # The same target passes on a systematic candidate — the hook is a real discriminator.
    ok = auditor.audit(problem, _claim(problem, T.HAMMING_7_4), T.HAMMING_7_4, GOOD_PROGRAM)
    assert ok.passed, ok.summary()


def test_target_check_that_raises_is_a_kill_not_a_pass(auditor):
    class Exploding(T.CodeProblem):
        def audit_checks(self):
            def boom(ctx):
                raise RuntimeError("check blew up")

            return [boom]

    problem = Exploding()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    reason = _assert_killed_by(report, "target[boom]")
    assert "kill, not a pass" in reason


# --------------------------------------------------------------------------- #
# The report itself
# --------------------------------------------------------------------------- #
def test_report_is_serializable_and_attaches_to_a_record(auditor):
    problem = T.CodeProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    assert report.passed

    blob = report.to_json()
    parsed = json.loads(blob)
    assert parsed["passed"] is True
    assert parsed["kills"] == []
    assert parsed["problem"] == "code[7,4]"
    assert any(c["name"] == "reproduce_from_witness" for c in parsed["checks"])

    record = Record(
        problem=problem.name,
        score=verdict.score,
        best_known_at_time=problem.best_known(),
        candidate=T.HAMMING_7_4,
        witness=dict(verdict.detail),
        program_code=GOOD_PROGRAM.code,
        program_id=GOOD_PROGRAM.id,
        verifier_fingerprint=verifier_fingerprint(problem),
    )
    report.attach(record)

    # It must survive Record.to_json() — that is the only thing a publisher actually exports.
    exported = json.loads(record.to_json())
    carried = json.loads(exported["notes"])["audit"]
    assert carried["passed"] is True
    assert carried["verifier_fingerprint"] == record.verifier_fingerprint
    assert record.audit_report is report


def test_audit_or_raise(auditor):
    problem = T.NoProvenanceProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    with pytest.raises(AuditError, match="baseline_realness"):
        auditor.audit_or_raise(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)


def test_kill_reasons_are_specific_and_actionable(auditor):
    """"Failed audit" is useless. Every kill must name the thing and say what to do about it."""
    problem = T.NoProvenanceProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    assert not report.passed
    for kill in report.kills:
        assert len(kill) > 40, f"kill reason is too vague to act on: {kill!r}"
        assert ":" in kill                       # "<check_name>: <why>"


def test_verifier_fingerprint_is_stable_and_discriminating():
    a = verifier_fingerprint(T.CodeProblem())
    b = verifier_fingerprint(T.CodeProblem(n=7, k=4, best=99.0))
    c = verifier_fingerprint(T.BrokenVerifierProblem())
    assert a == b          # same verifier source => same fingerprint (params are not the verifier)
    assert a != c          # a different verify() => a different fingerprint


def test_verifier_fingerprint_covers_declared_helper_modules():
    """A thin adapter's verify() delegates to a domain module. If the fingerprint does not cover that
    module, the real checking logic can change while the fingerprint sits still."""

    class Declaring(T.CodeProblem):
        def verifier_sources(self):
            return ["propab.evolve.problem"]

    plain = verifier_fingerprint(T.CodeProblem())
    declaring = verifier_fingerprint(Declaring())
    assert plain != declaring, "declared verifier sources are not being hashed"


def test_pickle_reconstruction_warns_that_instance_state_was_carried(auditor):
    """The process is fresh; the object is not. Pickle drags `self` across, so an instance-level memo
    cache would survive the trip and reproduce its own wrong answer. Say so."""
    problem = T.CodeProblem(known_constructions=[[[1, 1, 1, 1, 1, 1, 1]]])   # non-empty list on self
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)

    assert report.passed
    carried = [c for c in report.checks if c.name == "reproduce_from_witness[state_carried]"]
    assert carried, [c.name for c in report.checks]
    assert "audit_spec()" in carried[0].reason
    assert any("known_constructions" in s for s in carried[0].detail["carried"])


def test_best_known_source_rejects_junk_attribution():
    assert BestKnownSource(3.0, "Grassl table", "https://codetables.de/", T.NOW).problems() == []
    assert BestKnownSource(3.0, "", "", 0.0, kind="guess").problems()
    assert BestKnownSource(float("nan"), "s", "c", T.NOW).problems()


def test_check_result_kill_semantics():
    assert CheckResult("x", passed=False).is_kill
    assert not CheckResult("x", passed=False, severity="warn").is_kill
    assert not CheckResult("x", passed=True).is_kill


def test_exact_score_is_exact_not_float():
    problem = T.RationalSumProblem()
    assert problem.exact_score([[1, 10], [2, 10]]) == Fraction(3, 10)
    assert problem.verify([[1, 10], [2, 10]]).score != 0.3   # the float disagrees; the Fraction does not


def test_child_sys_path_has_no_relative_entries():
    """A relative sys.path entry (or "", i.e. the cwd) is meaningless in the re-check process, which
    runs in a scratch dir. Left unresolved it would fail to import the verifier and kill a perfectly
    good discovery for the wrong reason."""
    from propab.evolve.auditor import _child_sys_path

    entries = _child_sys_path()
    assert entries
    assert all(entry for entry in entries), "empty sys.path entry leaked to the child"
    assert all(Path(entry).is_absolute() for entry in entries), entries


def test_every_kill_check_reports_even_when_it_passes(auditor):
    """A check that silently emits nothing on success is indistinguishable from a check that never
    ran — which is how a check comes to 'pass' for the wrong reason. Every one must speak up."""
    problem = T.CodeProblem()
    verdict = _claim(problem, T.HAMMING_7_4)
    report = auditor.audit(problem, verdict, T.HAMMING_7_4, GOOD_PROGRAM)
    assert report.passed

    # Every check family in the kill-list must appear in a PASSING report, not just a failing one.
    families = {c.name.split("[")[0] for c in report.checks}
    assert families >= {
        "verifier_sanity",
        "claim_coherence",
        "witness_present",
        "candidate_serializable",
        "score_provenance",
        "verifier_determinism",
        "degenerate_candidate",
        "class_membership",
        "baseline_realness",
        "trivial_rediscovery",
        "float_artifact",
        "reproduce_from_witness",
        "program_hygiene",
    }, sorted(families)
