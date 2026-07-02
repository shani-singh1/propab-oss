"""Tests for scoped claims (fixes.md Step 2-3)."""
from __future__ import annotations

from propab.scoped_claim import (
    ScopedClaim,
    apply_ood_gate_to_verdict,
    check_ood_evidence,
    enrich_entry_with_scope,
    infer_domain_scope_template,
    parse_scope_from_text,
    passes_scope_gate,
    validate_scoped_claim,
)


def test_validate_scoped_claim_complete():
    s = ScopedClaim(
        text="k-shell predicts spread",
        population="BA graphs N=500",
        distribution="Barabási–Albert m=4",
        claimed_generalization="Transfer to WS graphs",
        expected_failure_modes="Fails on ER",
        ood_test="LOFO on WS family",
    )
    ok, missing = validate_scoped_claim(s)
    assert ok and missing == []


def test_parse_scope_from_text():
    text = (
        "k-shell predicts velocity.\n"
        "Population: BA and SBM graphs\n"
        "Distribution: avg degree 8\n"
        "Claimed generalization: WS transfer\n"
        "Expected failure modes: ER collapse\n"
        "OOD test: hold out WS"
    )
    s = parse_scope_from_text(text)
    assert s is not None
    assert "WS" in s.claimed_generalization


def test_enrich_entry_fills_contagion_template():
    entry = enrich_entry_with_scope(
        {"text": "Hub removal slows SIS outbreaks.", "id": "h1"},
        "Investigate contagion spreading on complex networks.",
    )
    assert passes_scope_gate(entry, question="contagion networks")
    assert "Population:" in entry["text"]
    assert entry.get("claim_scope")


def test_ood_gate_blocks_confirm_without_evidence():
    v, r = apply_ood_gate_to_verdict(
        "confirmed",
        "significance ok",
        {"n_metric_steps": 2, "p_value": 0.01},
        hypothesis_text=infer_domain_scope_template("contagion").formatted_text(),
        test_methodology=infer_domain_scope_template("contagion").methodology_json(),
    )
    assert v == "inconclusive"
    assert "OOD" in r


def test_scope_integrity_pass_on_matching_lofo():
    from propab.scoped_claim import (
        ExecutedOOD,
        ScopedClaim,
        check_scope_executed_integrity,
    )
    scope = ScopedClaim(
        text="thermal",
        population="56 RT sequences",
        distribution="7 rt_family groups",
        claimed_generalization="cross-family",
        expected_failure_modes="geometry collapse",
        ood_test="LOFO on held-out family; label-shuffle permutation required",
    )
    executed = ExecutedOOD(
        train_contexts=["A", "B"],
        held_out_contexts=["C"],
        metric_used="lofo_r2",
        evaluation_code_hash="abc",
        summary="LOFO train=['A','B'] hold_out=['C'] label-shuffle permutation",
    )
    r = check_scope_executed_integrity(scope, executed)
    assert r.scope_gate_result == "PASS"


def test_materials_scope_boilerplate_passes_when_lofo_executed():
    from propab.scoped_claim import ExecutedOOD, check_scope_executed_integrity, infer_domain_scope_template

    q = "[domain_profile:materials] matbench dielectric crystal system LOFO"
    scope = infer_domain_scope_template(q)
    executed = ExecutedOOD(
        train_contexts=["cubic", "tetragonal"],
        held_out_contexts=["orthorhombic"],
        metric_used="lofo_r2",
        evaluation_code_hash="x",
        summary="LOFO train=['cubic','tetragonal'] hold_out=['orthorhombic'] label-shuffle permutation",
    )
    r = check_scope_executed_integrity(scope, executed, question=q)
    assert r.scope_gate_result == "PASS"


def test_scope_integrity_fail_on_mismatch():
    from propab.scoped_claim import ExecutedOOD, ScopedClaim, check_scope_executed_integrity
    scope = ScopedClaim(
        text="x",
        population="BA graphs",
        distribution="SBM",
        claimed_generalization="WS transfer",
        expected_failure_modes="ER fail",
        ood_test="Hold out Watts-Strogatz family; evaluate LOFO R² on WS",
    )
    executed = ExecutedOOD(
        train_contexts=["BA"],
        held_out_contexts=["BA"],
        metric_used="p_value",
        evaluation_code_hash="x",
        summary="single sandbox p=0.001 on same BA ensemble",
    )
    r = check_scope_executed_integrity(scope, executed)
    assert r.scope_gate_result == "FAIL"
    assert r.audit_class == "mismatched_scope"
    ok, reason = check_ood_evidence(
        {"ood_passed": True, "lofo_r2": 0.08},
        infer_domain_scope_template("mandrake"),
    )
    assert ok
    scope = infer_domain_scope_template("mandrake")
    v, r = apply_ood_gate_to_verdict(
        "confirmed",
        "LOFO ok",
        {"ood_passed": True, "lofo_r2": 0.08, "label_shuffle_permutation_p": 0.02},
        hypothesis_text=scope.formatted_text(),
        test_methodology=scope.methodology_json(),
    )
    assert v == "confirmed"
