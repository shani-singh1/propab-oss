"""Dry-run routing inspector for math_combinatorics hypotheses (fixes.md)."""
from __future__ import annotations

from typing import Any

from propab.domain_modules.math_combinatorics.constructors import (
    evaluate_experiment_claim,
    extract_claim_text,
    is_cap_set_hypothesis,
    is_sidon_hypothesis,
    parse_interval_claim,
    parse_minimum_int_claim,
    parse_ratio_upper_bound,
    requires_unimplemented_statistics,
)
from propab.domain_modules.math_combinatorics.verifier import (
    _extract_cap_dims,
    _extract_n_list,
    _wants_asymptotic_analysis,
    _wants_bose_chowla,
    _wants_greedy_vs_algebraic,
    run_combinatorics_experiment,
)


def _infer_features(hypothesis: dict[str, Any], claim: str) -> list[str]:
    text = str(hypothesis.get("text") or hypothesis.get("statement") or "")
    methodology = str(hypothesis.get("test_methodology") or "")
    if is_cap_set_hypothesis(text, methodology, full_text=text):
        return ["cap_set_size"]
    if "sumset" in claim.lower():
        return ["sumset_growth"]
    if "ap-free" in claim.lower() or "arithmetic progression" in claim.lower():
        return ["arithmetic_progression_free_density"]
    return ["sidon_set_density"]


def _resolve_verifier(claim: str, methodology: str, full_text: str) -> str:
    if is_cap_set_hypothesis(full_text, methodology, full_text=full_text):
        return "cap_set_sweep"
    if is_sidon_hypothesis(full_text, methodology, full_text=full_text):
        if _wants_greedy_vs_algebraic(claim):
            return "sidon_bc_vs_greedy_compare"
        if _wants_bose_chowla(claim) and not _wants_greedy_vs_algebraic(claim):
            return "sidon_sweep_bose_chowla"
        if _wants_asymptotic_analysis(claim) or len(_extract_n_list(claim)) >= 2:
            return "sidon_sweep_greedy"
        return "sidon_single_point"
    if "sumset" in claim.lower():
        return "sumset"
    if "ap-free" in claim.lower():
        return "ap_free"
    return "sidon_sweep_greedy"


def _claim_types(claim: str) -> list[str]:
    types: list[str] = []
    if parse_interval_claim(claim):
        types.append("ratio_band")
    if parse_ratio_upper_bound(claim):
        types.append("ratio_upper_bound")
    if parse_minimum_int_claim(claim):
        types.append("minimum_size")
    if "monotonic" in claim.lower():
        types.append("monotonic")
    if requires_unimplemented_statistics(claim):
        types.append("unimplemented_statistics")
    if _wants_greedy_vs_algebraic(claim):
        types.append("bc_vs_greedy")
    return types or ["none_parsed"]


def _expected_metric(verifier: str) -> str:
    return {
        "cap_set_sweep": "cap_set_clp_ratio",
        "sidon_bc_vs_greedy_compare": "bose_chowla_vs_greedy_ratio",
        "sidon_sweep_bose_chowla": "sidon_ratio_to_sqrt_n",
        "sidon_sweep_greedy": "sidon_ratio_to_sqrt_n",
        "sidon_single_point": "sidon_density",
        "sumset": "sumset_growth",
        "ap_free": "ap_free_density",
    }.get(verifier, "sidon_ratio_to_sqrt_n")


def _routing_ok(verifier: str, claim: str) -> bool:
    sidon_verifiers = {
        "sidon_bc_vs_greedy_compare",
        "sidon_sweep_bose_chowla",
        "sidon_sweep_greedy",
        "sidon_single_point",
    }
    cap_verifiers = {"cap_set_sweep"}
    claim_is_sidon = is_sidon_hypothesis(claim, full_text=claim)
    claim_is_cap = is_cap_set_hypothesis(claim, full_text=claim)
    if claim_is_sidon and verifier in sidon_verifiers:
        return True
    if claim_is_cap and verifier in cap_verifiers:
        return True
    if not claim_is_sidon and not claim_is_cap:
        return True
    return False


def inspect_routing(
    hypothesis: dict[str, Any],
    *,
    domain: str = "math_combinatorics",
    dry_run_experiment: bool = True,
) -> dict[str, Any]:
    """
    Show which verifier, features, and claim parsers would activate for a hypothesis.
    Does not call the LLM or launch a campaign.
    """
    _ = domain
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    methodology = str(hypothesis.get("test_methodology") or "")
    claim = extract_claim_text(statement, test_methodology=methodology, full_text=statement)
    features = _infer_features(hypothesis, claim)
    verifier = _resolve_verifier(claim, methodology, statement)
    expected_metric = _expected_metric(verifier)

    flags = {
        "is_cap_set_hypothesis": is_cap_set_hypothesis(statement, methodology, full_text=statement),
        "is_sidon_hypothesis": is_sidon_hypothesis(statement, methodology, full_text=statement),
        "requires_unimplemented_statistics": requires_unimplemented_statistics(claim),
        "wants_asymptotic_sweep": _wants_asymptotic_analysis(claim),
        "wants_bc_vs_greedy": _wants_greedy_vs_algebraic(claim),
    }

    parsed_claim: dict[str, Any] = {}
    band = parse_interval_claim(claim)
    if band:
        parsed_claim["ratio_band"] = list(band)
    upper = parse_ratio_upper_bound(claim)
    if upper is not None:
        parsed_claim["ratio_upper_bound"] = upper
    minimum = parse_minimum_int_claim(claim)
    if minimum is not None:
        parsed_claim["minimum_size"] = minimum

    out: dict[str, Any] = {
        "claim_text": claim[:500],
        "resolved_verifier": verifier,
        "feature_extracted": features[0] if features else "sidon_set_density",
        "expected_metric_name": expected_metric,
        "claim_types": _claim_types(claim),
        "parsed_claim": parsed_claim or None,
        "routing_flags": flags,
        "routing_ok": _routing_ok(verifier, claim),
        "explicit_ns": _extract_n_list(claim),
        "cap_dims": _extract_cap_dims(claim),
    }

    if dry_run_experiment:
        evidence = run_combinatorics_experiment(
            {"statement": statement, "text": statement, "test_methodology": methodology},
            features,
        )
        out["actual_metric_name"] = evidence.get("metric_name")
        out["metric_match"] = evidence.get("metric_name") == expected_metric
        out["claim_checked"] = evidence.get("claim_checked")
        out["verified_true_steps"] = evidence.get("verified_true_steps")
        out["verified_false_steps"] = evidence.get("verified_false_steps")
        claim_eval = evaluate_experiment_claim(claim, evidence)
        out["claim_validation"] = {
            k: claim_eval.get(k)
            for k in ("claim_checked", "claim_supported", "claim_type", "notes_suffix")
            if claim_eval.get(k) is not None
        }
        out["routing_ok"] = out["routing_ok"] and bool(out.get("metric_match"))

    return out


def inspect_corpus(hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    """Run inspect_routing over a list of hypothesis dicts; summarize mismatches."""
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for hyp in hypotheses:
        result = inspect_routing(hyp)
        row = {
            "id": hyp.get("id"),
            "text_preview": (hyp.get("statement") or hyp.get("text") or "")[:120],
            **result,
        }
        rows.append(row)
        if not result.get("routing_ok"):
            mismatches.append(row)
    return {
        "total": len(rows),
        "routing_ok": sum(1 for r in rows if r.get("routing_ok")),
        "routing_mismatches": len(mismatches),
        "mismatch_rate": round(len(mismatches) / max(len(rows), 1), 3),
        "rows": rows,
        "mismatches": mismatches,
    }
