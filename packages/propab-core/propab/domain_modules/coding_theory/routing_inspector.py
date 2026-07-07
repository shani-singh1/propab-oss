"""Dry-run routing inspector for coding_theory hypotheses."""
from __future__ import annotations

from typing import Any

from propab.domain_modules.coding_theory.constructors import (
    best_known_distance,
    parse_code_params,
    parse_construction_name,
)
from propab.domain_modules.coding_theory.verifier import (
    _select_generator,
    run_coding_experiment,
)


def inspect_routing(
    hypothesis: dict[str, Any],
    *,
    domain: str = "coding_theory",
    dry_run_experiment: bool = True,
) -> dict[str, Any]:
    """
    Show which generator, construction, and claim parsing would activate for a
    hypothesis. Does not call the LLM or launch a campaign.
    """
    _ = domain
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    methodology = str(hypothesis.get("test_methodology") or "")
    params = parse_code_params(statement)
    construction_req = parse_construction_name(statement, methodology)

    generator, construction, source_note = _select_generator(hypothesis)
    can_build = generator is not None
    n = params.get("n")
    k = params.get("k")

    out: dict[str, Any] = {
        "statement_preview": statement[:200],
        "parsed_params": params,
        "requested_construction": construction_req,
        "resolved_construction": construction,
        "resolution_note": source_note,
        "can_build_code": can_build,
        "best_known_distance": (
            best_known_distance(n, k) if n is not None and k is not None else None
        ),
        "expected_metric_name": "code_minimum_distance",
    }

    if dry_run_experiment and can_build:
        evidence = run_coding_experiment(hypothesis)
        out["computed_min_distance"] = evidence.get("computed_min_distance")
        out["witness_recheck_ok"] = evidence.get("witness_recheck_ok")
        out["discovery_worthy"] = evidence.get("discovery_worthy")
        out["trivial_rediscovery"] = evidence.get("trivial_rediscovery")
        out["verified_true_steps"] = evidence.get("verified_true_steps")
        out["verified_false_steps"] = evidence.get("verified_false_steps")
        out["metric_match"] = evidence.get("metric_name") == "code_minimum_distance"
        out["routing_ok"] = bool(out["metric_match"] and evidence.get("witness_recheck_ok") is not False)
    else:
        out["routing_ok"] = can_build

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
