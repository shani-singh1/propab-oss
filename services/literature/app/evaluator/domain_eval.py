"""
Domain-specific novelty evaluation (agent3.md Phase 2).

For each domain that implements ``literature_profile()``, build a 50-100
case test set: known-established claims, known-open claims, tabulated
exact-matches, and values known to be absent from any tabulation. This
module scores the pipeline's ``/novelty`` verdicts against that set —
accuracy must clear 90% before a domain integrates with campaign launch.

Test case schema (JSON-serializable, one dict per case):
    {"category": "established"|"open"|"tabulated_match"|"not_tabulated",
     "claim": "...", "evidence": {...}, "expected_verdict": "known"|"novel"|"uncertain"}
"""
from __future__ import annotations

from typing import Any

from services.literature.app.models import Finding


async def run_domain_eval(
    pipeline: Any, *, domain_id: str, profile: dict[str, Any], test_cases: list[dict[str, Any]]
) -> dict[str, Any]:
    misses: list[dict[str, Any]] = []
    by_category: dict[str, list[bool]] = {}

    for case in test_cases:
        finding = Finding(claim=case["claim"], evidence=case.get("evidence", {}), domain_id=domain_id)
        response = await pipeline.check_novelty(finding=finding, profile=profile)
        correct = response.verdict == case["expected_verdict"]
        category = case.get("category", "uncategorized")
        by_category.setdefault(category, []).append(correct)
        if not correct:
            misses.append(
                {
                    "claim": case["claim"],
                    "category": category,
                    "expected_verdict": case["expected_verdict"],
                    "actual_verdict": response.verdict,
                    "confidence": response.confidence,
                    "explanation": response.explanation,
                }
            )

    total = sum(len(v) for v in by_category.values())
    correct_total = sum(sum(v) for v in by_category.values())
    accuracy = (correct_total / total) if total else 0.0

    return {
        "domain_id": domain_id,
        "n_cases": total,
        "accuracy": round(accuracy, 4),
        "by_category": {
            cat: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for cat, v in by_category.items()
        },
        "misses": misses,
        "meets_bar": accuracy >= 0.90,
    }
