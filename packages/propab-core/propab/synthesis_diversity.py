"""Synthesis diversity enforcement (fixes.md Track B1, B2, B3)."""
from __future__ import annotations

from collections import Counter
from typing import Any

SYNTHESIS_DIVERSITY_REQUIREMENTS = """
Each synthesis round must propose hypotheses covering at least 2 of these 4 dimensions:
1. PROBLEM TYPE: vary between Sidon sets, cap sets, AP-free sets, sumset growth
2. PARAMETER SCALE: vary between small n (100-1000), large n (1000-50000),
   or asymptotic characterization
3. CLAIM TYPE: vary between threshold claims (first n where X < Y),
   density claims (X for all n in range), comparison claims (A vs B at scale)
4. DIRECTION: mix confirming hypotheses (expected true) with refuting hypotheses

If all recent hypotheses have been in the same dimension bucket,
explicitly break out of that bucket.
"""

PROBLEM_TYPES = ("sidon", "cap_set", "ap_free", "sumset", "bc_comparison")


def history_problem_counts(recent_buckets: list[dict[str, str]]) -> Counter[str]:
    return Counter(b.get("problem_type", "sidon") for b in recent_buckets)


def forced_problem_type(recent_buckets: list[dict[str, str]], *, streak: int = 5) -> str | None:
    """If last `streak` rounds same problem type, return a different one to force."""
    if len(recent_buckets) < streak:
        return None
    tail = recent_buckets[-streak:]
    counts = history_problem_counts(tail)
    if len(counts) != 1:
        return None
    dominant = next(iter(counts))
    for alt in PROBLEM_TYPES:
        if alt != dominant:
            return alt
    return None


def diversity_reset_instruction(
    recent_buckets: list[dict[str, str]],
    *,
    attempt: int = 0,
) -> str:
    """Prompt snippet for B3 diversity reset before SYNTHESIS_EMPTY."""
    if not recent_buckets:
        return (
            "DIVERSITY RESET: Prior rounds exhausted similar hypotheses. "
            "Propose AP-free density sweep AND sumset growth comparison — "
            "at least one hypothesis each, with falsifiable numeric claims."
        )
    counts = history_problem_counts(recent_buckets)
    least = min(PROBLEM_TYPES, key=lambda p: counts.get(p, 0))
    alts = sorted(PROBLEM_TYPES, key=lambda p: counts.get(p, 0))
    focus = alts[attempt % len(alts)] if alts else least
    return (
        f"DIVERSITY RESET (attempt {attempt + 1}): Least explored problem type is '{least}'. "
        f"Required focus for this round: '{focus}'. Do NOT repeat greedy Sidon sweeps "
        f"already confirmed unless testing a new threshold or n-range above 10000."
    )


def methodology_implementable(
    text: str,
    methodology: str,
    keywords: list[str],
) -> bool:
    combined = f"{text}\n{methodology}".lower()
    return any(kw.lower() in combined for kw in keywords)


def aggregate_diversity_distribution(buckets: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = {
        "problem_type": Counter(),
        "parameter_scale": Counter(),
        "claim_type": Counter(),
    }
    for b in buckets:
        for dim in out:
            out[dim][b.get(dim, "unknown")] += 1
    return {k: dict(v) for k, v in out.items()}
