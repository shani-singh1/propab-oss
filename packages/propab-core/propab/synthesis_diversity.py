"""Synthesis diversity enforcement (fixes.md Track B1, B2, B3)."""
from __future__ import annotations

import re
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

# Tree monoculture forcing applies once the tree has enough nodes to measure skew
# but before NO_DISPATCHABLE (typically ~15-20 nodes in math_combinatorics).
MIN_TREE_NODES_FOR_MONOCULTURE = 8

_SUBJECT_TO_PROBLEM = {
    "cap_set": "cap_set",
    "sidon": "sidon",
    "bc": "bc_comparison",
}


def q1_question_active(question: str) -> bool:
    """True when the campaign question targets the Sidon 0.60 threshold (Q1)."""
    q = (question or "").lower()
    if "0.60" in q or ("0.6" in q and ("cross" in q or "fall" in q or "below" in q)):
        return True
    return "q1" in q and ("0.6" in q or "threshold" in q)


def q1_threshold_target(question: str, *, default: float = 0.60) -> float:
    m = re.search(r"(?:below|cross(?:es)?|under)\s*(0\.\d{2})", (question or "").lower())
    return float(m.group(1)) if m else default


def bootstrap_forced_problem_type(question: str) -> str | None:
    """Before the tree is large enough for monoculture stats, bias away from cap_set."""
    q = (question or "").lower()
    if "cap-set table lookups are lower priority" in q:
        return "sidon"
    if "ap-free" in q:
        return "ap_free"
    return None


def filter_seed_dicts_for_diversity(
    seed_dicts: list[dict[str, Any]],
    *,
    tree_nodes: dict[str, Any],
    active_belief_statements: list[str] | None = None,
    question: str = "",
    generation: int = 0,
) -> list[dict[str, Any]]:
    """Apply the same forced-type rules as synthesis to bootstrap seeds."""
    from propab.numerical_seeds import classify_hypothesis_bucket

    tree_counts = tree_problem_counts_from_nodes(tree_nodes)
    n_typed = sum(tree_counts.values())
    forced = resolve_forced_problem_type(
        [],
        active_belief_statements,
        tree_problem_counts=tree_counts if n_typed >= MIN_TREE_NODES_FOR_MONOCULTURE else None,
    )
    if forced is None and n_typed < MIN_TREE_NODES_FOR_MONOCULTURE:
        forced = bootstrap_forced_problem_type(question)
    if not forced:
        return seed_dicts
    filtered = [
        sd for sd in seed_dicts
        if classify_hypothesis_bucket(
            str(sd.get("text") or ""),
            str(sd.get("test_methodology") or ""),
        ).get("problem_type") == forced
    ]
    if filtered:
        return filtered
    return fallback_synthesis_seeds(forced, generation=generation, question=question)


def forced_from_tree_monoculture(
    tree_counts: dict[str, int],
    *,
    max_fraction: float = 0.40,
    min_nodes: int = MIN_TREE_NODES_FOR_MONOCULTURE,
) -> str | None:
    """When one problem type dominates the tree, force the least-explored type."""
    total = sum(tree_counts.values())
    if total < min_nodes or not tree_counts:
        return None
    dominant, dom_count = max(tree_counts.items(), key=lambda kv: kv[1])
    if dom_count / total <= max_fraction:
        return None
    non_dominant = [p for p in tree_counts if p != dominant]
    if non_dominant:
        return min(non_dominant, key=lambda p: tree_counts[p])
    for alt in PROBLEM_TYPES:
        if alt != dominant:
            return alt
    return None


def tree_problem_counts_from_nodes(
    nodes: dict[str, Any],
) -> dict[str, int]:
    from propab.numerical_seeds import classify_hypothesis_bucket

    counts: Counter[str] = Counter()
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        text = node.get("text") or (getattr(node, "text", None) or "")
        meth = node.get("test_methodology") or (getattr(node, "test_methodology", None) or "")
        bucket = classify_hypothesis_bucket(str(text), str(meth))
        pt = bucket.get("problem_type")
        if pt and pt != "?":
            counts[str(pt)] += 1
    return dict(counts)


def resolve_forced_problem_type(
    recent_buckets: list[dict[str, str]],
    active_belief_statements: list[str] | None = None,
    *,
    streak: int = 3,
    tree_problem_counts: dict[str, int] | None = None,
) -> str | None:
    """Force a new problem type when history or active beliefs are monoculture."""
    from propab.belief_promotion import _belief_subject

    if tree_problem_counts:
        tree_forced = forced_from_tree_monoculture(tree_problem_counts)
        if tree_forced:
            return tree_forced
    # History streak — cap-set-heavy recent rounds should force Sidon even if beliefs already shifted.
    history_forced = forced_problem_type(recent_buckets, streak=streak)
    if history_forced:
        return history_forced
    if active_belief_statements:
        subjects = [_belief_subject(s) for s in active_belief_statements if s.strip()]
        if subjects and len(set(subjects)) == 1:
            dominant = _SUBJECT_TO_PROBLEM.get(subjects[0], subjects[0])
            for alt in PROBLEM_TYPES:
                if alt != dominant:
                    return alt
    return None


def diversity_requirement_prompt(forced_type: str, *, avoid_type: str | None = None) -> str:
    avoid = (
        f" Do NOT propose '{avoid_type}' hypotheses — that bucket is exhausted."
        if avoid_type
        else ""
    )
    return (
        f"MANDATORY DIVERSITY: This round MUST propose at least 2 frontier hypotheses "
        f"focused on problem type '{forced_type}' with falsifiable numeric claims.{avoid}"
    )


FALLBACK_SEED_TEMPLATES: dict[str, dict[str, str]] = {
    "sidon": {
        "text": (
            "Population: Greedy Sidon in {1,...,n} for n in [1000, 5000, 10000]. "
            "Claim: F(n)/sqrt(n) is strictly decreasing across the sweep."
        ),
        "test_methodology": "greedy Sidon density sweep",
    },
    "ap_free": {
        "text": (
            "Population: AP-free greedy sets in {1,...,n} for n in [500, 1000, 2000, 5000]. "
            "Claim: AP-free density decreases monotonically across the sweep."
        ),
        "test_methodology": "greedy AP-free density sweep",
    },
    "bc_comparison": {
        "text": (
            "Population: Greedy Sidon vs Bose-Chowla at matched n=q^2+q for prime q in [50, 200]. "
            "Claim: Greedy ratio exceeds Bose-Chowla ratio at every matched n in the range."
        ),
        "test_methodology": "matched BC vs greedy comparison",
    },
    "sumset": {
        "text": (
            "Population: Random vs Sidon-like sets in {1,...,n} for n=500. "
            "Claim: Sidon-like construction yields strictly smaller |A+A|/|A| than random."
        ),
        "test_methodology": "structured sumset growth comparison",
    },
    "cap_set": {
        "text": (
            "Population: Best-known cap sets in F_3^n for n in [8, 10]. "
            "Claim: CLP ratio size/3^n decreases from n=8 to n=10."
        ),
        "test_methodology": "cap-set CLP table lookup",
    },
}


def fallback_synthesis_seeds(
    forced_type: str,
    *,
    generation: int,
    question: str = "",
) -> list[dict[str, Any]]:
    """Deterministic seeds when LLM synthesis ignores diversity reset (B3 fallback)."""
    _ = question
    tpl = FALLBACK_SEED_TEMPLATES.get(forced_type)
    if not tpl:
        return []
    return [{
        "id": f"fallback_{forced_type}_{generation}",
        "text": tpl["text"],
        "test_methodology": tpl["test_methodology"],
        "expansion_type": "diagnostic",
        "expansion_reason": f"diversity_fallback_{forced_type}",
    }]


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
