"""
Known open problems in binary linear coding theory for campaign seeding.

The frontier: for many binary linear [n, k] parameters there is a GAP between the
best-known lower bound on the minimum distance d (an explicit construction) and
the theoretical upper bound (linear-programming / sphere-packing / Griesmer). A
construction producing a valid [n, k, d] code that meets or beats the best-known
lower bound at an open entry is a genuine result.
"""
from __future__ import annotations

from typing import Any

OPEN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "binary_min_distance_gap",
        "statement": (
            "For many binary linear [n, k] codes there is a gap between the best-known "
            "lower bound d_lo (a construction) and the upper bound d_hi (LP/Griesmer). "
            "Which [n, k] entries admit a construction meeting or beating d_lo?"
        ),
        "open_since": 1960,
        "difficulty": "hard",
        "computationally_approachable": True,
    },
    {
        "id": "optimal_short_codes",
        "statement": (
            "What is the exact maximum minimum distance d(n, k) for small binary "
            "linear [n, k] codes (n <= 32)? Many entries in the Brouwer/Grassl tables "
            "are only bounded, not determined."
        ),
        "open_since": 1998,
        "difficulty": "medium",
        "computationally_approachable": True,
    },
    {
        "id": "construction_vs_bound_families",
        "statement": (
            "Do algebraic families (Hamming, simplex, Reed-Muller, BCH) remain optimal "
            "as n grows, or can modified/shortened/extended constructions beat them at "
            "specific [n, k]?"
        ),
        "open_since": 1970,
        "difficulty": "medium",
        "computationally_approachable": True,
    },
]


def get_literature_prior(question: str) -> dict[str, Any]:
    """Structured literature prior for coding-theory campaigns."""
    _ = question
    return {
        "established_facts": [
            {
                "claim": (
                    "The Singleton bound gives d <= n - k + 1 for any [n, k] linear code; "
                    "binary linear codes rarely meet it (only trivial/MDS cases)."
                ),
                "citation": "Singleton, 1964",
                "confidence": "proven",
            },
            {
                "claim": (
                    "The binary Hamming code [2^r-1, 2^r-1-r, 3] is a perfect "
                    "single-error-correcting code; [7,4,3] is the smallest nontrivial case."
                ),
                "citation": "Hamming, 1950",
                "confidence": "proven",
            },
            {
                "claim": (
                    "The Griesmer bound lower-bounds n given k and d, and is met by many "
                    "small optimal binary codes."
                ),
                "citation": "Griesmer, 1960",
                "confidence": "proven",
            },
            {
                "claim": (
                    "Best-known lower bounds on d for binary linear [n, k] are tabulated "
                    "in the Brouwer/Grassl tables (codetables.de)."
                ),
                "citation": "Brouwer; Grassl, codetables.de",
                "confidence": "proven",
            },
        ],
        "open_gaps": [
            "Exact d(n, k) for many small binary linear codes (n <= 32)",
            "Gap between best-known construction (d_lo) and LP/Griesmer upper bound (d_hi)",
            "Whether shortened/extended variants beat algebraic families at specific [n,k]",
        ],
        "contradictions": [],
        "dead_ends": [
            "Reading a table value as if it were computed is not a discovery",
            "Reporting a minimum distance without an achieving witness codeword is invalid",
            "A code meeting (not exceeding) the best-known lower bound is a rediscovery",
            "k too large for exhaustive 2^k enumeration cannot be certified by this verifier",
        ],
        "required_evidence": [
            "An explicit k x n generator matrix over GF(2)",
            "The computed minimum distance with an achieving (witness) codeword",
            "Independent recomputation of the witness weight on the actual generator",
            "computed_d strictly greater than the best-known lower bound for novelty",
        ],
    }
