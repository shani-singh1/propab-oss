"""
Known open problems in additive combinatorics for campaign seeding.
"""

from __future__ import annotations

from typing import Any

OPEN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "sidon_density_asymptotics",
        "statement": (
            "What is the exact asymptotic behavior of the maximum Sidon set size F(n) "
            "in {1,...,n}? Known bounds: sqrt(n) - n^{5/16} < F(n) < sqrt(n) + O(n^{1/4})."
        ),
        "open_since": 1941,
        "difficulty": "hard",
        "computationally_approachable": True,
    },
    {
        "id": "cap_set_construction_gap",
        "statement": (
            "CLP/EG upper bound for cap sets in F_3^n is O(2.756^n). "
            "Best construction gives Omega(2.217^n). Can computation close this gap for small n?"
        ),
        "open_since": 2016,
        "difficulty": "medium",
        "computationally_approachable": True,
    },
    {
        "id": "sidon_set_structure",
        "statement": (
            "What structural properties do maximum Sidon sets share? "
            "Are extremal sets spread out or concentrated in intervals?"
        ),
        "open_since": 1960,
        "difficulty": "medium",
        "computationally_approachable": True,
    },
]


def get_literature_prior(question: str) -> dict[str, Any]:
    """Structured literature prior for combinatorics campaigns (static V1)."""
    _ = question
    return {
        "established_facts": [
            {
                "claim": (
                    "Maximum Sidon set size in {1,...,n} satisfies "
                    "sqrt(n)(1 - o(1)) <= F(n) <= sqrt(n) + O(n^{1/4})"
                ),
                "citation": "Erdos and Turan, 1941; Lindstrom, 1972",
                "confidence": "proven",
            },
            {
                "claim": "Cap set size in F_3^n is at most O((2.756)^n)",
                "citation": "Croot-Lev-Pach and Ellenberg-Gijswijt, 2016",
                "confidence": "proven",
            },
            {
                "claim": "Every AP-free subset of {1,...,n} has density o(1)",
                "citation": "Szemeredi, 1975",
                "confidence": "proven",
            },
            {
                "claim": (
                    "Best known AP-free construction achieves density exp(-c*sqrt(log n))"
                ),
                "citation": "Behrend, 1946",
                "confidence": "proven",
            },
        ],
        "open_gaps": [
            "Exact constant in Sidon set asymptotics",
            "Gap between cap set upper bound (2.756^n) and best construction (2.217^n)",
            "Structural characterization of extremal Sidon sets",
        ],
        "contradictions": [],
        "dead_ends": [
            "Simple greedy constructions do not achieve optimal Sidon set density",
            "Random sets achieve Omega(sqrt(n)) Sidon size but not the extremal constant",
        ],
    }
