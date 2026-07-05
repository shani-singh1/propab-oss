"""
One-off script: seed the remaining artifacts (astabench proxy score,
domain gap maps) with a real first data point from a live run against public
sources, mirroring what `smoke_run.py` did for `literature_coverage.json`.

Usage:
    python services/literature/scripts/seed_artifacts.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages" / "propab-core"))

from services.literature.app.config import settings
from services.literature.app.evaluator.astabench import record_score, run_litqa2_proxy
from services.literature.app.pipeline import LiteraturePipeline

# LitQA2-shaped: (question, correct_answer, distractors). Correct answers are
# real established facts (verified against arXiv/OEIS/Crossref in this same
# session); distractors are plausible-but-wrong numbers/claims in the same
# domain, including deliberately close near-misses (swapped bound direction,
# swapped attribution, off-by-a-detail) to actually stress the discrimination
# ability rather than just separating obviously-unrelated options.
LITQA2_PROXY_CASES = [
    {
        "question": "What is the best known lower-bound growth rate for the maximum Sidon set size F(n) in {1,...,n}?",
        "correct_answer": "F(n) is at least sqrt(n)(1 - o(1)), per Erdos-Turan/Lindstrom.",
        "distractors": [
            "F(n) is at least n/2 for all n, per a simple greedy argument.",
            "F(n) is at least log(n) for all sufficiently large n.",
        ],
    },
    {
        "question": "What is the best known upper bound on cap set size in F_3^n?",
        "correct_answer": "O(2.756^n), by the Croot-Lev-Pach / Ellenberg-Gijswijt polynomial method.",
        "distractors": [
            "O(3^n / n), by a simple counting argument with no known improvement.",
            "O(2.217^n), matching the best known construction with no gap remaining.",
        ],
    },
    {
        "question": "What sequence gives r3(n), the largest 3-term-AP-free subset size of {1,...,n}?",
        "correct_answer": "OEIS A003002.",
        "distractors": ["OEIS A000045 (Fibonacci numbers).", "OEIS A000040 (the prime numbers)."],
    },
    {
        "question": "Who first studied the problem that led to the Sidon set density question?",
        "correct_answer": "Sidon's original question was addressed by Erdos and Turan in 1941.",
        "distractors": [
            "Sidon's original question was addressed by Behrend in 1946.",
            "Sidon's original question was first resolved by Szemeredi in 1975.",
        ],
    },
    {
        "question": "What did Behrend's 1946 construction establish?",
        "correct_answer": (
            "Behrend's construction gives a 3-term-AP-free subset of {1,...,n} of size "
            "n / exp(c sqrt(log n)), showing the density need not be o(1) trivially."
        ),
        "distractors": [
            "Behrend's construction gives an AP-free subset of size exactly sqrt(n) for all n.",
            "Behrend's construction proves every subset of density > 0 contains a 3-term AP.",
        ],
    },
    {
        "question": "What does Szemeredi's theorem (1975) establish about AP-free sets?",
        "correct_answer": "Every subset of {1,...,n} with positive upper density contains arbitrarily long arithmetic progressions.",
        "distractors": [
            "Szemeredi's theorem shows no subset of {1,...,n} can be free of 3-term arithmetic progressions.",
            "Szemeredi's theorem gives the exact maximum size of a Sidon set for every n.",
        ],
    },
    {
        "question": "What is the current gap in the cap-set problem for F_3^n?",
        "correct_answer": (
            "The gap is between the Croot-Lev-Pach/Ellenberg-Gijswijt upper bound of O(2.756^n) "
            "and the best known construction lower bound of Omega(2.217^n)."
        ),
        "distractors": [
            "The gap has been fully closed: upper and lower bounds now match exactly at 2.756^n.",
            "The gap is between O(3^n) and Omega(2^n), unchanged since the 1990s.",
        ],
    },
    {
        "question": "What does the Mian-Chowla sequence (OEIS A005282) represent?",
        "correct_answer": "A greedily constructed B2/Sidon sequence where every new term keeps all pairwise sums distinct.",
        "distractors": [
            "The sequence of prime numbers used to construct optimal cap sets.",
            "The sequence of maximum AP-free set sizes for each n, per Behrend's bound.",
        ],
    },
]


async def main(*, skip_gaps: bool = False) -> None:
    from propab.domain_modules.math_combinatorics.plugin import MathCombinatoricsPlugin
    from propab.domain_modules.network_diffusion.plugin import NetworkDiffusionPlugin

    pipeline = await LiteraturePipeline.create(settings)
    try:
        math_profile = MathCombinatoricsPlugin().literature_profile()
        # Broad enough to cover every seed-paper topic (Sidon, cap-set, AP-free) —
        # a narrower question (e.g. Sidon-only) under-indexes the other two and
        # starves the AstaBench proxy of evidence for questions about them.
        await pipeline.build_prior(
            research_question=(
                "What is known about Sidon sets, cap sets, and AP-free sets — "
                "their maximum sizes and best known upper/lower bounds?"
            ),
            domain_id="math_combinatorics",
            profile=math_profile,
            depth="standard",
        )

        scores = await run_litqa2_proxy(pipeline, LITQA2_PROXY_CASES)
        print("LitQA2 proxy:", scores)
        record_score(scores, artifacts_dir=str(REPO_ROOT / "artifacts"))

        if skip_gaps:
            return

        gaps = await pipeline.map_gaps(domain_id="math_combinatorics", profile=math_profile)
        gap_dir = REPO_ROOT / "artifacts" / "domain_gap_maps"
        gap_dir.mkdir(parents=True, exist_ok=True)
        (gap_dir / "math_combinatorics.json").write_text(
            json.dumps([q.model_dump() for q in gaps.frontier_questions], indent=2), encoding="utf-8"
        )
        print(f"math_combinatorics gaps: {len(gaps.frontier_questions)} frontier questions")

        nd_profile = NetworkDiffusionPlugin().literature_profile()
        nd_gaps = await pipeline.map_gaps(domain_id="network_diffusion", profile=nd_profile)
        (gap_dir / "network_diffusion.json").write_text(
            json.dumps([q.model_dump() for q in nd_gaps.frontier_questions], indent=2), encoding="utf-8"
        )
        print(f"network_diffusion gaps: {len(nd_gaps.frontier_questions)} frontier questions")
    finally:
        await pipeline.aclose()


if __name__ == "__main__":
    asyncio.run(main(skip_gaps="--skip-gaps" in sys.argv))
