"""
Live measurement: citation verification rate specifically for claims
produced by the new LLM-location + code-verbatim extractor
(extractors/llm_claim_locator.py), against real PubMed/bioRxiv sources.

This is the empirical check for the safety-by-construction claim: verbatim
is always a code-side lookup by sentence index, never LLM-supplied text.
The unit tests already prove this against a mocked LLM; this script proves
it against a real one, on real papers.

Usage:
    python services/literature/scripts/verify_llm_locator_citations.py
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
from services.literature.app.evaluator.metrics import citation_verification_rate
from services.literature.app.pipeline import LiteraturePipeline

BIOLOGY_PROFILE = {
    "seed_papers": [],
    "search_terms": ["CRISPR gene editing", "tumor suppressor", "mouse knockout phenotype", "enzyme kinetics catalysis"],
    "source_priorities": ["pubmed", "biorxiv", "semantic_scholar"],
    "classification_codes": {"mesh": []},
    "open_problem_sources": [],
    "tabulation_sources": [],
    "canonical_surveys": [],
    "novelty_criteria": "",
}


async def main() -> None:
    print(f"llm_api_key configured: {bool(settings.google_api_key)}")
    pipeline = await LiteraturePipeline.create(settings)
    try:
        prior = await pipeline.build_prior(
            research_question="What genes and mutations affect tumor suppression and DNA repair in mouse models?",
            domain_id="llm_locator_verify",
            profile=BIOLOGY_PROFILE,
            depth="standard",
        )
        print(f"papers_indexed={prior.papers_indexed} sources={prior.sources_consulted}")

        all_claims = await pipeline.ctx.structured_store.get_claims("llm_locator_verify")
        llm_claims = [c for c in all_claims if "LLM-located" in c.location]
        regex_claims = [c for c in all_claims if "LLM-located" not in c.location]
        print(f"total claims={len(all_claims)} | llm-located={len(llm_claims)} | regex/structural={len(regex_claims)}")

        if all_claims:
            combined = await citation_verification_rate(pipeline.ctx.sources, all_claims, sample_size=len(all_claims))
            print("\n=== Citation verification rate: ALL claims combined ===")
            print(json.dumps(combined, indent=2))

        if not llm_claims:
            print("No LLM-located claims produced in this run — nothing to verify.")
            return

        report = await citation_verification_rate(pipeline.ctx.sources, llm_claims, sample_size=len(llm_claims))
        print("\n=== Citation verification rate: LLM-located claims only ===")
        print(json.dumps(report, indent=2))

        for c in llm_claims[:5]:
            print(f"\n[{c.claim_type}/{c.status}] {c.location}")
            print(f"  verbatim: {c.verbatim[:150]}")
    finally:
        await pipeline.aclose()


if __name__ == "__main__":
    asyncio.run(main())
