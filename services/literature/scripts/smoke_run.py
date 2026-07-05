"""
Live smoke run against real public sources — not a test (no assertions,
hits the network), a way to sanity-check the pipeline end-to-end and seed
`artifacts/literature_coverage.json` with a genuine first data point.

Usage:
    python services/literature/scripts/smoke_run.py
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


async def main() -> None:
    from propab.domain_modules.math_combinatorics.plugin import MathCombinatoricsPlugin

    profile = MathCombinatoricsPlugin().literature_profile()
    pipeline = await LiteraturePipeline.create(settings)
    try:
        prior = await pipeline.build_prior(
            research_question="What is known about the maximum size of Sidon sets in {1,...,n}?",
            domain_id="math_combinatorics",
            profile=profile,
            depth="standard",
        )
        print(f"papers_indexed={prior.papers_indexed} sources={prior.sources_consulted}")
        print(f"established_facts={len(prior.established_facts)} open_gaps={len(prior.open_gaps)} "
              f"tabulated_values={len(prior.tabulated_values)}")

        claims = await pipeline.ctx.structured_store.get_claims("math_combinatorics")
        citation_report = await citation_verification_rate(pipeline.ctx.sources, claims, sample_size=30)
        print("citation_verification_rate:", citation_report)

        coverage = await pipeline.coverage()
        artifacts_dir = REPO_ROOT / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        domains = [d.model_dump() for d in coverage.domains]
        for d in domains:
            if d["domain_id"] == "math_combinatorics":
                d["citation_verification_rate"] = citation_report["rate"]
                d["citation_verification_sample"] = citation_report
        (artifacts_dir / "literature_coverage.json").write_text(
            json.dumps({"domains": domains}, indent=2), encoding="utf-8"
        )
        print("Wrote artifacts/literature_coverage.json")
    finally:
        await pipeline.aclose()


if __name__ == "__main__":
    asyncio.run(main())
