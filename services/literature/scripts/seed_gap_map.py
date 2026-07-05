"""One-off: seed artifacts/domain_gap_maps/<domain>.json for a given domain via a live /gaps run."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages" / "propab-core"))

from services.literature.app.config import settings
from services.literature.app.pipeline import LiteraturePipeline


async def main(domain_id: str) -> None:
    import importlib

    from propab.domain_modules.base import DomainPlugin

    module = importlib.import_module(f"propab.domain_modules.{domain_id}.plugin")
    plugin_cls = next(
        v for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, DomainPlugin) and v is not DomainPlugin
    )
    profile = plugin_cls().literature_profile()

    pipeline = await LiteraturePipeline.create(settings)
    try:
        gaps = await pipeline.map_gaps(domain_id=domain_id, profile=profile)
        gap_dir = REPO_ROOT / "artifacts" / "domain_gap_maps"
        gap_dir.mkdir(parents=True, exist_ok=True)
        (gap_dir / f"{domain_id}.json").write_text(
            json.dumps([q.model_dump() for q in gaps.frontier_questions], indent=2), encoding="utf-8"
        )
        print(f"{domain_id} gaps: {len(gaps.frontier_questions)} frontier questions")
    finally:
        await pipeline.aclose()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "network_diffusion"))
