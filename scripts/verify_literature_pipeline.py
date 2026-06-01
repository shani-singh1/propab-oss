#!/usr/bin/env python3
"""
Fast verification for the literature pipeline (no 30-min campaign required).

Modes:
  python scripts/verify_literature_pipeline.py           # mocked integration tests (~5s)
  python scripts/verify_literature_pipeline.py --live      # real arXiv fetch, skip PDF (~1-3 min)

Live mode needs Postgres (migration 007) and embed API key for full gates; without embed key
relevance/coverage gates are lenient and the run still validates fetch + diagnostics shape.

Artifacts: artifacts/literature_verify_latest.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "literature_verify_latest.json"


def run_mock_tests() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_literature_pipeline.py",
        "tests/test_literature_build_prior.py",
        "-q",
    ]
    print("Running mocked literature tests:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


async def run_live_smoke(question: str, skip_pdf: bool) -> int:
    from propab.config import settings
    from propab.db import create_engine, create_session_factory
    from propab.events import EventEmitter
    from propab.llm import LLMClient
    from services.orchestrator.intake import parse_question
    from services.orchestrator.literature import build_prior
    from sqlalchemy import text
    from unittest.mock import AsyncMock, patch

    settings.literature_fetch_per_intent = min(settings.literature_fetch_per_intent, 4)
    settings.literature_max_candidates = min(settings.literature_max_candidates, 12)
    settings.literature_expansion_rounds = 1

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    session_id = str(uuid.uuid4())

    async with session_factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO research_sessions (id, question, status, stage)
                VALUES (CAST(:id AS uuid), :question, 'active', 'literature')
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"id": session_id, "question": question[:2000]},
        )
        await session.commit()

    class _NoRedisEmitter(EventEmitter):
        def __init__(self, sf):
            self.session_factory = sf

        async def emit(self, **kwargs):
            return None

    emitter = _NoRedisEmitter(session_factory)
    llm = LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.llm_api_secret,
        emitter=emitter,
        session_factory=session_factory,
    )

    parsed = await parse_question(question)
    patches = []
    if skip_pdf:
        patches.append(
            patch(
                "services.orchestrator.literature._enrich_papers_with_pdf",
                new_callable=AsyncMock,
            )
        )

    try:
        for p in patches:
            p.start()
        prior = await build_prior(
            parsed,
            session_id=session_id,
            emitter=emitter,
            session_factory=session_factory,
            paper_ttl_days=30,
            llm=llm,
        )
    finally:
        for p in patches:
            p.stop()
        await engine.dispose()

    payload = {
        "session_id": session_id,
        "question": question,
        "evidence_status": prior.evidence_status,
        "evidence_coverage": prior.evidence_coverage,
        "retrieval_diagnostics": prior.retrieval_diagnostics,
        "prior_facts": len(prior.established_facts or []),
        "prior_gaps": len(prior.open_gaps or []),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)

    diag = prior.retrieval_diagnostics or {}
    ok = (
        diag.get("question")
        and "papers_kept" in diag
        and prior.evidence_status in {
            "READY",
            "INSUFFICIENT_EVIDENCE",
            "LOW_COVERAGE",
            "CONFLICTING_EVIDENCE",
        }
    )
    if not ok:
        print("LIVE SMOKE FAILED: missing diagnostics or invalid status", file=sys.stderr)
        return 1
    print(f"LIVE SMOKE OK — status={prior.evidence_status}, kept={diag.get('papers_kept_count')}", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="Run live arXiv smoke (skip PDF by default)")
    p.add_argument("--with-pdf", action="store_true", help="Include PDF download in live mode (slower)")
    p.add_argument(
        "--question",
        default="What is known about Egyptian fraction representations of rational numbers?",
        help="Research question for live smoke",
    )
    args = p.parse_args()

    rc = run_mock_tests()
    if rc != 0:
        return rc

    if args.live:
        return asyncio.run(run_live_smoke(args.question, skip_pdf=not args.with_pdf))
    print("Mock tests passed. Use --live for optional arXiv smoke (~1-3 min, no PDF).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
