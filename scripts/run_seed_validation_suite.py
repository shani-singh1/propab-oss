#!/usr/bin/env python3
"""
Phase 1 seed-generation validation (fixes.md).

Per question: literature → prior → seed generation → relevance gate. No sub-agents.

Success criteria (suite-wide):
  - empty LLM generations < 5%
  - ≥3 discovery hypotheses survive per question
  - control share < 20% of survivors (mean)
  - themes not all general (≥80% of questions have a non-general theme)

Usage:
  python scripts/run_seed_validation_suite.py
  python scripts/run_seed_validation_suite.py --limit 5 --timeout 120
  python scripts/run_seed_validation_suite.py --dry-run   # list questions only

Requires: Postgres, LLM API, embed key (literature gates may be lenient without embed).

Artifacts: artifacts/seed_validation_suite_latest.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Use campaign timeouts/seed batch size (not dev profile defaults).
os.environ.setdefault("PROPAB_PROFILE", "campaign")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "seed_validation_suite_latest.json"


async def _run(args: argparse.Namespace) -> int:
    from propab.config import settings
    from propab.db import create_engine, create_session_factory
    from propab.llm import LLMClient
    from services.orchestrator.seed_validation import (
        SEED_VALIDATION_QUESTIONS,
        _NullEmitter,
        _apply_fast_literature_profile,
        evaluate_suite,
        run_seed_pipeline_for_question,
    )

    if args.dry_run:
        for qid, q in SEED_VALIDATION_QUESTIONS[: args.limit]:
            print(f"[{qid}] {q[:100]}...")
        print(f"Total: {min(args.limit, len(SEED_VALIDATION_QUESTIONS))} questions")
        return 0

    _apply_fast_literature_profile()
    if args.skip_literature:
        settings._seed_validation_skip_literature = True  # type: ignore[attr-defined]
        print("Mode: skip-literature (seed generation + relevance gate only)", flush=True)
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    llm = LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.llm_api_secret,
        emitter=_NullEmitter(session_factory),
        session_factory=session_factory,
    )

    selected = SEED_VALIDATION_QUESTIONS[: args.limit]
    results = []
    for i, (qid, question) in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {qid} …", flush=True)
        r = await run_seed_pipeline_for_question(
            qid,
            question,
            session_factory=session_factory,
            llm=llm,
            prior_timeout_sec=args.prior_timeout,
            total_timeout_sec=args.timeout,
            max_hypotheses=args.max_hypotheses,
        )
        results.append(r)
        status = "OK" if r.ok else "FAIL"
        print(
            f"    {status} disc={r.discovery_count} ctrl={r.control_count} "
            f"empty_llm={r.llm_empty_generation} themes={r.themes} {r.elapsed_sec}s",
            flush=True,
        )
        if r.error:
            print(f"    error: {r.error[:200]}", flush=True)

    report = evaluate_suite(results)
    payload = {
        "profile": settings.propab_profile,
        "questions_run": len(results),
        "empty_generation_rate": report.empty_generation_rate,
        "mean_discovery_survivors": report.mean_discovery_survivors,
        "mean_control_ratio": report.mean_control_ratio,
        "questions_with_non_general_theme": report.questions_with_non_general_theme,
        "checks": {
            "empty_generation_lt_5pct": report.pass_empty_rate,
            "discovery_ge_3_per_question": report.pass_discovery_count,
            "control_ratio_lt_20pct": report.pass_control_ratio,
            "theme_diversity": report.pass_theme_diversity,
        },
        "passed": report.passed,
        "results": [
            {
                "question_id": r.question_id,
                "ok": r.ok,
                "elapsed_sec": r.elapsed_sec,
                "error": r.error,
                "prior_status": r.prior_status,
                "llm_empty_generation": r.llm_empty_generation,
                "raw_llm_count": r.raw_llm_count,
                "surviving": r.surviving,
                "discovery_count": r.discovery_count,
                "control_count": r.control_count,
                "themes": r.themes,
                "hypotheses_preview": r.hypotheses_preview,
            }
            for r in report.results
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== SEED VALIDATION SUITE ===", flush=True)
    print(json.dumps(payload["checks"], indent=2), flush=True)
    print(f"PASSED: {report.passed}", flush=True)
    print(f"Report: {OUT}", flush=True)
    if report.passed:
        print("\nPhase 2: start long campaign with:", flush=True)
        print("  python scripts/start_contagion_campaign.py --hours 7", flush=True)
    return 0 if report.passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed-generation validation suite (fixes.md Phase 1)")
    parser.add_argument("--limit", type=int, default=25, help="Number of questions (max 25)")
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per question (total)")
    parser.add_argument("--prior-timeout", type=int, default=90, help="Seconds for literature+prior")
    parser.add_argument("--max-hypotheses", type=int, default=5, help="Seeds per question")
    parser.add_argument("--dry-run", action="store_true", help="List questions only")
    parser.add_argument(
        "--skip-literature",
        action="store_true",
        help="Skip literature/prior fetch (seed+gate only; use when embed API is rate-limited)",
    )
    args = parser.parse_args()
    args.limit = max(1, min(args.limit, 25))
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
