#!/usr/bin/env python3
"""
Hybrid think–act harness: require one LLM-chosen ``action_type=code`` after a stats tool.

Validates the fixed stub path (multiline ``code_description``, ``open(`` in prose) stays
inline — no Docker wall timeouts on ``code_exec`` rows.

Requires: Postgres + Redis (``python -m propab health``), LLM key or Ollama (``can_llm``).

Usage:
  python scripts/verify_think_act_code_harness.py
  python scripts/verify_think_act_code_harness.py --retries 2
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from uuid import uuid4

from sqlalchemy import text


HARNESS_QUESTION = "Sandbox think–act code-path regression (internal)."
HARNESS_HYPOTHESIS = (
    "HARNESS (two steps, order matters). "
    "(1) Call statistical_significance with results_a [0.50, 0.51, 0.49] and "
    "results_b [0.52, 0.51, 0.53]. "
    "(2) Mandatory: the tool list has no tool that returns SHA256 hexdigest of an arbitrary "
    "UTF-8 audit string. After (1) succeeds, you MUST take exactly one action_type=\"code\" "
    "step. In code_description use exactly two lines: first line `audit:v2`, second line "
    "`note: open( is prose only` — this exercises newline + blocked-substring handling in "
    "the worker stub. Then use action_type stop. Do not call train_model."
)


async def _delete_harness_rows(
    factory,
    *,
    session_id: str,
    hypothesis_id: str,
) -> None:
    async with factory() as session:
        await session.execute(
            text("DELETE FROM tool_calls WHERE hypothesis_id = CAST(:hid AS uuid)"),
            {"hid": hypothesis_id},
        )
        await session.execute(
            text("DELETE FROM experiment_steps WHERE hypothesis_id = CAST(:hid AS uuid)"),
            {"hid": hypothesis_id},
        )
        await session.execute(
            text("DELETE FROM events WHERE session_id = CAST(:id AS uuid)"),
            {"id": session_id},
        )
        await session.execute(
            text("DELETE FROM llm_calls WHERE session_id = CAST(:id AS uuid)"),
            {"id": session_id},
        )
        await session.execute(
            text("DELETE FROM hypotheses WHERE session_id = CAST(:id AS uuid)"),
            {"id": session_id},
        )
        await session.execute(
            text("DELETE FROM research_sessions WHERE id = CAST(:id AS uuid)"),
            {"id": session_id},
        )
        await session.commit()


async def _code_exec_rows(factory, hypothesis_id: str) -> list[dict]:
    async with factory() as session:
        res = await session.execute(
            text(
                """
                SELECT step_index, duration_ms, input_json::text, output_json::text, error_json::text
                FROM experiment_steps
                WHERE hypothesis_id = CAST(:hid AS uuid) AND step_type = 'code_exec'
                ORDER BY step_index
                """
            ),
            {"hid": hypothesis_id},
        )
        return [dict(r) for r in res.mappings().all()]


async def _one_run(*, factory, session_id: str, hypothesis_id: str) -> list[dict]:
    from propab.cli import _seed_harness_session
    from services.worker.sub_agent_loop import run_sub_agent_async

    await _seed_harness_session(
        session_id=session_id,
        hypothesis_id=hypothesis_id,
        question=HARNESS_QUESTION,
        hypothesis_text=HARNESS_HYPOTHESIS,
    )
    payload = {
        "session_id": session_id,
        "hypothesis_id": hypothesis_id,
        "hypothesis": {
            "id": "think_act_code_harness",
            "text": HARNESS_HYPOTHESIS,
            "test_methodology": "think_act code harness",
            "scores": {},
            "rank": 1,
            "gap_reference": "",
            "expected_result": "",
            "refinement_of": None,
        },
        "baseline": {
            "metric_name": "val_accuracy",
            "metric_value": 0.5,
            "description": "harness baseline",
            "lit_compare_safe": True,
        },
        "prior": {
            "established_facts": [],
            "contested_claims": [],
            "open_gaps": [],
            "dead_ends": [],
            "key_papers": [],
        },
        "domain": "ml_research",
        "question": HARNESS_QUESTION,
    }
    await run_sub_agent_async(payload)
    return await _code_exec_rows(factory, hypothesis_id)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retries", type=int, default=2, help="Extra attempts if LLM skips code")
    args = parser.parse_args()

    os.environ.setdefault("PROPAB_PROFILE", "dev")

    from propab.config import settings, _apply_profile

    _apply_profile(settings)
    # Dev profile forces heuristic; override for this harness only.
    settings.sub_agent_plan_source = "hybrid"
    settings.agent_max_steps = max(int(settings.agent_max_steps), 12)

    can_llm = settings.llm_provider.strip().lower() == "ollama" or bool(settings.llm_api_secret.strip())
    if not can_llm:
        print("SKIP: no LLM (set OPENAI_API_KEY / GOOGLE_API_KEY or use LLM_PROVIDER=ollama).", file=sys.stderr)
        return 0

    from propab.db import create_engine, create_session_factory

    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)

    attempts = 1 + max(0, int(args.retries))
    last_rows: list[dict] = []

    try:
        for attempt in range(attempts):
            sid, hid = str(uuid4()), str(uuid4())
            try:
                last_rows = await _one_run(factory=factory, session_id=sid, hypothesis_id=hid)
            finally:
                await _delete_harness_rows(factory, session_id=sid, hypothesis_id=hid)

            if last_rows:
                break
            print(
                f"Attempt {attempt + 1}/{attempts}: LLM did not take a code step; retrying…",
                file=sys.stderr,
            )

        if not last_rows:
            print(
                "FAIL: no code_exec rows after retries — LLM never chose action_type=code.",
                file=sys.stderr,
            )
            return 2

        print("think_act code harness: OK")
        print("  code_exec rows:", len(last_rows))
        rc = 0
        for r in last_rows:
            ej = r.get("error_json") or ""
            print("  step", r["step_index"], "duration_ms", r["duration_ms"])
            if "docker_timeout" in ej or "docker_read_timeout" in ej:
                print("  FAIL: Docker wall timeout in error_json", file=sys.stderr)
                rc = 1
            if int(r["duration_ms"] or 0) > 120_000:
                print("  FAIL: code_exec duration suspiciously high", file=sys.stderr)
                rc = 1
            inp = r.get("input_json") or ""
            if "import json, sys" not in inp and "import json,sys" not in inp.replace(" ", ""):
                print("  WARN: stored code does not look like think–act stub", file=sys.stderr)
        if rc == 0:
            print("  no docker wall timeouts; stub path exercised.")
        return rc
    finally:
        await engine.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
