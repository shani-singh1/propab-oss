"""
Run one sub-agent in-process (no Celery) and assert code_exec rows show no Docker wall timeouts.

Usage (from repo root):
  python scripts/verify_inprocess_sandbox.py
  python scripts/verify_inprocess_sandbox.py --hybrid   # think-act + LLM (needs API key or Ollama)

Cleanup deletes experiment_steps before hypotheses to satisfy FK order.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from uuid import uuid4

from sqlalchemy import text


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Set SUB_AGENT_PLAN_SOURCE=hybrid for think-act path (requires LLM).",
    )
    args = parser.parse_args()

    os.environ.setdefault("PROPAB_PROFILE", "dev")
    if args.hybrid:
        os.environ["SUB_AGENT_PLAN_SOURCE"] = "hybrid"

    from propab.config import settings, _apply_profile

    _apply_profile(settings)

    from propab.cli import _cleanup_harness_session, _seed_harness_session
    from propab.db import create_engine, create_session_factory
    from services.worker.sub_agent_loop import run_sub_agent_async

    sid, hid = str(uuid4()), str(uuid4())
    q = "Stats harness."
    hyp = (
        "Call statistical_significance with results_a [0.5,0.48,0.49] "
        "and results_b [0.52,0.51,0.53]."
    )
    await _seed_harness_session(session_id=sid, hypothesis_id=hid, question=q, hypothesis_text=hyp)
    payload = {
        "session_id": sid,
        "hypothesis_id": hid,
        "hypothesis": {
            "id": "verify",
            "text": hyp,
            "test_methodology": "verify sandbox",
            "scores": {},
            "rank": 1,
            "gap_reference": "",
            "expected_result": "",
            "refinement_of": None,
        },
        "baseline": {
            "metric_name": "val_accuracy",
            "metric_value": 0.5,
            "description": "verify",
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
        "question": q,
    }

    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)

    async def delete_steps_then_session() -> None:
        async with factory() as session:
            await session.execute(
                text("DELETE FROM tool_calls WHERE hypothesis_id = CAST(:hid AS uuid)"),
                {"hid": hid},
            )
            await session.execute(
                text("DELETE FROM experiment_steps WHERE hypothesis_id = CAST(:hid AS uuid)"),
                {"hid": hid},
            )
            await session.execute(
                text("DELETE FROM events WHERE session_id = CAST(:id AS uuid)"),
                {"id": sid},
            )
            await session.execute(
                text("DELETE FROM llm_calls WHERE session_id = CAST(:id AS uuid)"),
                {"id": sid},
            )
            await session.execute(
                text("DELETE FROM hypotheses WHERE session_id = CAST(:id AS uuid)"),
                {"id": sid},
            )
            await session.execute(
                text("DELETE FROM research_sessions WHERE id = CAST(:id AS uuid)"),
                {"id": sid},
            )
            await session.commit()
        await engine.dispose()

    try:
        out = await run_sub_agent_async(payload)
        print("plan_source:", settings.sub_agent_plan_source)
        print("verdict:", out.get("verdict"), "failure:", out.get("failure_reason"))

        async with factory() as session:
            res = await session.execute(
                text(
                    """
                    SELECT step_index, duration_ms, output_json::text, error_json::text
                    FROM experiment_steps
                    WHERE hypothesis_id = CAST(:hid AS uuid) AND step_type = 'code_exec'
                    ORDER BY step_index
                    """
                ),
                {"hid": hid},
            )
            rows = res.mappings().all()

        print("code_exec rows:", len(rows))
        rc = 0
        for r in rows:
            ej = r["error_json"] or ""
            print("  step", r["step_index"], "duration_ms", r["duration_ms"])
            if "docker_timeout" in ej or "docker_read_timeout" in ej:
                print("  FAIL: sandbox wall timeout in error_json")
                rc = 1
            if ej and ej.strip() not in ("null", "None"):
                print("  error_json head:", ej[:320])
        if rc == 0 and rows:
            print("OK: no docker wall timeout on code_exec rows (inline or fast Docker).")
        elif not rows:
            print("WARN: no code_exec rows this run (agent may not have reached code step).")
        return rc
    finally:
        await delete_steps_then_session()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
