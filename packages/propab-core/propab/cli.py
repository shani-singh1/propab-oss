"""
Propab developer CLI: fast infrastructure checks and in-process sub-agent harness.

See fixes.md (long feedback loop). Use ``PROPAB_PROFILE=dev`` for short budgets when
running ``propab agent`` so sandbox and step caps match local debugging.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any
from uuid import uuid4

from propab.replay_support import register_replay_parsers

DEFAULT_HARNESS_HYPOTHESIS = (
    "A 784-128-64-10 MLP with batch normalization on MNIST reaches at least baseline "
    "val_accuracy after a short train_model run; compare to baseline with statistical_significance."
)

# (domain, question, hypothesis, baseline_metric) — exercises distinct tool clusters.
HARNESS_BANK: list[tuple[str, str, str, float]] = [
    (
        "deep_learning",
        "MNIST harness bank.",
        "Train a small MLP on MNIST with train_model (n_steps<=40) and report val_accuracy.",
        0.9,
    ),
    (
        "ml_research",
        "Stats harness bank.",
        "Call statistical_significance with results_a [0.5,0.48,0.49] and results_b [0.52,0.51,0.53].",
        0.5,
    ),
    (
        "ml_research",
        "Literature compare harness.",
        "Call literature_baseline_compare with our_results [0.91,0.92,0.90], baseline_value 0.88, metric_direction higher_is_better.",
        0.88,
    ),
    (
        "algorithm_optimization",
        "Gradient harness.",
        "Call compare_gradient_methods with methods ['sgd','adam'], n_steps=80, learning_rate=0.02.",
        0.0,
    ),
]


def _cmd_health(args: argparse.Namespace) -> int:
    import redis
    from sqlalchemy import text

    from propab.config import settings
    from propab.db import create_engine, create_session_factory
    from propab.tools.registry import ToolRegistry

    print("redis …", end=" ", flush=True)
    r = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    r.ping()
    print("ok")

    print("postgres …", end=" ", flush=True)

    async def _pg() -> None:
        engine = create_engine(settings.database_url)
        factory = create_session_factory(engine)
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        await engine.dispose()

    asyncio.run(_pg())
    print("ok")

    print("tool_registry …", end=" ", flush=True)
    reg = ToolRegistry()
    n = len(reg.get_all_specs())
    print(f"ok ({n} tools)")

    if args.with_train_smoke:
        print("train_model smoke (mnist, n_steps=15) …", end=" ", flush=True)
        res = reg.call(
            "train_model",
            {"model_id": "auto", "dataset": "mnist", "n_steps": 15, "task": "classification"},
        )
        if not res.success:
            err = res.error.to_dict() if res.error and hasattr(res.error, "to_dict") else res.error
            print("FAIL", err)
            return 1
        print("ok")

    if args.with_celery:
        print("celery inspect ping …", end=" ", flush=True)
        try:
            from services.worker.celery_app import app as celery_app

            ping = celery_app.control.inspect(timeout=3.0).ping() if celery_app else None
            if not ping:
                print("FAIL (no workers responded — start worker container)", file=sys.stderr)
                return 1
            print("ok", f"({len(ping)} worker(s))")
        except Exception as exc:
            print(f"FAIL ({exc})", file=sys.stderr)
            return 1

    print("all checks passed.")
    return 0


async def _seed_harness_session(
    *,
    session_id: str,
    hypothesis_id: str,
    question: str,
    hypothesis_text: str,
) -> None:
    from sqlalchemy import text

    from propab.config import settings
    from propab.db import create_engine, create_session_factory

    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)
    async with factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO research_sessions (id, question, status, stage)
                VALUES (CAST(:id AS uuid), :question, 'running', 'harness')
                """
            ),
            {"id": session_id, "question": question},
        )
        await session.execute(
            text(
                """
                INSERT INTO hypotheses (
                    id, session_id, round_id, text, test_methodology, scores_json,
                    rank, status, verdict, confidence, evidence_summary, key_finding, created_at
                ) VALUES (
                    CAST(:id AS uuid), CAST(:session_id AS uuid), NULL,
                    :text, :tm, CAST(:scores AS jsonb),
                    1, 'pending', NULL, NULL, NULL, NULL, NOW()
                )
                """
            ),
            {
                "id": hypothesis_id,
                "session_id": session_id,
                "text": hypothesis_text,
                "tm": "Harness: minimal methodology for sub-agent smoke.",
                "scores": "{}",
            },
        )
        await session.commit()
    await engine.dispose()


async def _cleanup_harness_session(session_id: str) -> None:
    from sqlalchemy import text

    from propab.config import settings
    from propab.db import create_engine, create_session_factory

    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)
    async with factory() as session:
        hid_subq = "SELECT id FROM hypotheses WHERE session_id = CAST(:id AS uuid)"
        await session.execute(
            text(f"DELETE FROM tool_calls WHERE hypothesis_id IN ({hid_subq})"),
            {"id": session_id},
        )
        await session.execute(
            text(f"DELETE FROM experiment_steps WHERE hypothesis_id IN ({hid_subq})"),
            {"id": session_id},
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
    await engine.dispose()


def _cmd_agent(args: argparse.Namespace) -> int:
    import logging

    prof = (getattr(args, "profile", None) or "dev").strip().lower()
    os.environ["PROPAB_PROFILE"] = prof
    from propab.config import settings as _settings, _apply_profile

    _apply_profile(_settings)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        force=True,
    )

    from services.worker.runner import run_sub_agent_sync

    session_id = args.session_id or str(uuid4())
    hypothesis_id = args.hypothesis_id or str(uuid4())
    question = args.question
    hyp_text = args.hypothesis

    asyncio.run(
        _seed_harness_session(
            session_id=session_id,
            hypothesis_id=hypothesis_id,
            question=question,
            hypothesis_text=hyp_text,
        )
    )

    hypothesis_dict: dict[str, Any] = {
        "id": "harness",
        "text": hyp_text,
        "test_methodology": "Smoke run from propab agent harness.",
        "scores": {},
        "rank": 1,
        "gap_reference": "",
        "expected_result": "",
        "refinement_of": None,
    }
    payload: dict[str, Any] = {
        "session_id": session_id,
        "hypothesis_id": hypothesis_id,
        "hypothesis": hypothesis_dict,
        "baseline": {
            "metric_name": "val_accuracy",
            "metric_value": float(args.baseline_metric),
            "description": "Harness baseline (set via --baseline-metric).",
            "lit_compare_safe": abs(float(args.baseline_metric)) >= 1e-9,
        },
        "prior": {
            "established_facts": [],
            "contested_claims": [],
            "open_gaps": [],
            "dead_ends": [],
            "key_papers": [],
        },
        "domain": str(args.domain or "deep_learning"),
        "question": question,
    }
    if args.fast_baseline:
        payload["fast_path"] = "baseline_measurement"
        payload["baseline_measurement"] = {"dataset": "mnist", "n_steps": int(args.baseline_n_steps)}

    try:
        out = run_sub_agent_sync(payload)
    except Exception as exc:
        print(f"run_sub_agent_sync failed: {exc}", file=sys.stderr)
        if args.cleanup:
            asyncio.run(_cleanup_harness_session(session_id))
        return 1

    print(json.dumps(out, indent=2, default=str))

    if args.cleanup:
        asyncio.run(_cleanup_harness_session(session_id))
    else:
        print(
            f"\nKept DB rows: session_id={session_id} hypothesis_id={hypothesis_id} "
            f"(re-run with --cleanup to delete).",
            file=sys.stderr,
        )
    return 0


def _cmd_bank(args: argparse.Namespace) -> int:
    import logging

    prof = (getattr(args, "profile", None) or "dev").strip().lower()
    os.environ["PROPAB_PROFILE"] = prof
    from propab.config import settings as _settings, _apply_profile

    _apply_profile(_settings)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        force=True,
    )

    from services.worker.runner import run_sub_agent_sync

    fails = 0
    for i, (domain, question, hyp, baseline) in enumerate(HARNESS_BANK):
        if args.only is not None and i != int(args.only):
            continue
        sid = str(uuid4())
        hid = str(uuid4())
        print(f"\n=== bank case {i}: {domain} ===", flush=True)
        asyncio.run(_seed_harness_session(session_id=sid, hypothesis_id=hid, question=question, hypothesis_text=hyp))
        payload: dict[str, Any] = {
            "session_id": sid,
            "hypothesis_id": hid,
            "hypothesis": {
                "id": f"bank{i}",
                "text": hyp,
                "test_methodology": "propab bank",
                "scores": {},
                "rank": 1,
                "gap_reference": "",
                "expected_result": "",
                "refinement_of": None,
            },
            "baseline": {
                "metric_name": "val_accuracy",
                "metric_value": float(baseline),
                "description": "bank",
                "lit_compare_safe": abs(float(baseline)) >= 1e-9,
            },
            "prior": {
                "established_facts": [],
                "contested_claims": [],
                "open_gaps": [],
                "dead_ends": [],
                "key_papers": [],
            },
            "domain": domain,
            "question": question,
        }
        try:
            out = run_sub_agent_sync(payload)
            print("verdict:", out.get("verdict"), "failure:", out.get("failure_reason"))
        except Exception as exc:
            print("EXCEPTION", exc, file=sys.stderr)
            fails += 1
        finally:
            if args.cleanup:
                asyncio.run(_cleanup_harness_session(sid))
    return 1 if fails else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="propab", description="Propab developer utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_health = sub.add_parser("health", help="Redis + Postgres + tool registry (optional train smoke).")
    p_health.add_argument(
        "--with-train-smoke",
        action="store_true",
        help="Also run a tiny train_model call (slower; needs sandbox stack).",
    )
    p_health.add_argument(
        "--with-celery",
        action="store_true",
        help="Ping Celery workers via broker (needs worker container running).",
    )
    p_health.set_defaults(func=_cmd_health)

    p_agent = sub.add_parser(
        "agent",
        help="Run one sub-agent in-process (no Celery). Seeds a throwaway session + hypothesis row.",
    )
    p_agent.add_argument(
        "--profile",
        default="dev",
        help="PROPAB_PROFILE for this run (e.g. campaign, dev). Re-applies propab.config settings.",
    )
    p_agent.add_argument("--hypothesis", default=DEFAULT_HARNESS_HYPOTHESIS, help="Hypothesis text")
    p_agent.add_argument(
        "--question",
        default="MNIST MLP architecture smoke (harness).",
        help="Parent research question",
    )
    p_agent.add_argument("--session-id", default="", help="Fixed UUID for session (default: random)")
    p_agent.add_argument("--hypothesis-id", default="", help="Fixed UUID for hypothesis row (default: random)")
    p_agent.add_argument("--domain", default="deep_learning", help="Routed domain hint for heuristic mode")
    p_agent.add_argument(
        "--baseline-metric",
        type=float,
        default=0.92,
        help="Numeric baseline injected for literature_baseline_compare guards",
    )
    p_agent.add_argument(
        "--fast-baseline",
        action="store_true",
        help="Worker fast_path baseline_measurement (train_model only, no full agent loop).",
    )
    p_agent.add_argument("--baseline-n-steps", type=int, default=40, help="With --fast-baseline: train_model n_steps")
    p_agent.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the harness research_sessions row after run (CASCADE cleans hypotheses).",
    )
    p_agent.set_defaults(func=_cmd_agent)

    p_bank = sub.add_parser(
        "bank",
        help="Run the fixed multi-case harness bank (PROPAB_PROFILE=dev by default).",
    )
    p_bank.add_argument(
        "--profile",
        default="dev",
        help="PROPAB_PROFILE for all bank cases (e.g. campaign to match production caps).",
    )
    p_bank.add_argument("--only", type=int, default=None, help="Run single case index (0..n-1)")
    p_bank.add_argument("--cleanup", action="store_true", help="Delete harness DB rows after each case")
    p_bank.set_defaults(func=_cmd_bank)

    register_replay_parsers(sub)

    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(0 if code is None else int(code))


if __name__ == "__main__":
    main()
