"""CLI helpers for snapshot replay and tool-call trace replay."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.campaign_snapshot import read_snapshot
from propab.config import settings
from propab.db import create_engine, create_redis, create_session_factory
from propab.tools.registry import ToolRegistry


def cmd_replay(args: argparse.Namespace) -> int:
    from services.orchestrator.campaign_loop import build_campaign_synthesis_payload

    path = Path(args.snapshot)
    if not path.is_file():
        print(f"not found: {path}", file=sys.stderr)
        return 2
    _, campaign, prior = read_snapshot(path)
    syn = build_campaign_synthesis_payload(campaign)
    phase = (args.phase or "synthesis").strip().lower()
    if phase == "synthesis":
        print(json.dumps(syn, indent=2, default=str))
        return 0
    if phase == "abstract":
        from propab.paper_sections import generate_prose_sections

        async def _abs() -> None:
            out = await generate_prose_sections(
                llm=None,
                session_id=campaign.id,
                question=campaign.question,
                prior=prior,
                synthesis=syn,
            )
            print(out.get("abstract", ""))

        asyncio.run(_abs())
        return 0
    if phase == "paper":
        from propab.events import EventEmitter
        from propab.llm import LLMClient
        from services.orchestrator.paper import write_paper_minimal

        async def _paper() -> None:
            engine = create_engine(settings.database_url)
            factory = create_session_factory(engine)
            redis = await create_redis(settings.redis_url)
            emitter = EventEmitter(source="replay", redis=redis, session_factory=factory)
            llm = LLMClient(
                provider=settings.llm_provider,
                model=settings.llm_model,
                api_key=settings.llm_api_secret,
                emitter=emitter,
                session_factory=factory,
            )
            await write_paper_minimal(
                session_id=campaign.id,
                session_factory=factory,
                emitter=emitter,
                llm=llm,
                question=campaign.question,
                prior=prior,
                synthesis=syn,
            )
            await redis.close()
            await engine.dispose()

        asyncio.run(_paper())
        return 0
    print(f"unknown phase: {phase}", file=sys.stderr)
    return 2


async def _fetch_tool_calls(
    factory: async_sessionmaker,
    *,
    hypothesis_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    async with factory() as session:
        result = await session.execute(
            text(
                """
                SELECT tc.tool_name, tc.params_json, tc.result_json, tc.success, es.step_index
                FROM tool_calls tc
                JOIN experiment_steps es ON es.id = tc.step_id
                WHERE tc.hypothesis_id = CAST(:hid AS uuid)
                ORDER BY es.step_index NULLS LAST, tc.id
                """
            ),
            {"hid": hypothesis_id},
        )
        for r in result.mappings().all():
            rows.append(dict(r))
    return rows


def cmd_trace_replay(args: argparse.Namespace) -> int:
    reg = ToolRegistry()

    async def _run() -> None:
        engine = create_engine(settings.database_url)
        factory = create_session_factory(engine)
        rows = await _fetch_tool_calls(factory, hypothesis_id=args.hypothesis_id)
        await engine.dispose()
        if not rows:
            print("no tool_calls for hypothesis_id", file=sys.stderr)
            return
        for i, row in enumerate(rows):
            name = row.get("tool_name") or ""
            params = row.get("params_json")
            prev_out = row.get("result_json")
            if isinstance(params, str):
                params = json.loads(params) if params.strip() else {}
            elif params is None:
                params = {}
            print(f"--- step {i} {name} ---")
            fresh = reg.call(str(name), dict(params))
            print("replay_success:", fresh.success)
            if args.show_diff and prev_out is not None:
                try:
                    po = json.loads(prev_out) if isinstance(prev_out, str) else prev_out
                except json.JSONDecodeError:
                    po = prev_out
                no = fresh.output if fresh.success else fresh.error
                if po != no:
                    print("diff: stored != replay (expected when tools are non-deterministic)")
                    print("stored:", json.dumps(po, default=str)[:800])
                    print("replay:", json.dumps(no, default=str)[:800] if no is not None else str(no))

    asyncio.run(_run())
    return 0


def register_replay_parsers(sub: Any) -> None:
    p = sub.add_parser(
        "replay",
        help="Replay from a campaign JSON snapshot (synthesis / abstract / full paper).",
    )
    p.add_argument(
        "--snapshot",
        required=True,
        help="Path to snapshot JSON (e.g. PROPAB_DATA_DIR/campaign_snapshots/<id>/pre_paper.json)",
    )
    p.add_argument(
        "--phase",
        choices=("synthesis", "abstract", "paper"),
        default="synthesis",
        help="synthesis=JSON only; abstract=ledger-aware stub abstract; paper=full pipeline (needs DB + MinIO)",
    )
    p.set_defaults(func=cmd_replay)

    t = sub.add_parser(
        "trace-replay",
        help="Replay recorded tool_calls for a hypothesis UUID against current tool implementations.",
    )
    t.add_argument("--hypothesis-id", required=True, help="hypotheses.id (UUID) from a past run")
    t.add_argument(
        "--show-diff",
        action="store_true",
        help="Print truncated diff vs stored result_json",
    )
    t.set_defaults(func=cmd_trace_replay)
