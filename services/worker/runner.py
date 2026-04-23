from __future__ import annotations

import asyncio


def run_sub_agent_sync(payload: dict) -> dict:
    from services.worker.sub_agent_loop import run_sub_agent_async

    return asyncio.run(run_sub_agent_async(payload))
