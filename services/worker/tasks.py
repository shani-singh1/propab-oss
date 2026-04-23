from __future__ import annotations

from services.worker.celery_app import app
from services.worker.runner import run_sub_agent_sync


@app.task(name="propab.run_sub_agent")
def run_sub_agent_task(payload: dict) -> dict:
    return run_sub_agent_sync(payload)
