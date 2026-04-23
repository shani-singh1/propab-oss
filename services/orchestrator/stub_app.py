"""
Legacy stub kept for reference. Production entrypoint is ``services.orchestrator.main:app``
(Dockerfile CMD). Set ``ORCHESTRATOR_URL`` on the API to delegate ``run_research_loop`` here.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Propab Orchestrator", version="0.1.0-stub")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "orchestrator", "phase": "1-stub"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Orchestrator stub — research loop runs in API until Phase 3 split."}
