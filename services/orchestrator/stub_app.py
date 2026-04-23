"""
Phase 1 orchestrator service stub (ARCHITECTURE.md §16, §17).

The full `research_loop` still runs inside the API process today; this service
exists so compose matches the documented service map and exposes a liveness URL.
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
