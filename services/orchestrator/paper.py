from __future__ import annotations

from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.events import EventEmitter
from propab.paper_compiler import compile_session_methods_latex
from propab.types import EventType


async def write_paper_minimal(
    *,
    session_id: str,
    session_factory: async_sessionmaker,
    emitter: EventEmitter,
) -> dict:
    """
    Compile deterministic methods from DB traces and emit paper lifecycle events.
    PDF/MinIO URLs are deferred; payload includes methods LaTeX for download/API use.
    """
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_SECTION_STARTED,
        step="paper.methods",
        payload={"section": "methods"},
    )

    report = await compile_session_methods_latex(session_factory, session_id)

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_TRACE_COMPILED,
        step="paper.trace",
        payload={"per_hypothesis": report["per_hypothesis"]},
    )

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_SECTION_COMPLETED,
        step="paper.methods",
        payload={"section": "methods", "chars": len(report["combined_latex"])},
    )

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_LATEX_COMPILED,
        step="paper.latex",
        payload={"success": True, "engine": "none", "note": "pdflatex and MinIO upload not yet wired."},
    )

    payload = {
        "pdf_url": None,
        "tex_url": None,
        "methods_latex": report["combined_latex"],
    }
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_READY,
        step="paper.ready",
        payload=payload,
    )
    return payload
