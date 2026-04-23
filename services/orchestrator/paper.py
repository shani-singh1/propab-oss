from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.events import EventEmitter
from propab.paper_compiler import compile_session_methods_latex
from propab.storage import put_bytes
from propab.types import EventType


async def write_paper_minimal(
    *,
    session_id: str,
    session_factory: async_sessionmaker,
    emitter: EventEmitter,
) -> dict:
    """
    Compile deterministic methods from DB traces, optionally run pdflatex, upload to MinIO.
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

    methods_tex = report["combined_latex"]
    pdf_url = None
    tex_url = None
    latex_ok = False
    latex_log = ""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        body_path = tmp / "methods_body.tex"
        body_path.write_text(methods_tex, encoding="utf-8")
        main_path = tmp / "main.tex"
        main_path.write_text(
            "\\documentclass{article}\n"
            "\\usepackage[a4paper,margin=1in]{geometry}\n"
            "\\begin{document}\n"
            "\\input{methods_body}\n"
            "\\end{document}\n",
            encoding="utf-8",
        )

        if shutil.which("pdflatex"):
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=120,
            )
            latex_ok = proc.returncode == 0 and (tmp / "main.pdf").exists()
            latex_log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        else:
            latex_log = "pdflatex not found on PATH"

        main_tex_bytes = main_path.read_bytes()
        body_bytes = body_path.read_bytes()
        pdf_bytes = (tmp / "main.pdf").read_bytes() if (tmp / "main.pdf").exists() else b""

    base = f"sessions/{session_id}/paper"
    tex_url = put_bytes(object_name=f"{base}/main.tex", data=main_tex_bytes, content_type="application/x-tex")
    put_bytes(object_name=f"{base}/methods_body.tex", data=body_bytes, content_type="application/x-tex")
    if latex_ok and pdf_bytes:
        pdf_url = put_bytes(object_name=f"{base}/main.pdf", data=pdf_bytes, content_type="application/pdf")

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_LATEX_COMPILED,
        step="paper.latex",
        payload={
            "success": bool(latex_ok),
            "engine": "pdflatex" if shutil.which("pdflatex") else "none",
            "log_tail": latex_log[-4000:],
        },
    )

    payload = {
        "pdf_url": pdf_url,
        "tex_url": tex_url,
        "methods_latex": methods_tex,
    }
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_READY,
        step="paper.ready",
        payload=payload,
    )
    return payload
