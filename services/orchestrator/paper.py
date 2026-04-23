from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.claim_grounding import ground_session_claims
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.paper_compiler import (
    collect_figure_object_ids,
    compile_references_latex,
    compile_session_methods_latex,
    compile_session_results_latex,
    _latex_escape,
)
from propab.paper_sections import generate_prose_sections, render_paper_tex
from propab.storage import get_object_bytes, put_bytes
from propab.types import EventType


def _safe_figure_filename(index: int, object_id: str) -> str:
    tail = object_id.rsplit("/", 1)[-1] if object_id else f"fig{index}"
    tail = re.sub(r"[^a-zA-Z0-9._-]+", "_", tail).strip("._") or f"fig{index}"
    if "." not in tail or len(tail.split(".")[-1]) > 5:
        tail = f"{tail}.png"
    return f"fig_{index}_{tail}"


def _build_figures_tex_and_files(
    *,
    tmp: Path,
    synthesis: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    """Write figure binaries into tmp; return LaTeX fragment and local filenames for pdflatex."""
    ids = collect_figure_object_ids(synthesis)
    blocks: list[str] = []
    written: list[str] = []
    for i, oid in enumerate(ids):
        data = get_object_bytes(object_name=oid)
        if not data:
            continue
        fname = _safe_figure_filename(i, oid)
        (tmp / fname).write_bytes(data)
        written.append(fname)
        cap = _latex_escape(oid[:160])
        blocks.append(
            "\\begin{figure}[ht]\n\\centering\n"
            f"\\includegraphics[width=0.88\\linewidth]{{{fname}}}\n"
            f"\\caption{{Artifact \\texttt{{{cap}}}}}\n"
            "\\end{figure}\n\n"
        )
    return ("".join(blocks), written)


async def write_paper_minimal(
    *,
    session_id: str,
    session_factory: async_sessionmaker,
    emitter: EventEmitter,
    llm: LLMClient | None = None,
    question: str | None = None,
    prior: dict[str, Any] | None = None,
    synthesis: dict[str, Any] | None = None,
) -> dict:
    """
    Compile methods + results + references from DB; optional LLM prose; embed session figures;
    render arXiv-style LaTeX; run pdflatex when available; upload to MinIO.
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
        event_type=EventType.PAPER_SECTION_STARTED,
        step="paper.results",
        payload={"section": "results"},
    )
    results_tex = await compile_session_results_latex(session_factory, session_id)
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_SECTION_COMPLETED,
        step="paper.results",
        payload={"section": "results", "chars": len(results_tex)},
    )

    references_tex = compile_references_latex(prior or {})

    methods_tex = report["combined_latex"]
    q = (question or "").strip() or "Research session"
    title = q.replace("\n", " ")[:160]

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_SECTION_STARTED,
        step="paper.prose",
        payload={"section": "abstract_intro_discussion_conclusion"},
    )
    prose = await generate_prose_sections(
        llm=llm,
        session_id=session_id,
        question=q,
        prior=prior,
        synthesis=synthesis,
    )
    try:
        grounding = await ground_session_claims(session_factory, session_id, prose)
    except Exception as exc:  # noqa: BLE001 — paper should still render
        grounding = {"error": str(exc), "grounded": [], "ungrounded": []}
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_CLAIM_GROUNDING,
        step="paper.claim_grounding",
        payload=grounding,
    )
    un = grounding.get("ungrounded") or []
    if isinstance(un, list) and len(un) > 0:
        prose = dict(prose)
        prose["discussion"] = (
            (prose.get("discussion") or "")
            + r"\paragraph{Trace coverage.} Lexical grounding against stored \texttt{experiment\_steps} "
            + f"flagged {len(un)} sentence(s) with weak overlap; treat qualitative claims in those themes as "
            + "requiring manual verification against the trace."
        )
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_SECTION_COMPLETED,
        step="paper.prose",
        payload={"section": "abstract_intro_discussion_conclusion", "chars": sum(len(v) for v in prose.values())},
    )

    pdf_url = None
    tex_url = None
    latex_ok = False
    latex_log = ""
    figure_files: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        figures_tex, figure_files = _build_figures_tex_and_files(tmp=tmp, synthesis=synthesis)

        full_tex = render_paper_tex(
            title=title,
            abstract=prose["abstract"],
            introduction=prose["introduction"],
            methods_tex=methods_tex,
            results_tex=results_tex,
            figures_tex=figures_tex,
            discussion=prose["discussion"],
            conclusion=prose["conclusion"],
            references_tex=references_tex,
        )

        body_path = tmp / "methods_body.tex"
        body_path.write_text(methods_tex, encoding="utf-8")
        main_path = tmp / "main.tex"
        main_path.write_text(full_tex, encoding="utf-8")

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
        "results_latex": results_tex,
        "full_tex_chars": len(full_tex),
        "figures_embedded": len(figure_files),
    }
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_READY,
        step="paper.ready",
        payload=payload,
    )
    return payload
