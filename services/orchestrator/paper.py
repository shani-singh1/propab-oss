from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.claim_grounding import ground_session_claims
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.paper_compiler import (
    collect_figure_object_ids,
    compile_references_latex,
    compile_session_findings,
    compile_session_methods_latex,
    compile_session_results_latex,
)
from propab.paper_narrative import build_figure_specs, research_narrative_section
from propab.paper_sections import generate_prose_sections, render_paper_tex
from propab.storage import get_object_bytes, put_bytes
from propab.types import EventType


async def _session_experiment_step_count(session_factory: async_sessionmaker, session_id: str) -> int:
    async with session_factory() as session:
        count = (
            await session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM experiment_steps e
                    JOIN hypotheses h ON h.id = e.hypothesis_id
                    WHERE h.session_id = CAST(:sid AS uuid)
                    """
                ),
                {"sid": session_id},
            )
        ).scalar_one()
    return int(count or 0)


def _ensure_nonempty_trace(step_count: int) -> None:
    if step_count <= 0:
        raise RuntimeError(
            "Paper generation blocked: session has zero experiment steps. "
            "This indicates experiments did not execute; returning honest failure instead of speculative paper."
        )


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
        blocks.append(
            "\\begin{figure}[ht]\n\\centering\n"
            f"\\includegraphics[width=0.88\\linewidth]{{{fname}}}\n"
            f"\\caption{{Experimental result figure {i + 1}, generated during automated experimentation.}}\n"
            f"\\label{{fig:result-{i + 1}}}\n"
            "\\end{figure}\n\n"
        )
    return ("".join(blocks), written)


def _render_figure_spec_png(spec: dict[str, Any], path: Path) -> bool:
    """Render one gated figure spec to a PNG at ``path``. Returns True on success.

    All numbers plotted come straight from the spec's ``data`` block, which the
    narrative module built from the gated findings — so nothing is imputed or
    invented at plot time.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 — plotting is best-effort; paper still renders
        return False

    kind = spec.get("kind")
    data = spec.get("data") or {}
    try:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        if kind == "outcome_bars":
            labels = data.get("labels") or []
            values = data.get("values") or []
            colors = ["#2e7d32", "#c62828", "#f9a825"][: len(labels)]
            ax.bar(labels, values, color=colors)
            ax.set_ylabel("Number of hypotheses")
            for i, v in enumerate(values):
                ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
        elif kind == "metric_vs_baseline":
            points = data.get("points") or []
            names = [p.get("label", "") for p in points]
            vals = [float(p.get("value", 0.0)) for p in points]
            ypos = list(range(len(vals)))
            ax.barh(ypos, vals, color="#1565c0")
            ax.set_yticks(ypos)
            ax.set_yticklabels(names, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel(str(data.get("metric_name") or "metric"))
            base = data.get("baseline")
            if isinstance(base, (int, float)):
                ax.axvline(float(base), color="#c62828", linestyle="--", label=f"baseline={float(base):.3g}")
                ax.legend(fontsize=8)
        elif kind == "rounds_timeline":
            gens = data.get("generations") or []
            for key, color in (("confirmed", "#2e7d32"), ("refuted", "#c62828"), ("inconclusive", "#f9a825")):
                series = data.get(key) or []
                if any(series):
                    ax.plot(gens, series, marker="o", label=key, color=color)
            ax.set_xlabel("Round")
            ax.set_ylabel("Outcomes")
            ax.legend(fontsize=8)
            if gens:
                ax.set_xticks(gens)
        elif kind == "lineage_tree":
            _draw_lineage(ax, data)
        else:
            plt.close(fig)
            return False
        ax.set_title(str(spec.get("title") or ""), fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path.exists()
    except Exception:  # noqa: BLE001
        try:
            plt.close("all")
        except Exception:  # noqa: BLE001
            pass
        return False


def _draw_lineage(ax: Any, data: dict[str, Any]) -> None:
    """Simple layered layout of the hypothesis lineage, coloured by gated verdict."""
    edges = data.get("edges") or []
    verdicts = data.get("verdicts") or {}
    # Build adjacency and depth (BFS from roots = nodes never appearing as a child).
    children: dict[str, list[str]] = {}
    parents: dict[str, str] = {}
    nodes: set[str] = set()
    for e in edges:
        p, c = str(e.get("parent")), str(e.get("child"))
        children.setdefault(p, []).append(c)
        parents[c] = p
        nodes.add(p)
        nodes.add(c)
    roots = [n for n in nodes if n not in parents]
    depth: dict[str, int] = {}
    order: list[str] = []
    stack = [(r, 0) for r in roots]
    while stack:
        n, d = stack.pop(0)
        if n in depth:
            continue
        depth[n] = d
        order.append(n)
        for c in children.get(n, []):
            stack.append((c, d + 1))
    # x by depth, y by insertion order within depth
    per_depth: dict[int, int] = {}
    pos: dict[str, tuple[float, float]] = {}
    for n in order:
        d = depth.get(n, 0)
        y = per_depth.get(d, 0)
        per_depth[d] = y + 1
        pos[n] = (float(d), -float(y))
    color_map = {"confirmed": "#2e7d32", "refuted": "#c62828", "inconclusive": "#f9a825", "explored": "#90a4ae"}
    for e in edges:
        p, c = str(e.get("parent")), str(e.get("child"))
        if p in pos and c in pos:
            x0, y0 = pos[p]
            x1, y1 = pos[c]
            ax.plot([x0, x1], [y0, y1], color="#b0bec5", linewidth=0.8, zorder=1)
    for n, (x, y) in pos.items():
        v = verdicts.get(n, "explored")
        ax.scatter([x], [y], s=90, color=color_map.get(v, "#90a4ae"), zorder=2, edgecolors="white", linewidths=0.5)
    handles = [
        plt_line(color_map[k], k)
        for k in ("confirmed", "refuted", "inconclusive", "explored")
        if any(verdicts.get(n) == k for n in pos)
    ]
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.set_xlabel("Lineage depth")
    ax.set_yticks([])
    ax.set_xticks(sorted({int(x) for x, _ in pos.values()}))


def plt_line(color: str, label: str) -> Any:
    from matplotlib.lines import Line2D

    return Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label)


def _build_derived_figures(
    *,
    tmp: Path,
    findings: dict[str, Any],
    reasoning_trace: dict[str, Any] | None,
    baseline: float | None,
    metric_name: str,
) -> tuple[str, list[str]]:
    """Render figure specs (from the gated findings) to PNGs and return LaTeX + filenames."""
    specs = build_figure_specs(
        findings,
        reasoning_trace=reasoning_trace,
        baseline=baseline,
        metric_name=metric_name,
    )
    blocks: list[str] = []
    written: list[str] = []
    idx = 0
    for spec in specs:
        fname = f"derived_fig_{idx}_{re.sub(r'[^a-z0-9]+', '_', str(spec.get('kind') or 'fig'))}.png"
        if not _render_figure_spec_png(spec, tmp / fname):
            continue
        written.append(fname)
        caption = spec.get("caption") or spec.get("title") or ""
        blocks.append(
            "\\begin{figure}[ht]\n\\centering\n"
            f"\\includegraphics[width=0.82\\linewidth]{{{fname}}}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{fig:derived-{idx}}}\n"
            "\\end{figure}\n\n"
        )
        idx += 1
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
    step_count = await _session_experiment_step_count(session_factory, session_id)
    literature_only = bool((synthesis or {}).get("short_circuit"))
    if step_count <= 0 and not literature_only:
        _ensure_nonempty_trace(step_count)
    elif step_count <= 0 and literature_only:
        # Literature short-circuit may have no experiment rows; still emit a minimal note PDF.
        pass

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

    # Single gated evaluation — the ONE source of truth reused by results, tables,
    # figures, the narrative, and the abstract counts, so no section can disagree.
    synthesis = dict(synthesis or {})
    findings: dict[str, Any] | None = None
    baseline = synthesis.get("baseline_metric")
    baseline = float(baseline) if isinstance(baseline, (int, float)) else None
    metric_name = str(synthesis.get("metric_name") or "metric").replace("_", " ")
    if not literature_only:
        findings = await compile_session_findings(session_factory, session_id)
        synthesis["counts"] = findings["counts"]
        synthesis["confirmed_findings"] = findings["confirmed"]

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_SECTION_STARTED,
        step="paper.results",
        payload={"section": "results"},
    )
    results_tex = await compile_session_results_latex(
        session_factory, session_id, baseline=baseline, findings=findings
    )
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_SECTION_COMPLETED,
        step="paper.results",
        payload={"section": "results", "chars": len(results_tex)},
    )

    references_tex = compile_references_latex(prior or {})

    methods_tex = report["combined_latex"]
    q = (question or "").strip() or "Research session"

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.PAPER_SECTION_STARTED,
        step="paper.prose",
        payload={"section": "title_abstract_intro_discussion_conclusion"},
    )
    prose = await generate_prose_sections(
        llm=llm,
        session_id=session_id,
        question=q,
        prior=prior,
        synthesis=synthesis,
    )
    title = (prose.get("title") or q).replace("\n", " ").strip()[:180]
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

    # Chain-of-Reasoning / Research Narrative — deterministic, grounded in the gated
    # findings and the trace-derived reasoning context; never invents an outcome.
    reasoning_trace = synthesis.get("reasoning_trace") if isinstance(synthesis, dict) else None
    narrative_tex = ""
    if findings is not None:
        narrative_tex = research_narrative_section(
            findings, reasoning_trace=reasoning_trace, question=q
        )
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.PAPER_SECTION_COMPLETED,
            step="paper.narrative",
            payload={"section": "research_narrative", "chars": len(narrative_tex)},
        )

    pdf_url = None
    tex_url = None
    latex_ok = False
    latex_log = ""
    figure_files: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Figures embedded from experiment artifacts (existing behaviour) …
        figures_tex, figure_files = _build_figures_tex_and_files(tmp=tmp, synthesis=synthesis)
        # … plus figures we derive from the SAME gated findings (matplotlib).
        if findings is not None:
            derived_tex, derived_files = _build_derived_figures(
                tmp=tmp,
                findings=findings,
                reasoning_trace=reasoning_trace,
                baseline=baseline,
                metric_name=metric_name,
            )
            figures_tex = derived_tex + figures_tex
            figure_files = derived_files + figure_files

        full_tex = render_paper_tex(
            title=title,
            abstract=prose["abstract"],
            introduction=prose["introduction"],
            methods_tex=methods_tex,
            results_tex=results_tex,
            figures_tex=figures_tex,
            narrative_tex=narrative_tex,
            discussion=prose["discussion"],
            conclusion=prose["conclusion"],
            references_tex=references_tex,
        )

        body_path = tmp / "methods_body.tex"
        body_path.write_text(methods_tex, encoding="utf-8")
        main_path = tmp / "main.tex"
        main_path.write_text(full_tex, encoding="utf-8")

        if shutil.which("pdflatex"):
            # Preflight lint in draft mode to catch escaping issues early.
            pre = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-draftmode", "main.tex"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if pre.returncode == 0:
                proc = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                latex_ok = proc.returncode == 0 and (tmp / "main.pdf").exists()
                latex_log = (pre.stdout or "") + "\n" + (pre.stderr or "") + "\n" + (proc.stdout or "") + "\n" + (proc.stderr or "")
            else:
                latex_ok = False
                latex_log = (pre.stdout or "") + "\n" + (pre.stderr or "")
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
        "abstract_latex": prose["abstract"],
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
