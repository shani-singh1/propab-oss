from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker


def _latex_escape(s: str) -> str:
    """Escape minimal LaTeX special characters in user/tool-derived text."""
    if not s:
        return ""
    out: list[str] = []
    for ch in s:
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "&":
            out.append(r"\&")
        elif ch == "%":
            out.append(r"\%")
        elif ch == "$":
            out.append(r"\$")
        elif ch == "#":
            out.append(r"\#")
        elif ch == "_":
            out.append(r"\_")
        elif ch == "{":
            out.append(r"\{")
        elif ch == "}":
            out.append(r"\}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


def _format_params(params: Any) -> str:
    if params is None:
        return "{}"
    if isinstance(params, dict):
        return json.dumps(params, ensure_ascii=False)
    return str(params)


async def compile_methods_section(session_factory: async_sessionmaker, hypothesis_id: str) -> str:
    """
    Deterministic methods text from experiment_steps (no LLM), per architecture.
    """
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT step_type, input_json, output_json, duration_ms, memory_mb, timeout_sec, summary
                    FROM experiment_steps
                    WHERE hypothesis_id = :hypothesis_id
                    ORDER BY created_at ASC
                    """
                ),
                {"hypothesis_id": hypothesis_id},
            )
        ).mappings().all()

    lines: list[str] = []
    for row in rows:
        step_type = row["step_type"]
        if step_type == "tool_call":
            input_data = row["input_json"] or {}
            tool_name = _latex_escape(str(input_data.get("tool", "unknown_tool")))
            params = input_data.get("params")
            duration_ms = row["duration_ms"] or 0
            lines.append(
                f"Tool \\texttt{{{tool_name}}} was called with parameters {_latex_escape(_format_params(params))}. "
                f"Execution completed in {duration_ms}ms."
            )
        elif step_type == "code_exec":
            mem = row["memory_mb"] if row["memory_mb"] is not None else "512"
            timeout = row["timeout_sec"] if row["timeout_sec"] is not None else "30"
            lines.append(
                f"Custom computation was executed in an isolated sandbox "
                f"(memory limit: {mem}MB, timeout: {timeout}s). "
                f"Code is available in the experiment trace for this hypothesis."
            )
        elif step_type == "llm_reasoning":
            summary = _latex_escape(str(row["summary"] or "intermediate evaluation"))
            lines.append(f"The agent evaluated intermediate results and \\textit{{{summary}}}.")
    return "\n".join(lines) if lines else "No automated experiment steps were recorded for this hypothesis."


async def compile_session_methods_latex(session_factory: async_sessionmaker, session_id: str) -> dict[str, Any]:
    async with session_factory() as session:
        hyp_rows = (
            await session.execute(
                text(
                    """
                    SELECT id, text, rank
                    FROM hypotheses
                    WHERE session_id = :session_id
                    ORDER BY rank ASC
                    """
                ),
                {"session_id": session_id},
            )
        ).mappings().all()

    sections: list[str] = []
    per_hypothesis: dict[str, str] = {}
    for hyp in hyp_rows:
        hid = str(hyp["id"])
        body = await compile_methods_section(session_factory, hid)
        label = f"h{hyp['rank']}"
        hyp_text = _latex_escape(str(hyp["text"] or ""))
        section = (
            f"\\subsection{{Hypothesis {label}}}\\label{{subsec:{hid}}}\n"
            f"\\textit{{{hyp_text}}}\\\\\n"
            f"{body}\n"
        )
        sections.append(section)
        per_hypothesis[hid] = body

    combined = "\\section{Methods}\n" + "\n".join(sections) if sections else "\\section{Methods}\nNo hypotheses were executed.\n"
    return {"combined_latex": combined, "per_hypothesis": per_hypothesis}


def collect_figure_object_ids(synthesis: dict[str, Any] | None) -> list[str]:
    """MinIO object paths attached to experiment results (deduped, stable order)."""
    if not synthesis:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for row in synthesis.get("experiment_results") or []:
        if not isinstance(row, dict):
            continue
        for fig in row.get("figures") or []:
            if isinstance(fig, str):
                oid = fig.strip()
                if oid and oid not in seen:
                    seen.add(oid)
                    out.append(oid)
    return out


async def compile_session_results_latex(session_factory: async_sessionmaker, session_id: str) -> str:
    """Deterministic results from hypothesis verdicts and step counts (no LLM)."""
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT h.id, h.rank, h.text, h.verdict, h.confidence, h.evidence_summary, h.key_finding,
                           h.tool_trace_id,
                           (SELECT COUNT(*) FROM experiment_steps e WHERE e.hypothesis_id = h.id) AS step_count
                    FROM hypotheses h
                    WHERE h.session_id = :session_id
                    ORDER BY h.rank ASC NULLS LAST, h.created_at ASC
                    """
                ),
                {"session_id": session_id},
            )
        ).mappings().all()

    if not rows:
        return "\\section{Results}\nNo hypotheses were recorded for this session.\n"

    parts: list[str] = ["\\section{Results}"]
    for row in rows:
        label = f"h{row['rank']}" if row.get("rank") is not None else str(row["id"])[:8]
        verdict = _latex_escape(str(row.get("verdict") or "pending"))
        conf = row.get("confidence")
        conf_s = f"{float(conf):.2f}" if conf is not None else "n/a"
        ev = _latex_escape(str(row.get("evidence_summary") or "")[:1200])
        kf = row.get("key_finding")
        kf_s = _latex_escape(str(kf))[:500] if kf else ""
        trace = _latex_escape(str(row.get("tool_trace_id") or ""))
        hyp = _latex_escape(str(row.get("text") or "")[:900])
        steps = int(row.get("step_count") or 0)
        parts.append(
            f"\\subsection{{Hypothesis {label}}}\n"
            f"\\textbf{{Statement:}} {hyp}\\\\\n"
            f"\\textbf{{Verdict:}} {verdict} (confidence {conf_s}). "
            f"Recorded {steps} experiment step(s).\\\\\n"
            f"\\textbf{{Evidence:}} {ev}\\\\\n"
            + (f"\\textbf{{Key finding:}} {kf_s}\\\\\n" if kf_s else "")
            + (f"\\textbf{{Trace id:}} \\texttt{{{trace}}}\n" if trace else "")
        )
    return "\n".join(parts)


def compile_references_latex(prior: dict[str, Any] | None) -> str:
    """Manual bibliography from prior key papers (no BibTeX engine required)."""
    prior = prior or {}
    papers = prior.get("key_papers") or []
    if not isinstance(papers, list) or not papers:
        return "\\section{References}\nNo seed papers were attached to the prior for this session.\n"

    lines: list[str] = ["\\section{References}", "\\begin{thebibliography}{99}"]
    for i, p in enumerate(papers[:48]):
        if not isinstance(p, dict):
            continue
        pid = _latex_escape(str(p.get("paper_id") or f"ref{i}"))
        title = _latex_escape(str(p.get("title") or "Untitled"))[:400]
        summ = _latex_escape(str(p.get("summary") or ""))[:280]
        lines.append(f"\\bibitem{{{pid}}}{title}. \\textit{{Abstract excerpt:}} {summ}")
    lines.append("\\end{thebibliography}")
    return "\n".join(lines)
