from __future__ import annotations

import json
import re
from itertools import groupby
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker


def _latex_escape(s: str) -> str:
    """Escape minimal LaTeX special characters in user/tool-derived text."""
    if not s:
        return ""
    greek_map = {
        "α": r"$\alpha$",
        "β": r"$\beta$",
        "γ": r"$\gamma$",
        "δ": r"$\delta$",
        "η": r"$\eta$",
        "σ": r"$\sigma$",
        "λ": r"$\lambda$",
        "μ": r"$\mu$",
    }
    out: list[str] = []
    for ch in s:
        if ch in greek_map:
            out.append(greek_map[ch])
            continue
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
        blob = json.dumps(params, ensure_ascii=False)
    else:
        blob = str(params)
    return blob[:280] + ("..." if len(blob) > 280 else "")


def _latex_cell_scalarish(val: Any, max_len: int = 200) -> str:
    """Single table cell: escape; nested structures as compact JSON."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return _latex_escape(str(val))
    if isinstance(val, str):
        s = val.strip()
        return _latex_escape(s[:max_len] + ("..." if len(s) > max_len else ""))
    try:
        blob = json.dumps(val, ensure_ascii=False)
    except TypeError:
        blob = str(val)
    return _latex_escape(blob[:max_len] + ("..." if len(blob) > max_len else ""))


def latex_tabular_from_jsonish(obj: Any, *, max_rows: int = 28, max_cols: int = 10) -> str | None:
    """
    Render dict / list-of-dicts / numeric list as a LaTeX ``tabular`` (article-safe, no extra packages).
    Returns None when a plain prose/JSON one-liner is more appropriate.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        if not obj:
            return None
        items = list(obj.items())[:max_rows]
        lines = [
            "\\begin{tabular}{|l|l|}",
            "\\hline",
            r"\textbf{Field} & \textbf{Value} \\",
            "\\hline",
        ]
        for k, v in items:
            lines.append(f"{_latex_escape(str(k))} & {_latex_cell_scalarish(v, max_len=360)} \\\\")
        lines.extend(["\\hline", "\\end{tabular}"])
        return "\n".join(lines)
    if isinstance(obj, list) and obj:
        if all(isinstance(x, (int, float)) for x in obj) and len(obj) >= 3:
            rows = list(enumerate(obj))[:max_rows]
            lines = [
                "\\begin{tabular}{|r|l|}",
                "\\hline",
                r"\textbf{Index} & \textbf{Value} \\",
                "\\hline",
            ]
            for i, v in rows:
                lines.append(f"{i} & {_latex_cell_scalarish(v, max_len=80)} \\\\")
            lines.extend(["\\hline", "\\end{tabular}"])
            return "\n".join(lines)
        if all(isinstance(x, dict) for x in obj) and obj:
            sample = [d for d in obj[:12] if isinstance(d, dict) and d]
            if not sample:
                return None
            keys: list[str] = []
            for d in sample:
                for kk in d:
                    if kk not in keys:
                        keys.append(kk)
            keys = keys[:max_cols]
            if not keys:
                return None
            colspec = "|" + "|".join(["l"] * len(keys)) + "|"
            lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\hline"]
            lines.append(" & ".join(_latex_escape(str(k)) for k in keys) + r" \\")
            lines.append("\\hline")
            for d in obj[:max_rows]:
                if not isinstance(d, dict):
                    continue
                cells = [_latex_cell_scalarish(d.get(k), max_len=140) for k in keys]
                lines.append(" & ".join(cells) + r" \\")
            lines.extend(["\\hline", "\\end{tabular}"])
            return "\n".join(lines)
    return None


def _tabular_payload_from_tool_output(out: Any) -> Any:
    """Prefer inner ``output`` / ``data`` for table rendering when the DB row is a wrapper dict."""
    if not isinstance(out, dict) or not out:
        return out
    inner = out.get("output")
    if isinstance(inner, dict) and inner:
        return inner
    if isinstance(inner, list) and inner:
        return inner
    data = out.get("data")
    if isinstance(data, dict) and data:
        return data
    if isinstance(data, list) and data:
        return data
    return out


def _format_evidence_for_paper(evidence_summary: str | None) -> str:
    text = (evidence_summary or "").strip()
    if not text:
        return ""
    m = re.search(r"evidence=(\{[\s\S]*?\})\s*;", text)
    if m:
        try:
            ev = json.loads(m.group(1))
            p = ev.get("p_value")
            e = ev.get("effect_size")
            ci = ev.get("confidence_interval")
            p_s = f"{float(p):.6f}" if isinstance(p, (float, int)) else "n/a"
            e_s = f"{float(e):.4f}" if isinstance(e, (float, int)) else "n/a"
            if isinstance(ci, list) and len(ci) >= 2 and all(isinstance(x, (float, int)) for x in ci[:2]):
                ci_s = f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"
            else:
                ci_s = "n/a"
            return f"p={p_s}, effect={e_s}, CI={ci_s}"
        except Exception:
            pass
    return text[:220] + ("..." if len(text) > 220 else "")


def _extract_metric_steps_from_evidence(evidence_summary: str | None) -> int | None:
    text = (evidence_summary or "").strip()
    if not text:
        return None
    # New format: evidence={...}; plan_origin=...
    m = re.search(r"evidence=(\{[\s\S]*?\})\s*;", text)
    if m:
        try:
            obj = json.loads(m.group(1))
            v = obj.get("n_metric_steps")
            if isinstance(v, int):
                return v
        except json.JSONDecodeError:
            pass
    # Legacy fallback: n_metric_steps=<int>
    m2 = re.search(r"n_metric_steps\s*=\s*(\d+)", text)
    if m2:
        return int(m2.group(1))
    return None


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
            out_json = row.get("output_json")
            out_for_table = _tabular_payload_from_tool_output(out_json)
            tab_p = latex_tabular_from_jsonish(params)
            tab_o = latex_tabular_from_jsonish(out_for_table) if out_json not in (None, {}, []) else None
            if tab_p:
                lines.append(
                    f"Tool \\texttt{{{tool_name}}} (completed in {duration_ms}ms). Parameters:\\\\\n"
                    f"{tab_p}"
                )
            else:
                lines.append(
                    f"Tool \\texttt{{{tool_name}}} was called with parameters {_latex_escape(_format_params(params))}. "
                    f"Execution completed in {duration_ms}ms."
                )
            if tab_o:
                lines.append("Recorded tool output:\\\\\n" + tab_o)
            elif out_json not in (None, {}, []):
                blob = (
                    json.dumps(out_json, ensure_ascii=False)
                    if not isinstance(out_json, str)
                    else out_json
                )
                lines.append(
                    "Tool output (summary): "
                    + _latex_escape(str(blob)[:400] + ("..." if len(str(blob)) > 400 else ""))
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


def _human_round_num(round_number_0_indexed: int | None) -> int:
    """DB stores round_number starting at 0; paper displays Round 1, Round 2, …"""
    base = 0 if round_number_0_indexed is None else int(round_number_0_indexed)
    return base + 1


async def compile_session_methods_latex(session_factory: async_sessionmaker, session_id: str) -> dict[str, Any]:
    async with session_factory() as session:
        hyp_rows = (
            await session.execute(
                text(
                    """
                    SELECT h.id, h.text, h.rank,
                           COALESCE(r.round_number, 0) AS round_number
                    FROM hypotheses h
                    LEFT JOIN research_rounds r ON r.id = h.round_id
                    WHERE h.session_id = CAST(:session_id AS uuid)
                    ORDER BY COALESCE(r.round_number, 0) ASC,
                             h.rank ASC NULLS LAST,
                             h.created_at ASC
                    """
                ),
                {"session_id": session_id},
            )
        ).mappings().all()

    sections: list[str] = []
    per_hypothesis: dict[str, str] = {}
    if not hyp_rows:
        return {
            "combined_latex": "\\section{Methods}\nNo hypotheses were executed.\n",
            "per_hypothesis": {},
        }

    sections.append("\\section{Methods}")
    for rn_key, group_iter in groupby(hyp_rows, key=lambda row: int(row["round_number"])):
        group = list(group_iter)
        hr = _human_round_num(rn_key)
        if hr >= 2:
            sections.append(
                f"\\subsection{{Round {hr} hypotheses \\textit{{(informed by prior rounds)}}}}\n"
            )
        else:
            sections.append(f"\\subsection{{Round {hr} hypotheses}}\n")

        for hyp in group:
            hid = str(hyp["id"])
            body = await compile_methods_section(session_factory, hid)
            label_rank = hyp["rank"] if hyp.get("rank") is not None else "?"
            hyp_text = _latex_escape(str(hyp["text"] or ""))
            short_head = hyp_text[:100] + ("..." if len(hyp_text) > 100 else "")
            title = _latex_escape(f"Round {hr} -- Hypothesis {label_rank}: {short_head}")
            has_steps = "No automated experiment steps" not in body
            if not has_steps:
                section = (
                    f"\\subsubsection{{{title}}}\\label{{subsec:{hid}}}\n"
                    f"\\textit{{{hyp_text}}}\\\\\n"
                    "\\textit{{Note: No experiment steps were executed for this hypothesis; "
                    "excluded from methods and results analysis.}}\n"
                )
            else:
                section = (
                    f"\\subsubsection{{{title}}}\\label{{subsec:{hid}}}\n"
                    f"\\textit{{{hyp_text}}}\\\\\n"
                    f"{body}\n"
                )
            sections.append(section)
            per_hypothesis[hid] = body

    combined = "\n".join(sections)
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


def _results_block_for_row(row: Any) -> str:
    """One hypothesis result block (subsubsection body) with shared formatting."""
    label_rank = row["rank"] if row.get("rank") is not None else str(row["id"])[:8]
    hr = _human_round_num(row.get("round_number"))
    title = _latex_escape(f"Round {hr} -- Hypothesis {label_rank}")
    raw_verdict = str(row.get("verdict") or "pending")
    verdict = _latex_escape(raw_verdict)
    conf = row.get("confidence")
    conf_s = f"{float(conf):.2f}" if conf is not None else "n/a"
    ev = _latex_escape(_format_evidence_for_paper(str(row.get("evidence_summary") or "")))
    kf = row.get("key_finding")
    kf_s = _latex_escape(str(kf))[:500] if kf else ""
    trace = _latex_escape(str(row.get("tool_trace_id") or ""))
    hyp = _latex_escape(str(row.get("text") or "")[:900])
    steps = int(row.get("step_count") or 0)
    n_metric_steps = _extract_metric_steps_from_evidence(str(row.get("evidence_summary") or ""))
    if steps == 0:
        return (
            f"\\subsubsection{{{title}}}\n"
            f"\\textbf{{Statement:}} {hyp}\\\\\n"
            "\\textbf{Status:} excluded from analysis (no experiment steps executed).\\\\\n"
            "\\textbf{Note:} This hypothesis was generated but never executed, so it is not included in conclusions."
        )
    if raw_verdict.lower() == "confirmed" and n_metric_steps == 0:
        return (
            f"\\subsubsection{{{title}}}\n"
            f"\\textbf{{Statement:}} {hyp}\\\\\n"
            "\\textbf{Status:} excluded from analysis (confirmed without metric evidence).\\\\\n"
            "\\textbf{Note:} Confirmation requires metric-bearing evidence; this row is excluded pending rerun."
        )
    return (
        f"\\subsubsection{{{title}}}\n"
        f"\\textbf{{Statement:}} {hyp}\\\\\n"
        f"\\textbf{{Verdict:}} {verdict} (confidence {conf_s}). "
        f"Recorded {steps} experiment step(s).\\\\\n"
        f"\\textbf{{Evidence:}} {ev}\\\\\n"
        + (f"\\textbf{{Key finding:}} {kf_s}\\\\\n" if kf_s else "")
        + (f"\\textbf{{Trace id:}} \\texttt{{{trace}}}\n" if trace else "")
    )


async def compile_session_results_latex(session_factory: async_sessionmaker, session_id: str) -> str:
    """Deterministic results from hypothesis verdicts and step counts (no LLM)."""
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT h.id, h.rank, h.text, h.verdict, h.confidence, h.evidence_summary, h.key_finding,
                           h.tool_trace_id,
                           COALESCE(r.round_number, 0) AS round_number,
                           (SELECT COUNT(*) FROM experiment_steps e WHERE e.hypothesis_id = h.id) AS step_count
                    FROM hypotheses h
                    LEFT JOIN research_rounds r ON r.id = h.round_id
                    WHERE h.session_id = CAST(:session_id AS uuid)
                    ORDER BY COALESCE(r.round_number, 0) ASC,
                             h.rank ASC NULLS LAST,
                             h.created_at ASC
                    """
                ),
                {"session_id": session_id},
            )
        ).mappings().all()

    if not rows:
        return "\\section{Results}\nNo hypotheses were recorded for this session.\n"

    parts: list[str] = ["\\section{Results}"]
    for rn_key, group_iter in groupby(rows, key=lambda row: int(row["round_number"])):
        group = list(group_iter)
        hr = _human_round_num(rn_key)
        if hr >= 2:
            parts.append(
                f"\\subsection{{Round {hr} results \\textit{{(following Round {hr - 1})}}}}\n"
            )
        else:
            parts.append(f"\\subsection{{Round {hr} results}}\n")
        for row in group:
            parts.append(_results_block_for_row(row))
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
