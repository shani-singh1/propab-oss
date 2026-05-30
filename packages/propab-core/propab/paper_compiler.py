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


def parse_evidence(evidence_summary: str | None) -> dict[str, Any]:
    """
    Structured statistical evidence from the worker's ``evidence_summary`` string.

    Returns a dict with p_value, effect_size, confidence_interval, n_metric_steps,
    metric_value (any of which may be None). This is the single parser used by both
    the abstract (headline statistics) and the results table, so the two never diverge.
    """
    out: dict[str, Any] = {
        "p_value": None,
        "effect_size": None,
        "confidence_interval": None,
        "n_metric_steps": None,
        "metric_value": None,
    }
    raw = (evidence_summary or "").strip()
    if not raw:
        return out
    m = re.search(r"evidence=(\{[\s\S]*?\})\s*;", raw)
    if m:
        try:
            ev = json.loads(m.group(1))
            for k in ("p_value", "effect_size", "n_metric_steps", "metric_value"):
                if isinstance(ev.get(k), (int, float)):
                    out[k] = ev[k]
            ci = ev.get("confidence_interval")
            if isinstance(ci, list) and len(ci) >= 2 and all(isinstance(x, (int, float)) for x in ci[:2]):
                out["confidence_interval"] = [float(ci[0]), float(ci[1])]
            return out
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    m2 = re.search(r"n_metric_steps\s*=\s*(\d+)", raw)
    if m2:
        out["n_metric_steps"] = int(m2.group(1))
    return out


def format_stats(ev: dict[str, Any]) -> str:
    """Render available statistics as a compact LaTeX math fragment (omits missing pieces)."""
    parts: list[str] = []
    p = ev.get("p_value")
    if isinstance(p, (int, float)):
        parts.append(f"$p = {float(p):.3g}$")
    e = ev.get("effect_size")
    if isinstance(e, (int, float)):
        parts.append(f"Cohen's $d = {float(e):.2f}$")
    ci = ev.get("confidence_interval")
    if isinstance(ci, list) and len(ci) >= 2:
        parts.append(f"95\\% CI $[{ci[0]:.3g}, {ci[1]:.3g}]$")
    return ", ".join(parts) if parts else "no inferential statistic recorded"


def _effective_verdict(row: Any) -> str:
    """
    The verdict a reader should see, after applying the platform's evidence bar.

    This is the SINGLE source of truth for counting outcomes, so the abstract,
    the results section, and any summary always agree:
      - never executed                  -> "unexecuted" (excluded from all counts)
      - DB says confirmed, no metric    -> "inconclusive" (cannot claim without evidence)
      - otherwise                       -> the recorded verdict (defaulting to inconclusive)
    """
    steps = int(row.get("step_count") or 0)
    if steps == 0:
        return "unexecuted"
    raw = str(row.get("verdict") or "").strip().lower()
    ev = parse_evidence(str(row.get("evidence_summary") or ""))
    n_metric = ev.get("n_metric_steps")
    if raw == "confirmed" and not n_metric:
        return "inconclusive"
    if raw in ("confirmed", "refuted", "inconclusive"):
        return raw
    return "inconclusive"


def _human_round_num(round_number_0_indexed: int | None) -> int:
    """DB stores round_number starting at 0; paper displays Round 1, Round 2, …"""
    base = 0 if round_number_0_indexed is None else int(round_number_0_indexed)
    return base + 1


async def _fetch_result_rows(session_factory: async_sessionmaker, session_id: str) -> list[dict[str, Any]]:
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT h.id, h.rank, h.text, h.verdict, h.confidence, h.evidence_summary,
                           h.key_finding, h.tool_trace_id,
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
    return [dict(r) for r in rows]


async def compile_session_findings(session_factory: async_sessionmaker, session_id: str) -> dict[str, Any]:
    """
    Authoritative, DB-derived view of session outcomes. Used to keep the abstract,
    the results section, and the synthesis ledger perfectly consistent.

    Returns ``counts`` (confirmed/refuted/inconclusive/tested/unexecuted) and the
    classified findings, each with parsed statistics, ordered by confidence.
    """
    rows = await _fetch_result_rows(session_factory, session_id)
    buckets: dict[str, list[dict[str, Any]]] = {"confirmed": [], "refuted": [], "inconclusive": []}
    unexecuted = 0
    for row in rows:
        verdict = _effective_verdict(row)
        if verdict == "unexecuted":
            unexecuted += 1
            continue
        ev = parse_evidence(str(row.get("evidence_summary") or ""))
        buckets[verdict].append(
            {
                "id": str(row.get("id")),
                "rank": row.get("rank"),
                "round": _human_round_num(row.get("round_number")),
                "text": str(row.get("text") or "").strip(),
                "key_finding": str(row.get("key_finding") or "").strip(),
                "confidence": float(row.get("confidence")) if row.get("confidence") is not None else None,
                "stats": ev,
                "stats_text": format_stats(ev),
            }
        )
    for b in buckets.values():
        b.sort(key=lambda f: (f.get("confidence") or 0.0), reverse=True)
    counts = {
        "confirmed": len(buckets["confirmed"]),
        "refuted": len(buckets["refuted"]),
        "inconclusive": len(buckets["inconclusive"]),
        "unexecuted": unexecuted,
    }
    counts["tested"] = counts["confirmed"] + counts["refuted"] + counts["inconclusive"]
    return {"counts": counts, **buckets}


async def _aggregate_tool_usage(session_factory: async_sessionmaker, session_id: str) -> dict[str, Any]:
    """Tool-usage and round counts for the deterministic Methods narrative."""
    async with session_factory() as session:
        tool_rows = (
            await session.execute(
                text(
                    """
                    SELECT COALESCE(tc.tool_name, e.input_json->>'tool') AS tool_name,
                           COUNT(*) AS n
                    FROM experiment_steps e
                    JOIN hypotheses h ON h.id = e.hypothesis_id
                    LEFT JOIN tool_calls tc ON tc.step_id = e.id
                    WHERE h.session_id = CAST(:sid AS uuid) AND e.step_type = 'tool_call'
                    GROUP BY 1
                    ORDER BY n DESC
                    """
                ),
                {"sid": session_id},
            )
        ).mappings().all()
        rounds = (
            await session.execute(
                text(
                    """
                    SELECT COUNT(DISTINCT COALESCE(r.round_number, 0))
                    FROM hypotheses h
                    LEFT JOIN research_rounds r ON r.id = h.round_id
                    WHERE h.session_id = CAST(:sid AS uuid)
                    """
                ),
                {"sid": session_id},
            )
        ).scalar_one()
        code_steps = (
            await session.execute(
                text(
                    """
                    SELECT COUNT(*) FROM experiment_steps e
                    JOIN hypotheses h ON h.id = e.hypothesis_id
                    WHERE h.session_id = CAST(:sid AS uuid) AND e.step_type = 'code_exec'
                    """
                ),
                {"sid": session_id},
            )
        ).scalar_one()
    tools = {str(r["tool_name"]): int(r["n"]) for r in tool_rows if r["tool_name"]}
    return {"tools": tools, "rounds": int(rounds or 1), "code_steps": int(code_steps or 0)}


_SIGNIFICANCE_TOOLS = {
    "statistical_significance",
    "bootstrap_confidence",
    "literature_baseline_compare",
}


async def compile_session_methods_latex(session_factory: async_sessionmaker, session_id: str) -> dict[str, Any]:
    """
    Deterministic Methods section, compiled from the experiment trace (no LLM).

    Produces a readable methodological narrative rather than a per-call log dump:
    the experimental protocol, the instruments used (with usage counts), and how
    statistical support was established. The full step-level trace remains queryable
    via the API for replication, but is not dumped into the manuscript.
    """
    findings = await compile_session_findings(session_factory, session_id)
    counts = findings["counts"]
    agg = await _aggregate_tool_usage(session_factory, session_id)
    tools = agg["tools"]

    if counts["tested"] == 0 and not tools:
        return {
            "combined_latex": (
                "\\section{Methods}\nNo experiments were executed for this session.\n"
            ),
            "per_hypothesis": {},
        }

    rounds = max(1, agg["rounds"])
    round_phrase = "a single round" if rounds == 1 else f"{rounds} successive rounds"
    protocol = (
        "We addressed the research question through autonomous, fully automated experimentation. "
        f"A total of {counts['tested']} falsifiable hypotheses were generated and tested across {round_phrase} of "
        "investigation, with each later round informed by the confirmed and refuted findings of the previous one. "
        "Every hypothesis was assigned to an independent sub-agent that selected computational instruments, executed "
        "them (and, where required, wrote and ran code in an isolated sandbox), observed the results, and decided its "
        "next action before reaching a verdict."
    )

    if tools:
        ranked = sorted(tools.items(), key=lambda kv: kv[1], reverse=True)
        tool_phrases = ", ".join(
            f"\\texttt{{{_latex_escape(name)}}} ({n} call{'s' if n != 1 else ''})" for name, n in ranked[:12]
        )
        instruments = f"The experiments employed the following instruments: {tool_phrases}."
        if agg["code_steps"]:
            instruments += (
                f" In addition, {agg['code_steps']} custom computation"
                f"{'s were' if agg['code_steps'] != 1 else ' was'} executed in the sandbox where no pre-built "
                "instrument was sufficient."
            )
    else:
        instruments = (
            f"All {agg['code_steps']} experimental computations were performed as custom code executed in an "
            "isolated sandbox."
        )

    sig_used = sorted(n for n in tools if n in _SIGNIFICANCE_TOOLS)
    if sig_used:
        sig_list = ", ".join(f"\\texttt{{{_latex_escape(n)}}}" for n in sig_used)
        stats_method = (
            f"Statistical support was established using {sig_list}. A hypothesis was marked \\emph{{confirmed}} only "
            "when a metric-bearing experiment yielded significant evidence -- a $p$-value below 0.05, an absolute "
            "effect size above 0.2, or a 95\\% confidence interval excluding the null -- in the direction predicted "
            "by the hypothesis; otherwise it was recorded as refuted or inconclusive."
        )
    else:
        stats_method = (
            "A hypothesis was marked \\emph{confirmed} only when a metric-bearing experiment yielded significant "
            "evidence (a $p$-value below 0.05, an absolute effect size above 0.2, or a 95\\% confidence interval "
            "excluding the null) in the predicted direction; otherwise it was recorded as refuted or inconclusive."
        )

    reproducibility = (
        "All hypotheses, parameters, intermediate outputs, and statistical computations are persisted in a structured "
        "trace and are available for independent inspection and replication."
    )

    body = "\n\n".join([protocol, instruments, stats_method, reproducibility])
    combined = f"\\section{{Methods}}\n{body}\n"
    return {"combined_latex": combined, "per_hypothesis": {}}


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


def _finding_sentence(f: dict[str, Any]) -> str:
    """A single prose sentence describing one finding, grounded in its statistics."""
    claim = f.get("key_finding") or f.get("text") or ""
    claim = claim.strip().rstrip(".")
    stats = f.get("stats_text") or ""
    sent = _latex_escape(claim[:400])
    if stats and stats != "no inferential statistic recorded":
        sent += f" ({stats})"
    return sent + "."


def _findings_table(findings: list[dict[str, Any]]) -> str:
    """A clean summary table of confirmed findings (no trace IDs, no raw JSON)."""
    if not findings:
        return ""
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Summary of supported findings and their statistical evidence.}",
        "\\begin{tabular}{p{0.46\\linewidth} c p{0.30\\linewidth}}",
        "\\hline",
        "\\textbf{Finding} & \\textbf{Confidence} & \\textbf{Statistical evidence} \\\\",
        "\\hline",
    ]
    for f in findings[:20]:
        claim = (f.get("key_finding") or f.get("text") or "").strip().rstrip(".")
        cell = _latex_escape(claim[:220])
        conf = f.get("confidence")
        conf_s = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "--"
        stats = f.get("stats_text") or "--"
        lines.append(f"{cell} & {conf_s} & {stats} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


async def compile_session_results_latex(session_factory: async_sessionmaker, session_id: str) -> str:
    """
    Deterministic Results section as readable prose plus a findings table.

    Outcomes are organised by what was learned (supported / refuted / inconclusive),
    not as a chronological per-hypothesis verdict log. Counts here are produced by the
    same classifier (:func:`compile_session_findings`) that feeds the abstract.
    """
    findings = await compile_session_findings(session_factory, session_id)
    counts = findings["counts"]
    if counts["tested"] == 0:
        return "\\section{Results}\nNo hypotheses were executed for this session.\n"

    parts: list[str] = ["\\section{Results}"]

    lead = (
        f"We evaluated {counts['tested']} hypotheses. "
        f"{counts['confirmed']} were supported by statistically significant evidence, "
        f"{counts['refuted']} were refuted, and {counts['inconclusive']} remained inconclusive."
    )
    if counts["unexecuted"]:
        lead += (
            f" A further {counts['unexecuted']} generated hypotheses were not executed and are excluded from "
            "the analysis."
        )
    parts.append(lead)

    confirmed = findings["confirmed"]
    if confirmed:
        parts.append("\\subsection{Supported findings}")
        parts.append(_findings_table(confirmed))
        for f in confirmed[:10]:
            parts.append(_finding_sentence(f))
    else:
        parts.append("\\subsection{Supported findings}")
        parts.append(
            "No hypothesis met the significance bar against the recorded measurements. The question remains "
            "open under the methods applied here."
        )

    refuted = findings["refuted"]
    if refuted:
        parts.append("\\subsection{Refuted hypotheses}")
        bullets = "\n".join(f"\\item {_finding_sentence(f)}" for f in refuted[:10])
        parts.append("\\begin{itemize}\n" + bullets + "\n\\end{itemize}")

    inconclusive = findings["inconclusive"]
    if inconclusive:
        parts.append("\\subsection{Inconclusive directions}")
        parts.append(
            f"{len(inconclusive)} hypotheses produced experiments without decisive statistical support and warrant "
            "better-powered follow-up. The highest-confidence among them concerned: "
            + _latex_escape("; ".join((f.get("text") or "")[:120] for f in inconclusive[:3]))
            + "."
        )
    return "\n\n".join(p for p in parts if p)


def compile_references_latex(prior: dict[str, Any] | None) -> str:
    """Bibliography from prior key papers in a conventional reference style."""
    prior = prior or {}
    papers = prior.get("key_papers") or []
    if not isinstance(papers, list) or not papers:
        return "\\section{References}\nNo external references were retrieved for this study.\n"

    lines: list[str] = ["\\section{References}", "\\begin{thebibliography}{99}"]
    for i, p in enumerate(papers[:48]):
        if not isinstance(p, dict):
            continue
        pid = _latex_escape(str(p.get("paper_id") or f"ref{i}"))
        title = _latex_escape(str(p.get("title") or "Untitled").strip())[:400]
        authors = p.get("authors")
        if isinstance(authors, list) and authors:
            names = ", ".join(_latex_escape(str(a)) for a in authors[:6])
            if len(authors) > 6:
                names += " et al."
            author_str = names + ". "
        elif isinstance(authors, str) and authors.strip():
            author_str = _latex_escape(authors.strip()[:200]) + ". "
        else:
            author_str = ""
        arxiv = str(p.get("paper_id") or "").strip()
        arxiv_str = f" arXiv:{_latex_escape(arxiv)}." if arxiv and not arxiv.startswith("ref") else ""
        lines.append(f"\\bibitem{{{pid}}}{author_str}\\textit{{{title}}}.{arxiv_str}")
    lines.append("\\end{thebibliography}")
    return "\n".join(lines)
