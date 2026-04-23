from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker


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
            tool_name = input_data.get("tool", "unknown_tool")
            params = input_data.get("params")
            duration_ms = row["duration_ms"] or 0
            lines.append(
                f"Tool \\texttt{{{tool_name}}} was called with parameters {_format_params(params)}. "
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
            summary = row["summary"] or "intermediate evaluation"
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
        section = (
            f"\\subsection{{Hypothesis {label}}}\\label{{subsec:{hid}}}\n"
            f"\\textit{{{hyp['text']}}}\\\\\n"
            f"{body}\n"
        )
        sections.append(section)
        per_hypothesis[hid] = body

    combined = "\\section{Methods}\n" + "\n".join(sections) if sections else "\\section{Methods}\nNo hypotheses were executed.\n"
    return {"combined_latex": combined, "per_hypothesis": per_hypothesis}
