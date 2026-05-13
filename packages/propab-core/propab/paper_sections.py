from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from propab.llm import LLMClient
from propab.paper_compiler import _latex_escape


def _template_env() -> Path:
    return Path(__file__).resolve().parent / "templates"


def render_paper_tex(
    *,
    title: str,
    abstract: str,
    introduction: str,
    methods_tex: str,
    results_tex: str,
    figures_tex: str,
    discussion: str,
    conclusion: str,
    references_tex: str,
) -> str:
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(_template_env())), autoescape=False)
    tpl = env.get_template("paper_arxiv.tex.j2")
    return tpl.render(
        title=_latex_escape(title.strip()[:200] or "Research report"),
        abstract=abstract,
        introduction=introduction,
        methods_tex=methods_tex,
        results_tex=results_tex,
        figures_tex=figures_tex,
        discussion=discussion,
        conclusion=conclusion,
        references_tex=references_tex,
    )


def _confirmed_hypothesis_count(synthesis: dict[str, Any] | None) -> int:
    syn = synthesis or {}
    ledger = syn.get("ledger")
    if isinstance(ledger, dict):
        confirmed = ledger.get("confirmed")
        if isinstance(confirmed, list) and confirmed:
            return len(confirmed)
    tc = syn.get("total_confirmed")
    if isinstance(tc, int) and tc > 0:
        return tc
    return 0


def _ledger_counts(synthesis: dict[str, Any]) -> tuple[int, int, int]:
    """Confirmed/refuted/inconclusive counts: prefer synthesis['ledger'], else top-level totals."""
    syn = synthesis or {}
    ledger = syn.get("ledger")
    if isinstance(ledger, dict):
        n_c = len(ledger["confirmed"]) if isinstance(ledger.get("confirmed"), list) else 0
        n_r = len(ledger["refuted"]) if isinstance(ledger.get("refuted"), list) else 0
        n_i = len(ledger["inconclusive"]) if isinstance(ledger.get("inconclusive"), list) else 0
        if n_c or n_r or n_i:
            return n_c, n_r, n_i
    n_c = int(syn["total_confirmed"]) if isinstance(syn.get("total_confirmed"), int) else 0
    n_r = int(syn["total_refuted"]) if isinstance(syn.get("total_refuted"), int) else 0
    n_i = int(syn["total_inconclusive"]) if isinstance(syn.get("total_inconclusive"), int) else 0
    return n_c, n_r, n_i


def _annotate_synthesis_blob_for_llm(synthesis: dict[str, Any] | None) -> str:
    """Prefix synthesis JSON so the LLM sees authoritative ledger counts."""
    n_c, n_r, n_i = _ledger_counts(synthesis or {})
    if n_c or n_r or n_i:
        header = (
            "AUTHORITATIVE_LEDGER_COUNTS: "
            f"confirmed={n_c}, refuted={n_r}, inconclusive={n_i}. "
            "Use these exact integers when describing experiment outcomes.\n"
        )
        return header + json.dumps(synthesis or {}, ensure_ascii=False)[:4000]
    return json.dumps(synthesis or {}, ensure_ascii=False)[:4000]


def _merge_ledger_into_llm_abstract(abstract: str, synthesis: dict[str, Any] | None) -> str:
    """LLM prose often hallucinates ``confirmed=0``; align with synthesis ledger."""
    n_c, n_r, n_i = _ledger_counts(synthesis or {})
    if not (n_c or n_r or n_i):
        return abstract
    fixed = abstract
    fixed = re.sub(r"(?i)confirmed\s*=\s*\d+", f"confirmed={n_c}", fixed, count=1)
    fixed = re.sub(r"(?i)refuted\s*=\s*\d+", f"refuted={n_r}", fixed, count=1)
    fixed = re.sub(r"(?i)inconclusive\s*=\s*\d+", f"inconclusive={n_i}", fixed, count=1)
    if (
        re.search(rf"(?i)confirmed\s*=\s*{n_c}\b", fixed)
        and re.search(rf"(?i)refuted\s*=\s*{n_r}\b", fixed)
        and re.search(rf"(?i)inconclusive\s*=\s*{n_i}\b", fixed)
    ):
        return fixed
    tag = _latex_escape(
        f"[Hypothesis ledger (authoritative): {n_c} confirmed, {n_r} refuted, {n_i} inconclusive.] "
    )
    return tag + fixed


def _fallback_from_context(question: str, prior: dict[str, Any], synthesis: dict[str, Any] | None) -> dict[str, str]:
    syn = synthesis or {}
    if syn.get("short_circuit"):
        ans = str(syn.get("short_answer") or "")
        body = _latex_escape(ans[:2000])
        return {
            "abstract": body or _latex_escape(question[:1200]),
            "introduction": _latex_escape(
                "This report documents a literature-aligned answer produced without full hypothesis testing."
            ),
            "discussion": _latex_escape(
                "The short-circuit path skipped multi-hypothesis experiments; see abstract and prior literature."
            ),
            "conclusion": _latex_escape(
                "Established literature supports the stated answer; reproduce with full experiments if stakes are high."
            ),
        }
    kpapers = prior.get("key_papers") or []
    cite = ""
    if isinstance(kpapers, list) and kpapers:
        first = kpapers[0] if isinstance(kpapers[0], dict) else {}
        cite = str(first.get("title", ""))[:200]
    intro_plain = (
        f"We address: {question[:900]}. "
        f"Prior retrieval surfaced {len(kpapers)} seed papers"
        + (f", including ``{cite}''." if cite else ".")
    )
    n_c, n_r, n_i = _ledger_counts(syn)
    abs_plain = (
        f"This session investigated: {question[:700]}. "
        f"Automated sub-agent runs produced stored traces summarized in Results "
        f"(confirmed={n_c}, refuted={n_r}, inconclusive={n_i}). "
    )
    if n_c == 0:
        abs_plain += (
            "No hypothesis met the platform confirmation bar against the recorded tool outputs; "
            "do not treat narrative optimizer rankings as established facts without independent replication."
        )
    else:
        abs_plain += "Claims tied to confirmed hypotheses are summarized in Results; other rows remain tentative."

    disc_plain = (
        f"Automated experiments concluded with ledger "
        f"(confirmed={n_c}, refuted={n_r}, inconclusive={n_i}). "
        "Limitations include sandboxed execution and tool proxies where noted in outputs."
    )
    concl_plain = (
        "Primary takeaways depend on hypotheses marked confirmed in the results section; "
        "inconclusive rows indicate insufficient signal or tooling limits."
    )
    return {
        "abstract": _latex_escape(abs_plain[:2000]),
        "introduction": _latex_escape(intro_plain),
        "discussion": _latex_escape(disc_plain),
        "conclusion": _latex_escape(concl_plain),
    }


async def generate_prose_sections(
    *,
    llm: LLMClient | None,
    session_id: str,
    question: str,
    prior: dict[str, Any] | None,
    synthesis: dict[str, Any] | None,
) -> dict[str, str]:
    prior = prior or {}
    use_llm = llm is not None and (
        str(getattr(llm, "provider", "") or "").lower() in ("ollama", "gemini")
        or bool(str(getattr(llm, "api_key", "") or "").strip())
    )
    # LLM prose routinely asserts definitive rankings when every hypothesis is inconclusive;
    # the ledger from synthesis is authoritative for what may be claimed at high level.
    if use_llm and _confirmed_hypothesis_count(synthesis) == 0:
        use_llm = False
    if not use_llm:
        return _fallback_from_context(question, prior, synthesis)

    prior_blob = json.dumps(prior, ensure_ascii=False)[:6000]
    syn_blob = _annotate_synthesis_blob_for_llm(synthesis)

    async def one(purpose: str, instruction: str) -> str:
        prompt = (
            f"You write one concise LaTeX-free paragraph for an ML research report.\n"
            f"Research question: {question}\n"
            f"Context JSON (truncated): {prior_blob}\n"
            f"Synthesis JSON (truncated): {syn_blob}\n"
            f"Task: {instruction}\n"
            "Output plain text only, no markdown, no citations markup, <= 180 words."
        )
        for attempt in range(3):
            try:
                raw = (await llm.call(prompt=prompt, purpose=purpose, session_id=session_id)).strip()
                if raw.startswith("[") or raw.startswith("{"):
                    return ""
                return _latex_escape(raw[:2500])
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(1.5 * (2**attempt))
                    continue
                return ""

    abstract = await one("paper.abstract", "Write the abstract summarizing problem, approach, and high-level outcome.")
    introduction = await one("paper.introduction", "Write the introduction: motivation, gap, and what was executed.")
    discussion = await one("paper.discussion", "Write discussion: interpret experiment ledger, limitations, rival explanations.")
    conclusion = await one("paper.conclusion", "Write a tight conclusion: what was learned, what remains open, one recommendation.")

    fb = _fallback_from_context(question, prior, synthesis)
    return {
        "abstract": _merge_ledger_into_llm_abstract(abstract or fb["abstract"], synthesis),
        "introduction": introduction or fb["introduction"],
        "discussion": discussion or fb["discussion"],
        "conclusion": conclusion or fb["conclusion"],
    }
