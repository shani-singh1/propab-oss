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


def _ledger_counts(synthesis: dict[str, Any]) -> tuple[int, int, int]:
    """Confirmed/refuted/inconclusive: prefer synthesis['ledger'] lists, else top-level totals."""
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


def outcome_counts(synthesis: dict[str, Any] | None) -> dict[str, int]:
    """
    Authoritative outcome counts for prose. Prefers the DB-derived ``counts`` block
    (set by the paper writer from :func:`compile_session_findings`) so the abstract
    always matches the results section; falls back to the in-memory ledger shape.
    """
    syn = synthesis or {}
    c = syn.get("counts")
    if isinstance(c, dict) and any(isinstance(c.get(k), int) for k in ("confirmed", "refuted", "inconclusive")):
        confirmed = int(c.get("confirmed") or 0)
        refuted = int(c.get("refuted") or 0)
        inconclusive = int(c.get("inconclusive") or 0)
        tested = int(c.get("tested") or (confirmed + refuted + inconclusive))
        return {"confirmed": confirmed, "refuted": refuted, "inconclusive": inconclusive, "tested": tested}
    n_c, n_r, n_i = _ledger_counts(syn)
    return {"confirmed": n_c, "refuted": n_r, "inconclusive": n_i, "tested": n_c + n_r + n_i}


def _best_finding(synthesis: dict[str, Any] | None) -> dict[str, Any] | None:
    syn = synthesis or {}
    cf = syn.get("confirmed_findings")
    if isinstance(cf, list) and cf and isinstance(cf[0], dict):
        return cf[0]
    bf = syn.get("best_finding")
    if isinstance(bf, dict) and bf:
        return bf
    return None


def _metric_summary(synthesis: dict[str, Any] | None) -> dict[str, Any] | None:
    """Baseline-relative metric facts for honest reporting, or None when unavailable."""
    syn = synthesis or {}
    baseline = syn.get("baseline_metric")
    best = syn.get("best_metric")
    if not isinstance(baseline, (int, float)) or not isinstance(best, (int, float)):
        return None
    if abs(float(baseline)) < 1e-12:
        return None
    imp = syn.get("improvement_pct_over_baseline")
    return {
        "baseline": float(baseline),
        "best": float(best),
        "improvement_pct": float(imp) if isinstance(imp, (int, float)) else None,
        "metric_name": str(syn.get("metric_name") or "the primary metric").replace("_", " "),
    }


def _baseline_clause(metric: dict[str, Any] | None) -> str:
    """LaTeX-ready sentence stating whether any result beat the measured baseline.

    This is the honesty guard: "supported" means a statistically significant effect was
    detected within the experiment, which is NOT the same as beating the baseline. State
    the baseline comparison explicitly so the abstract cannot imply an improvement that
    did not happen.
    """
    if not metric:
        return ""
    name = _latex_escape(metric["metric_name"])
    baseline = metric["baseline"]
    best = metric["best"]
    imp = metric["improvement_pct"]
    imp_txt = "" if imp is None else f" ({imp:+.1f}\\%)"
    if best > baseline + 1e-9:
        return (
            f" The strongest configuration reached {best:.3f} {name}, improving on the "
            f"{baseline:.3f} baseline{imp_txt}."
        )
    return (
        f" No configuration exceeded the {baseline:.3f} {name} baseline; the best result was "
        f"{best:.3f}{imp_txt}, so the supported findings are within-experiment comparisons "
        "rather than improvements over the baseline."
    )


def _outcome_sentence(
    counts: dict[str, int],
    best: dict[str, Any] | None,
    metric: dict[str, Any] | None = None,
) -> str:
    """
    A natural-language sentence stating the authoritative outcome (no count tuples).

    Returns LaTeX-ready text: free text is escaped here, while ``stats_text`` (which is
    already LaTeX math) is passed through. Callers must NOT re-escape the result.
    """
    tested, c, r = counts["tested"], counts["confirmed"], counts["refuted"]
    if tested == 0:
        return "No experiments were executed in this study."
    if c == 0:
        s = (
            f"Across {tested} hypotheses evaluated, none met the significance threshold against the recorded "
            "measurements, so the question remains open under the methods applied here."
        )
        return s + _baseline_clause(metric)
    s = (
        f"Across {tested} hypotheses evaluated, {c} {'was' if c == 1 else 'were'} supported by a statistically "
        "significant within-experiment effect"
    )
    if r:
        s += f" and {r} {'was' if r == 1 else 'were'} refuted"
    s += "."
    if best:
        claim = str(best.get("key_finding") or best.get("text") or "").strip().rstrip(".")
        stats = str(best.get("stats_text") or "")
        if claim:
            claim = _latex_escape(claim[0].lower() + claim[1:])
            frag = f" The most significant effect was that {claim}"
            if stats and stats != "no inferential statistic recorded":
                frag += f" ({stats})"
            s += frag + "."
    # Always anchor the abstract to the baseline comparison so "supported" is never
    # mistaken for "beat the baseline".
    s += _baseline_clause(metric)
    return s


_TITLE_QUESTION_WORDS = r"(what|how|does|do|is|are|can|could|why|which|when|where|will|would|should)"


def _fallback_title(question: str) -> str:
    q = (question or "").strip().rstrip("?.! ")
    # Strip a leading interrogative phrase (e.g. "What is", "How does") to get a declarative title.
    for _ in range(2):
        stripped = re.sub(rf"^{_TITLE_QUESTION_WORDS}\b[\s,:-]*", "", q, flags=re.IGNORECASE).strip()
        if stripped == q or not stripped:
            break
        q = stripped
    if not q:
        q = (question or "Automated research report").strip()
    return (q[0].upper() + q[1:])[:180] if q else "Automated research report"


async def _llm_title(llm: LLMClient, session_id: str, question: str, best: dict[str, Any] | None) -> str:
    finding = ""
    if best:
        finding = str(best.get("key_finding") or best.get("text") or "")[:300]
    prompt = (
        "Write a single concise, specific academic paper title (at most 16 words) for a study answering this "
        f"question: {question}\n"
        + (f"Main finding: {finding}\n" if finding else "")
        + "Output only the title text: no quotes, no markdown, no trailing period."
    )
    try:
        raw = (await llm.call(prompt=prompt, purpose="paper.title", session_id=session_id)).strip()
        raw = raw.strip().strip('"').strip("'").rstrip(".")
        if 0 < len(raw) <= 200 and "\n" not in raw:
            return raw
    except Exception:
        pass
    return _fallback_title(question)


def _fallback_from_context(
    question: str, prior: dict[str, Any], synthesis: dict[str, Any] | None
) -> dict[str, str]:
    syn = synthesis or {}
    if syn.get("short_circuit"):
        ans = str(syn.get("short_answer") or "")
        return {
            "title": _fallback_title(question),
            "abstract": _latex_escape(ans[:2000]) or _latex_escape(question[:1200]),
            "introduction": _latex_escape(
                f"We address the question: {question[:900]}. This report documents an answer that is well supported "
                "by existing literature and therefore did not require new hypothesis testing."
            ),
            "discussion": _latex_escape(
                "Because the question was resolvable from established prior work, the system returned the "
                "literature-grounded answer directly rather than running independent experiments."
            ),
            "conclusion": _latex_escape(
                "The stated answer follows from established results; independent experimental replication is "
                "recommended where the stakes are high."
            ),
        }

    kpapers = prior.get("key_papers") or []
    cite = ""
    if isinstance(kpapers, list) and kpapers and isinstance(kpapers[0], dict):
        cite = str(kpapers[0].get("title", ""))[:200]

    counts = outcome_counts(syn)
    best = _best_finding(syn)
    outcome = _outcome_sentence(counts, best, _metric_summary(syn))  # LaTeX-ready (do not re-escape)
    q_esc = _latex_escape(question[:900])
    cite_esc = _latex_escape(cite)
    n_papers = len(kpapers)
    works_word = "work" if n_papers == 1 else "works"

    if kpapers:
        lit_sentence = (
            f"Retrieval over the prior literature surfaced {n_papers} relevant {works_word}"
            + (f", including ``{cite_esc}''. " if cite_esc else ". ")
        )
    else:
        lit_sentence = "No closely matching prior work was retrieved, motivating direct experimentation. "

    intro = (
        f"We study the following question: {q_esc} "
        + lit_sentence
        + "We approach it with autonomous experimentation: generating falsifiable hypotheses, testing each with "
        "computational instruments under a fixed protocol, and admitting a claim only when it is backed by "
        "significant statistical evidence."
    )
    abstract = (
        f"We investigate the following question through autonomous, fully automated experimentation: {q_esc} "
        "Falsifiable hypotheses were generated and tested under a uniform protocol, with claims admitted only on "
        f"significant statistical evidence. {outcome}"
    )
    if counts["confirmed"] == 0:
        discussion = (
            "The experiments did not surface evidence strong enough to confirm any hypothesis under the present "
            "protocol. This is an honest negative result: it constrains the space of likely answers rather than "
            "establishing one. Limitations include the breadth of instruments available to the agents and the "
            "compute budget allotted per hypothesis."
        )
        conclusion = (
            "No claim met the evidence bar in this study. Promising but inconclusive directions, summarised in the "
            "results, are the natural targets for a better-powered follow-up."
        )
    else:
        discussion = (
            "The supported findings are reported with their statistical evidence in the results section. They should "
            "be read as automated, reproducible results subject to the breadth of the available instruments and the "
            "per-hypothesis compute budget; independent replication remains valuable before strong claims are made."
        )
        conclusion = (
            "The study yields concrete, statistically supported findings together with refuted and inconclusive "
            "directions that scope future work. We recommend independent replication of the strongest result."
        )
    # abstract/introduction are already LaTeX-ready (escaped fragments + LaTeX outcome);
    # discussion/conclusion are static safe prose.
    return {
        "title": _fallback_title(question),
        "abstract": abstract[:2400],
        "introduction": intro[:2400],
        "discussion": discussion[:2200],
        "conclusion": conclusion[:1600],
    }


async def generate_prose_sections(
    *,
    llm: LLMClient | None,
    session_id: str,
    question: str,
    prior: dict[str, Any] | None,
    synthesis: dict[str, Any] | None,
) -> dict[str, str]:
    """
    Produce publishable narrative sections (title, abstract, introduction, discussion,
    conclusion). Quantitative outcome statements are always derived deterministically
    from the authoritative counts, so they match the results section exactly; the LLM,
    when available, supplies only the surrounding framing.
    """
    prior = prior or {}
    syn = synthesis or {}
    counts = outcome_counts(syn)
    best = _best_finding(syn)
    metric = _metric_summary(syn)
    fb = _fallback_from_context(question, prior, synthesis)

    use_llm = llm is not None and (
        str(getattr(llm, "provider", "") or "").lower() in ("ollama", "gemini")
        or bool(str(getattr(llm, "api_key", "") or "").strip())
    )
    if not use_llm or syn.get("short_circuit"):
        return fb

    prior_blob = json.dumps(prior, ensure_ascii=False)[:5000]
    outcome = _outcome_sentence(counts, best, metric)

    async def one(purpose: str, instruction: str) -> str:
        prompt = (
            "You write one concise, formal paragraph for an academic research paper. "
            "Do NOT state any numeric counts of confirmed/refuted/inconclusive hypotheses (those are added "
            "separately and authoritatively). Do not invent results.\n"
            f"Research question: {question}\n"
            f"Prior-literature context (JSON, truncated): {prior_blob}\n"
            f"Authoritative outcome (use as the factual basis, do not restate the numbers verbatim): {outcome}\n"
            f"Task: {instruction}\n"
            "Output plain prose only: no markdown, no citation markup, no bullet points, <= 170 words."
        )
        for attempt in range(3):
            try:
                raw = (await llm.call(prompt=prompt, purpose=purpose, session_id=session_id)).strip()
                if raw.startswith("[") or raw.startswith("{"):
                    return ""
                return _latex_escape(raw[:2400])
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(1.5 * (2**attempt))
                    continue
                return ""

    title = await _llm_title(llm, session_id, question, best)
    intro = await one("paper.introduction", "Write the introduction: motivation, the gap in prior work, and the approach taken.")
    framing = await one("paper.abstract", "Write the abstract's opening: the problem and the method, ending right before the results.")
    discussion = await one(
        "paper.discussion",
        "Write the discussion: interpret what the supported findings mean, give rival explanations, and state limitations.",
    )
    conclusion = await one(
        "paper.conclusion", "Write a tight conclusion: the main takeaway and the most important open question."
    )

    # The abstract's quantitative claims are always the authoritative outcome sentence
    # (already LaTeX-ready), appended to LLM framing — guaranteeing it matches Results.
    abstract = (framing.strip() + " " + outcome).strip() if framing else fb["abstract"]
    return {
        "title": title or fb["title"],
        "abstract": abstract,
        "introduction": intro or fb["introduction"],
        "discussion": discussion or fb["discussion"],
        "conclusion": conclusion or fb["conclusion"],
    }
