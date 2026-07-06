"""
Research-paper artifacts derived from the *gated* trace.

This module turns the authoritative, honesty-gated findings (produced by
:func:`propab.paper_compiler.compile_session_findings`, whose single source of
truth is :func:`propab.paper_compiler._effective_verdict`) into the components a
real research paper needs:

    * a summary-counts table (confirmed / refuted / inconclusive / tested),
    * a findings table (hypothesis | verdict | metric vs baseline | evidence type
      | replication),
    * figure *specifications* — pure data (labels + numbers) that the orchestrator
      renders with matplotlib and embeds,
    * a deterministic Chain-of-Reasoning / Research Narrative section reconstructing
      how the campaign moved from question -> beliefs -> experiments -> findings.

HONESTY INVARIANT
-----------------
Everything here consumes the SAME gated ``findings`` dict. The reasoning trace
(hypothesis lineage, belief history) is *structural* context read off the tree, but
whenever a node is referenced as an outcome its verdict is re-mapped to the gated
verdict via ``gated_verdict_by_id``. A node that the tree calls "confirmed" but that
the DB gate demoted to inconclusive (missing metric, control, duplicate, unexecuted)
is therefore NEVER presented as a supported result — in any table, figure, or
narrative sentence. Controls are excluded from findings entirely.
"""
from __future__ import annotations

from typing import Any

from propab.paper_compiler import _latex_escape


# ── synthetic-data provenance (DOM2) ─────────────────────────────────────────

SYNTHETIC_LABEL = "synthetic dataset (illustrative)"


def _is_synthetic(f: dict[str, Any]) -> bool:
    """True if a finding is backed by a seed-generated (illustrative) dataset.

    DOM2 honesty fix: a finding whose evidence carries
    ``data_provenance == "synthetic"`` must be labelled as illustrative in every
    paper section — never presented as a real-world result.
    """
    return str((f.get("stats") or {}).get("data_provenance") or f.get("data_provenance") or "").lower() == "synthetic"


def _any_synthetic(findings: dict[str, Any]) -> bool:
    for verdict in ("confirmed", "refuted", "inconclusive"):
        if any(_is_synthetic(f) for f in findings.get(verdict) or []):
            return True
    return False


# ── gated verdict lookup ─────────────────────────────────────────────────────

def gated_verdict_by_id(findings: dict[str, Any]) -> dict[str, str]:
    """Map hypothesis-id -> gated verdict, using ONLY the gated finding buckets.

    Any id not present here was excluded by the gate (unexecuted, control, or
    filtered) and must never be shown as a result. This is the choke point that
    keeps the narrative honest: callers ask this map, not the raw tree verdict.
    """
    out: dict[str, str] = {}
    for verdict in ("confirmed", "refuted", "inconclusive"):
        for f in findings.get(verdict) or []:
            fid = str(f.get("id") or "")
            if fid:
                out[fid] = verdict
    return out


def _metric_vs_baseline_cell(f: dict[str, Any], baseline: float | None) -> str:
    """Text for the 'metric vs baseline' column, honest about missing/relative data."""
    stats = f.get("stats") or {}
    mv = stats.get("metric_value")
    if isinstance(mv, (int, float)) and isinstance(baseline, (int, float)) and abs(baseline) > 1e-12:
        delta = (float(mv) - float(baseline)) / abs(float(baseline)) * 100.0
        sign = "+" if delta >= 0 else ""
        return _latex_escape(f"{float(mv):.3g} vs {float(baseline):.3g} ({sign}{delta:.1f}%)")
    if isinstance(mv, (int, float)):
        return _latex_escape(f"{float(mv):.3g} (no baseline)")
    es = stats.get("effect_size")
    if isinstance(es, (int, float)):
        return _latex_escape(f"within-exp d={float(es):.2f}")
    return "within-experiment"


def _evidence_type_cell(f: dict[str, Any]) -> str:
    ct = f.get("claim_type") or "--"
    vm = f.get("verification_method")
    return _latex_escape(f"{ct}" + (f" / {vm}" if vm else ""))


# ── Tables ───────────────────────────────────────────────────────────────────

def summary_counts_table(counts: dict[str, int]) -> str:
    """A compact table of the authoritative outcome counts."""
    rows = [
        ("Confirmed (supported by significant evidence)", counts.get("confirmed", 0)),
        ("Refuted", counts.get("refuted", 0)),
        ("Inconclusive", counts.get("inconclusive", 0)),
        ("Total executed and tested", counts.get("tested", 0)),
    ]
    if counts.get("unexecuted"):
        rows.append(("Generated but not executed (excluded)", counts["unexecuted"]))
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Authoritative outcome counts. All figures, tables, and narrative in "
        "this manuscript are derived from these same gated classifications.}",
        "\\label{tab:summary-counts}",
        "\\begin{tabular}{|l|r|}",
        "\\hline",
        "\\textbf{Outcome} & \\textbf{Count} \\\\",
        "\\hline",
    ]
    for label, n in rows:
        lines.append(f"{_latex_escape(label)} & {int(n)} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def findings_table(
    findings: dict[str, Any],
    *,
    baseline: float | None = None,
    max_rows: int = 20,
) -> str:
    """Findings table: hypothesis | verdict | metric vs baseline | evidence type | replication.

    Confirmed and refuted findings are listed (both are decisive, gated outcomes).
    Inconclusive rows are intentionally omitted from the findings table so an
    inconclusive direction can never read as a result; its count is in the summary
    table and its prose lives in the Results section.
    """
    confirmed = findings.get("confirmed") or []
    refuted = findings.get("refuted") or []
    ordered = [("Confirmed", f) for f in confirmed] + [("Refuted", f) for f in refuted]
    if not ordered:
        return ""
    # DOM2: if any listed finding runs on a synthetic (illustrative) dataset, the
    # caption must say so and each such row must be marked, so the table cannot
    # read as a set of real-world results.
    synthetic_present = any(_is_synthetic(f) for _, f in ordered[:max_rows])
    caption = (
        "Decisive findings and their evidence. `Metric vs baseline' reports the "
        "measured metric against the campaign baseline where one exists; otherwise the "
        "within-experiment effect. Inconclusive directions are excluded by construction."
    )
    if synthetic_present:
        caption += (
            " Rows marked [synthetic dataset (illustrative)] were computed on a "
            "seed-generated dataset and are illustrative of the pipeline, not real-world results."
        )
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:findings}",
        "\\small",
        "\\begin{tabular}{p{0.34\\linewidth} l p{0.20\\linewidth} l l}",
        "\\hline",
        "\\textbf{Hypothesis} & \\textbf{Verdict} & \\textbf{Metric vs baseline} & "
        "\\textbf{Evidence} & \\textbf{Repl.} \\\\",
        "\\hline",
    ]
    for verdict_label, f in ordered[:max_rows]:
        claim = (f.get("key_finding") or f.get("text") or "").strip().rstrip(".")
        cell = _latex_escape(claim[:140])
        if _is_synthetic(f):
            cell += f" [{_latex_escape(SYNTHETIC_LABEL)}]"
        mvb = _metric_vs_baseline_cell(f, baseline)
        ev = _evidence_type_cell(f)
        repl = _latex_escape(str(f.get("replication_level") or "--"))
        lines.append(f"{cell} & {verdict_label} & {mvb} & {ev} & {repl} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


# ── Figure specifications (pure data; rendered by the orchestrator) ───────────

def build_figure_specs(
    findings: dict[str, Any],
    *,
    reasoning_trace: dict[str, Any] | None = None,
    baseline: float | None = None,
    metric_name: str = "metric",
) -> list[dict[str, Any]]:
    """Return matplotlib-ready figure specs, all sourced from the gated findings.

    Each spec is ``{"kind", "title", "caption", "data": {...}}``. The orchestrator
    turns these into PNGs. Returning data (not images) keeps the honesty gate unit-
    testable without a plotting backend and guarantees every plotted number came
    through the same gate as the counts.
    """
    counts = findings.get("counts") or {}
    specs: list[dict[str, Any]] = []

    # (1) Outcome-distribution bar chart — straight from gated counts.
    specs.append({
        "kind": "outcome_bars",
        "title": "Hypothesis outcomes",
        "caption": "Distribution of gated hypothesis outcomes across the campaign. "
                   "Counts match Table~\\ref{tab:summary-counts}.",
        "data": {
            "labels": ["Confirmed", "Refuted", "Inconclusive"],
            "values": [
                int(counts.get("confirmed", 0)),
                int(counts.get("refuted", 0)),
                int(counts.get("inconclusive", 0)),
            ],
        },
    })

    # (2) Confirmed-finding metric-vs-baseline chart (only findings that carry a metric).
    metric_points: list[dict[str, Any]] = []
    for f in findings.get("confirmed") or []:
        mv = (f.get("stats") or {}).get("metric_value")
        if isinstance(mv, (int, float)):
            label = (f.get("key_finding") or f.get("text") or "finding").strip()
            metric_points.append({"label": label[:48], "value": float(mv)})
    if metric_points:
        specs.append({
            "kind": "metric_vs_baseline",
            "title": f"Confirmed findings vs baseline ({metric_name})",
            "caption": "Measured metric for each confirmed finding that recorded one, "
                       "against the campaign baseline (dashed). Findings without a metric "
                       "are omitted rather than imputed.",
            "data": {
                "points": metric_points[:12],
                "baseline": float(baseline) if isinstance(baseline, (int, float)) else None,
                "metric_name": metric_name,
            },
        })

    # (3) Confirmation-over-rounds timeline — from the per-generation gated histogram.
    gens = (reasoning_trace or {}).get("generations") or []
    if gens:
        specs.append({
            "kind": "rounds_timeline",
            "title": "Outcomes by round",
            "caption": "How decisive outcomes accumulated across successive rounds of "
                       "investigation, using the gated per-round verdict histogram.",
            "data": {
                "generations": [int(g.get("generation", i)) + 1 for i, g in enumerate(gens)],
                "confirmed": [int(g.get("confirmed", 0)) for g in gens],
                "refuted": [int(g.get("refuted", 0)) for g in gens],
                "inconclusive": [int(g.get("inconclusive", 0)) for g in gens],
            },
        })

    # (4) Hypothesis-lineage tree — parent->child edges, gated verdicts per node.
    if reasoning_trace and reasoning_trace.get("lineage_edges"):
        gv = gated_verdict_by_id(findings)
        nodes = reasoning_trace.get("nodes") or {}
        # Only keep edges whose endpoints exist; label each node by its gated verdict
        # (defaulting to 'explored' when the gate excluded it, so nothing is over-claimed).
        edges = []
        node_labels: dict[str, str] = {}
        for e in reasoning_trace["lineage_edges"]:
            p, c = str(e.get("parent")), str(e.get("child"))
            if p in nodes and c in nodes:
                edges.append({"parent": p, "child": c})
                node_labels[p] = gv.get(p, "explored")
                node_labels[c] = gv.get(c, "explored")
        if edges:
            specs.append({
                "kind": "lineage_tree",
                "title": "Tree of inquiry",
                "caption": "Hypothesis lineage (parent to child). Node colour encodes the "
                           "gated verdict; nodes the evidence bar did not clear are shown as "
                           "`explored', never as confirmed.",
                "data": {"edges": edges[:120], "verdicts": node_labels},
            })

    return specs


# ── Research narrative / chain of reasoning ──────────────────────────────────

_VERDICT_HUMAN = {
    "confirmed": "was supported",
    "refuted": "was refuted",
    "inconclusive": "remained inconclusive",
}


def _lineage_paragraph(reasoning_trace: dict[str, Any], gv: dict[str, str]) -> str:
    """Describe the parent->child reasoning path using only gated verdicts."""
    nodes = reasoning_trace.get("nodes") or {}
    edges = reasoning_trace.get("lineage_edges") or []
    if not edges:
        return ""
    # Report a few representative lineage steps rooted at gated (decisive) parents.
    sentences: list[str] = []
    for e in edges:
        p, c = str(e.get("parent")), str(e.get("child"))
        if p not in nodes or c not in nodes:
            continue
        pv = gv.get(p)
        if pv not in ("confirmed", "refuted"):
            continue  # only narrate expansions off decisive, gated parents
        parent_txt = _latex_escape((nodes[p].get("text") or "")[:120].rstrip("."))
        child_txt = _latex_escape((nodes[c].get("text") or "")[:120].rstrip("."))
        exp = _latex_escape(str(e.get("expansion_type") or "follow-up"))
        cv = gv.get(c, "explored")
        cv_h = _VERDICT_HUMAN.get(cv, "was explored")
        sentences.append(
            f"The finding that ``{parent_txt}'' ({_VERDICT_HUMAN.get(pv, 'was evaluated')}) "
            f"prompted a {exp} hypothesis, ``{child_txt}'', which {cv_h}."
        )
        if len(sentences) >= 4:
            break
    if not sentences:
        return ""
    return "\\paragraph{Hypothesis lineage.} " + " ".join(sentences)


def _belief_paragraph(reasoning_trace: dict[str, Any]) -> str:
    beliefs = reasoning_trace.get("beliefs") or {}
    active = beliefs.get("active") or []
    closed = beliefs.get("closed") or []
    if not active and not closed:
        return ""
    parts: list[str] = ["\\paragraph{Belief evolution.}"]
    if active:
        stmts = []
        for b in active[:3]:
            s = _latex_escape(str(b.get("statement") or "")[:200].rstrip("."))
            conf = _latex_escape(str(b.get("confidence") or "unclear"))
            if s:
                stmts.append(f"``{s}'' (held with {conf} confidence)")
        if stmts:
            parts.append(
                "By the end of the campaign the system held the following working beliefs: "
                + "; ".join(stmts) + "."
            )
    if closed:
        ruled = []
        for c in closed[:3]:
            s = _latex_escape(str(c.get("statement") or "")[:160].rstrip("."))
            r = _latex_escape(str(c.get("reason") or "")[:140].rstrip("."))
            if s:
                ruled.append(f"``{s}'' ({r})" if r else f"``{s}''")
        if ruled:
            parts.append(
                "The following rival directions were abandoned once the evidence turned "
                "against them: " + "; ".join(ruled) + "."
            )
    return " ".join(parts) if len(parts) > 1 else ""


def _rival_paragraph(findings: dict[str, Any]) -> str:
    """What was ruled out and why — grounded in the gated refuted bucket."""
    refuted = findings.get("refuted") or []
    if not refuted:
        return ""
    items = []
    for f in refuted[:4]:
        claim = _latex_escape((f.get("text") or f.get("key_finding") or "")[:150].rstrip("."))
        stats = f.get("stats_text") or ""
        tail = f" ({stats})" if stats and stats != "no inferential statistic recorded" else ""
        if claim:
            items.append(f"``{claim}''{tail}")
    if not items:
        return ""
    return (
        "\\paragraph{Rival hypotheses ruled out.} The campaign actively falsified "
        f"{len(refuted)} competing hypothes{'is' if len(refuted) == 1 else 'es'}, "
        "narrowing the space of viable explanations. These included: "
        + "; ".join(items) + "."
    )


def _synthetic_provenance_paragraph(findings: dict[str, Any]) -> str:
    """State plainly that reported findings rest on a synthetic (illustrative) dataset.

    DOM2 honesty fix: findings whose evidence carries
    ``data_provenance == "synthetic"`` run on a locally seed-generated dataset (the
    genomics / graph-invariants / enzyme-kinetics demo domains present under real
    dataset names such as GTEx, SNAP, and BRENDA). Such a finding must never read as
    a real-world result; this limitations paragraph makes the illustrative status
    explicit in the narrative.
    """
    synth = [
        f
        for verdict in ("confirmed", "refuted")
        for f in (findings.get(verdict) or [])
        if _is_synthetic(f)
    ]
    if not synth:
        return ""
    claims = "; ".join(
        f"``{_latex_escape((f.get('key_finding') or f.get('text') or '')[:120].rstrip('.'))}''"
        for f in synth[:3]
    )
    return (
        "\\paragraph{Data provenance (synthetic).} "
        f"{len(synth)} of the reported finding{'s' if len(synth) != 1 else ''} were computed on a "
        "\\emph{synthetic dataset (illustrative)} --- a locally seed-generated frame produced by "
        "the domain adapter, not the real public dataset whose name it borrows. These results "
        "demonstrate the pipeline end-to-end but must not be read as real-world scientific claims, "
        "and any relationship they exhibit may reflect the data generator rather than nature. "
        f"Affected finding(s): {claims}."
    )


def research_narrative_section(
    findings: dict[str, Any],
    *,
    reasoning_trace: dict[str, Any] | None = None,
    question: str = "",
) -> str:
    """The deterministic Chain-of-Reasoning section, grounded in the gated trace.

    Structure: how the question was decomposed (rounds), the hypothesis lineage
    (parent -> child, gated verdicts), how beliefs evolved / which rivals were ruled
    out. Every outcome word maps to the gated verdict; no inconclusive/absent finding
    is presented as a result.
    """
    counts = findings.get("counts") or {}
    if counts.get("tested", 0) == 0:
        return (
            "\\section{Research Narrative}\n"
            "No hypotheses were executed, so there is no reasoning trace to report.\n"
        )
    rt = reasoning_trace or {}
    gv = gated_verdict_by_id(findings)

    parts: list[str] = ["\\section{Research Narrative}"]

    # Opening: how the question was approached across rounds.
    n_gen = int(rt.get("max_generation", 0)) + 1 if rt.get("generations") else 1
    round_phrase = "a single round" if n_gen <= 1 else f"{n_gen} successive rounds"
    q_esc = _latex_escape((question or "the research question")[:300])
    n_ref = int(counts.get("refuted", 0))
    ref_verb = "was falsified" if n_ref == 1 else "were falsified"
    n_inc = int(counts.get("inconclusive", 0))
    parts.append(
        f"This section reconstructs how the campaign arrived at its conclusions for "
        f"{q_esc}. The system proceeded over {round_phrase} of autonomous investigation. "
        f"In each round it generated falsifiable hypotheses, dispatched each to an "
        f"independent experimental agent, and let the gated outcomes of one round seed "
        f"the next. Of the {counts.get('tested', 0)} hypotheses that were executed and "
        f"tested, {counts.get('confirmed', 0)} cleared the evidence bar, "
        f"{n_ref} {ref_verb}, and {n_inc} remained inconclusive."
    )

    # DOM2: if any reported finding rests on a synthetic (seed-generated) dataset,
    # state that plainly in the narrative before any lineage/rival discussion, so a
    # reader can never mistake an illustrative result for a real-world one.
    synth = _synthetic_provenance_paragraph(findings)
    if synth:
        parts.append(synth)

    lin = _lineage_paragraph(rt, gv)
    if lin:
        parts.append(lin)

    rival = _rival_paragraph(findings)
    if rival:
        parts.append(rival)

    belief = _belief_paragraph(rt)
    if belief:
        parts.append(belief)

    # Closing: what the reasoning path leaves standing.
    if counts.get("confirmed", 0) > 0:
        parts.append(
            "\\paragraph{Where the reasoning lands.} The confirmed findings above are the "
            "claims that survived this process of hypothesis generation, experimental test, "
            "and falsification of rivals. They are reported with their statistical evidence "
            "in the Results section and should be read as the endpoint of the reasoning path, "
            "not as isolated positive results."
        )
    else:
        parts.append(
            "\\paragraph{Where the reasoning lands.} No hypothesis survived the full process "
            "under the evidence bar applied here. The reasoning path is therefore an honest "
            "negative result: it constrains, rather than resolves, the question, and the "
            "inconclusive directions summarised in the Results section scope the natural "
            "follow-up."
        )
    return "\n\n".join(parts)
