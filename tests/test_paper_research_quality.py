"""
Research-paper structure + honesty-gate regression tests.

Covers the reworked paper-generation layer:
  * the compiled manuscript contains every required section,
  * figures/tables/narrative are derived from real gated trace rows,
  * the honesty gate holds — an inconclusive finding and a confirmed control never
    appear as a result in ANY section, table, or figure.

Run from the worktree root so the edited propab-core is the code under test:
    PYTHONPATH="packages/propab-core;." python -m pytest tests/test_paper_research_quality.py -q
"""
import asyncio

import propab.paper_compiler as pc
from propab.paper_compiler import (
    _effective_verdict,
    compile_references_latex,
    compile_session_findings,
)
from propab.paper_narrative import (
    build_figure_specs,
    findings_table,
    gated_verdict_by_id,
    research_narrative_section,
    summary_counts_table,
)
from propab.paper_sections import render_paper_tex


# ── Synthetic trace rows (the shape _fetch_result_rows returns from the DB) ───

def _confirmed_metric_row(hid: str, rank: int, round_n: int = 0) -> dict:
    return {
        "id": hid,
        "rank": rank,
        "text": "Higher width improves validation accuracy on the target task",
        "verdict": "confirmed",
        "confidence": 0.82,
        "evidence_summary": (
            'evidence={"n_metric_steps": 3, "metric_value": 0.87, "effect_size": 0.6, '
            '"p_value": 0.003, "verified_true_steps": 3}; significance={}; steps=4.'
        ),
        "key_finding": "Wider hidden layers raise validation accuracy",
        "tool_trace_id": None,
        "round_number": round_n,
        "step_count": 4,
    }


def _refuted_row(hid: str, rank: int, round_n: int = 1) -> dict:
    return {
        "id": hid,
        "rank": rank,
        "text": "Dropout above 0.5 improves generalization on this dataset",
        "verdict": "refuted",
        "confidence": 0.4,
        "evidence_summary": (
            'evidence={"n_metric_steps": 2, "metric_value": 0.61, "effect_size": -0.3, '
            '"p_value": 0.02, "verified_false_steps": 2, "verdict_reason": "counterexample found"}; '
            "significance={}; steps=3."
        ),
        "key_finding": "",
        "tool_trace_id": None,
        "round_number": round_n,
        "step_count": 3,
    }


def _inconclusive_confirmed_no_metric_row(hid: str, rank: int, round_n: int = 0) -> dict:
    """DB says 'confirmed' but there is NO metric and NO verified step -> gate = inconclusive."""
    return {
        "id": hid,
        "rank": rank,
        "text": "Some speculative effect that was never actually measured",
        "verdict": "confirmed",
        "confidence": 0.9,
        "evidence_summary": 'evidence={"n_metric_steps": 0}; significance={}; steps=1.',
        "key_finding": "Speculative unmeasured effect",
        "tool_trace_id": None,
        "round_number": round_n,
        "step_count": 1,
    }


def _confirmed_control_row(hid: str, rank: int, round_n: int = 0) -> dict:
    """A control/null hypothesis the DB marked 'confirmed' -> gate must exclude from findings."""
    return {
        "id": hid,
        "rank": rank,
        "text": "Null hypothesis: no statistically significant effect beyond noise",
        "verdict": "confirmed",
        "confidence": 0.95,
        "evidence_summary": (
            'evidence={"n_metric_steps": 3, "metric_value": 0.5, "verified_true_steps": 3}; '
            "significance={}; steps=3."
        ),
        "key_finding": "Control calibrated as expected",
        "tool_trace_id": None,
        "round_number": round_n,
        "step_count": 3,
    }


def _confirmed_synthetic_row(hid: str, rank: int, round_n: int = 0) -> dict:
    """A confirmed finding whose evidence carries data_provenance='synthetic'.

    This mirrors the plugin verification path (genomics/graph_invariants/enzyme),
    which stores evidence_summary as a bare json.dumps(output) with no ``evidence=``
    prefix and stamps ``data_provenance``.
    """
    return {
        "id": hid,
        "rank": rank,
        "text": "Tissue-specificity tau predicts cross-tissue expression on the GTEx subset",
        "verdict": "confirmed",
        "confidence": 0.88,
        # Bare JSON object (plugin path), including data_provenance and a verified step.
        "evidence_summary": (
            '{"lofo_r2": 0.42, "verification_method": "leave_tissue_out", '
            '"verified_true_steps": 1, "metric_value": 0.42, "data_provenance": "synthetic"}'
        ),
        "key_finding": "Tau index carries cross-tissue signal (synthetic GTEx)",
        "tool_trace_id": None,
        "round_number": round_n,
        "step_count": 2,
    }


def _unexecuted_row(hid: str, rank: int, round_n: int = 2) -> dict:
    return {
        "id": hid,
        "rank": rank,
        "text": "A hypothesis that was generated but never executed",
        "verdict": None,
        "confidence": None,
        "evidence_summary": None,
        "key_finding": "",
        "tool_trace_id": None,
        "round_number": round_n,
        "step_count": 0,
    }


def _gated_findings(monkeypatch, rows: list[dict]) -> dict:
    """Run the REAL gate (compile_session_findings -> _effective_verdict) over synthetic rows."""

    async def _fake_fetch(session_factory, session_id):  # noqa: ANN001
        return rows

    monkeypatch.setattr(pc, "_fetch_result_rows", _fake_fetch)
    return asyncio.run(compile_session_findings(session_factory=None, session_id="sid"))


_REASONING_TRACE = {
    "nodes": {
        "h1": {"id": "h1", "text": "Higher width improves validation accuracy", "parent_id": None,
               "depth": 0, "generation": 0, "verdict": "confirmed", "expansion_type": None,
               "node_role": "DISCOVERY", "primary_theme": "capacity", "mechanism": None, "confidence": 0.82},
        "h2": {"id": "h2", "text": "Dropout above 0.5 improves generalization", "parent_id": "h1",
               "depth": 1, "generation": 1, "verdict": "refuted", "expansion_type": "boundary",
               "node_role": "DISCOVERY", "primary_theme": "regularization", "mechanism": None, "confidence": 0.4},
        "hc": {"id": "hc", "text": "Null hypothesis: no effect beyond random", "parent_id": None,
               "depth": 0, "generation": 0, "verdict": "confirmed", "expansion_type": None,
               "node_role": "CONTROL", "primary_theme": "control", "mechanism": None, "confidence": 0.95},
    },
    "lineage_edges": [{"parent": "h1", "child": "h2", "expansion_type": "boundary"}],
    "beliefs": {
        "active": [{"statement": "Model capacity is the dominant factor", "confidence": "strong"}],
        "closed": [{"statement": "Heavy dropout helps", "reason": "refuted after two rounds"}],
        "recent_activity": "",
        "branch_exhausted": False,
    },
    "generations": [
        {"generation": 0, "confirmed": 1, "refuted": 0, "inconclusive": 1, "pending": 0},
        {"generation": 1, "confirmed": 0, "refuted": 1, "inconclusive": 0, "pending": 0},
    ],
    "max_depth": 1,
    "max_generation": 1,
}


# ── Structure: all required sections present ─────────────────────────────────

def test_compiled_paper_contains_all_required_sections(monkeypatch) -> None:
    rows = [_confirmed_metric_row("h1", 1), _refuted_row("h2", 2)]
    findings = _gated_findings(monkeypatch, rows)

    from propab.paper_compiler import compile_session_results_latex

    async def _fake_fetch(session_factory, session_id):  # noqa: ANN001
        return rows

    monkeypatch.setattr(pc, "_fetch_result_rows", _fake_fetch)
    results_tex = asyncio.run(
        compile_session_results_latex(None, "sid", baseline=0.80, findings=findings)
    )
    methods = asyncio.run(_methods(monkeypatch, rows))
    narrative = research_narrative_section(findings, reasoning_trace=_REASONING_TRACE, question="Does width help?")
    refs = compile_references_latex({"key_papers": [{"paper_id": "1234.5678", "title": "Prior work"}]})

    tex = render_paper_tex(
        title="A study of width",
        abstract="We investigate width. Across 2 hypotheses evaluated ...",
        introduction="Introduction prose.",
        methods_tex=methods,
        results_tex=results_tex,
        figures_tex="\\begin{figure}[ht]\\end{figure}",
        narrative_tex=narrative,
        discussion="Discussion prose. \\paragraph{Threats to validity.} ...",
        conclusion="Conclusion prose.",
        references_tex=refs,
    )
    assert "\\begin{abstract}" in tex
    assert "\\section{Introduction}" in tex
    assert "\\section{Methods}" in tex
    assert "Verification protocol" in tex  # methods describes the evidence bar
    assert "\\section{Results}" in tex
    assert "\\section{Discussion}" in tex
    assert "Threats to validity" in tex
    assert "\\section{Research Narrative}" in tex
    assert "\\section{References}" in tex


async def _methods(monkeypatch, rows):  # helper: compile methods with a stubbed DB
    from propab.paper_compiler import compile_session_methods_latex

    async def _fake_fetch(session_factory, session_id):  # noqa: ANN001
        return rows

    async def _fake_agg(session_factory, session_id):  # noqa: ANN001
        return {"tools": {"train_model": 3, "statistical_significance": 2}, "rounds": 2, "code_steps": 1}

    monkeypatch.setattr(pc, "_fetch_result_rows", _fake_fetch)
    monkeypatch.setattr(pc, "_aggregate_tool_usage", _fake_agg)
    out = await compile_session_methods_latex(None, "sid")
    return out["combined_latex"]


# ── Tables + figures derive from real gated rows ─────────────────────────────

def test_tables_and_figures_from_real_gated_rows(monkeypatch) -> None:
    rows = [_confirmed_metric_row("h1", 1), _refuted_row("h2", 2)]
    findings = _gated_findings(monkeypatch, rows)

    counts_tab = summary_counts_table(findings["counts"])
    assert "\\begin{tabular}" in counts_tab
    assert "Confirmed" in counts_tab

    ftab = findings_table(findings, baseline=0.80)
    assert "\\begin{tabular}" in ftab
    # A real confirmed finding's metric (0.87) vs the baseline (0.80) appears.
    assert "0.87" in ftab and "0.8" in ftab
    assert "Confirmed" in ftab and "Refuted" in ftab

    specs = build_figure_specs(findings, reasoning_trace=_REASONING_TRACE, baseline=0.80, metric_name="val accuracy")
    kinds = {s["kind"] for s in specs}
    assert "outcome_bars" in kinds
    assert "metric_vs_baseline" in kinds  # confirmed finding carried a metric
    assert "rounds_timeline" in kinds
    # outcome bars reflect the gated counts exactly
    bars = next(s for s in specs if s["kind"] == "outcome_bars")
    assert bars["data"]["values"] == [
        findings["counts"]["confirmed"],
        findings["counts"]["refuted"],
        findings["counts"]["inconclusive"],
    ]


# ── HONESTY REGRESSION: inconclusive + confirmed control never shown as result ──

def test_honesty_inconclusive_and_control_never_appear_as_results(monkeypatch) -> None:
    rows = [
        _confirmed_metric_row("hgood", 1),                 # legitimate confirmed finding
        _inconclusive_confirmed_no_metric_row("hincon", 2),  # DB 'confirmed' but no metric
        _confirmed_control_row("hctrl", 3),                # DB 'confirmed' control/null
        _unexecuted_row("hnull", 4),                       # never executed
    ]

    # 1) The gate itself demotes / excludes correctly.
    assert _effective_verdict(rows[1]) == "inconclusive"       # no-metric confirmed -> inconclusive
    assert _effective_verdict(rows[2]) == "inconclusive"       # control -> inconclusive
    assert _effective_verdict(rows[3]) == "unexecuted"         # excluded from counts

    findings = _gated_findings(monkeypatch, rows)

    # The confirmed bucket must contain ONLY the legitimate finding.
    confirmed_ids = {f["id"] for f in findings["confirmed"]}
    assert confirmed_ids == {"hgood"}
    # Control is excluded from findings entirely (not even in inconclusive as a discovery).
    all_finding_ids = {
        f["id"] for b in ("confirmed", "refuted", "inconclusive") for f in findings[b]
    }
    assert "hctrl" not in all_finding_ids
    # Unexecuted is not tested.
    assert "hnull" not in all_finding_ids
    assert findings["counts"]["unexecuted"] == 1

    speculative = "Speculative unmeasured effect"
    control_text = "Control calibrated as expected"
    null_marker = "Null hypothesis"

    # 2) Findings TABLE: only the confirmed finding, none of the excluded ones.
    ftab = findings_table(findings, baseline=0.80)
    assert "Wider hidden layers" in ftab or "width" in ftab.lower()
    assert speculative not in ftab
    assert control_text not in ftab
    assert null_marker not in ftab

    # 3) FIGURES: metric-vs-baseline plots only the gated confirmed finding.
    specs = build_figure_specs(findings, reasoning_trace=_REASONING_TRACE, baseline=0.80)
    mvb = [s for s in specs if s["kind"] == "metric_vs_baseline"]
    assert mvb, "confirmed finding with a metric should yield a metric figure"
    plotted_labels = " ".join(p["label"] for p in mvb[0]["data"]["points"])
    assert speculative not in plotted_labels
    assert control_text not in plotted_labels
    # Outcome bars: the confirmed count is exactly 1 (only hgood), not 3.
    bars = next(s for s in specs if s["kind"] == "outcome_bars")
    assert bars["data"]["values"][0] == 1

    # 4) Lineage figure: the control node is never coloured 'confirmed'.
    gv = gated_verdict_by_id(findings)
    assert gv.get("hctrl") != "confirmed"  # control not a confirmed result
    lineage = [s for s in specs if s["kind"] == "lineage_tree"]
    if lineage:
        for nid, v in lineage[0]["data"]["verdicts"].items():
            if nid == "hc":  # the control node id in the reasoning trace
                assert v != "confirmed"

    # 5) NARRATIVE: never presents the inconclusive/control text as a supported result.
    narrative = research_narrative_section(findings, reasoning_trace=_REASONING_TRACE, question="Q?")
    assert speculative not in narrative
    assert control_text not in narrative
    # It may mention the control was NOT confirmed, but must not say it 'was supported'.
    assert "no effect beyond random`` was supported" not in narrative.lower()
    # The confirmed count stated in the narrative is 1.
    assert "1 cleared the evidence bar" in narrative


def test_narrative_honest_when_zero_confirmed(monkeypatch) -> None:
    rows = [_inconclusive_confirmed_no_metric_row("hincon", 1), _confirmed_control_row("hctrl", 2)]
    findings = _gated_findings(monkeypatch, rows)
    assert findings["counts"]["confirmed"] == 0
    narrative = research_narrative_section(findings, reasoning_trace=_REASONING_TRACE, question="Q?")
    assert "honest negative result" in narrative
    # No figure claims a confirmed metric when there are no gated confirmations.
    specs = build_figure_specs(findings, reasoning_trace=_REASONING_TRACE)
    assert not [s for s in specs if s["kind"] == "metric_vs_baseline"]


# ── DOM2: synthetic-data provenance labelled honestly end-to-end ─────────────

_SYNTH_LABEL = "synthetic dataset (illustrative)"


def test_parse_evidence_extracts_synthetic_provenance() -> None:
    """Both the plugin bare-JSON path and the evidence= path surface provenance."""
    bare = (
        '{"lofo_r2": 0.42, "verified_true_steps": 1, "metric_value": 0.42, '
        '"data_provenance": "synthetic"}'
    )
    assert pc.parse_evidence(bare)["data_provenance"] == "synthetic"
    prefixed = 'evidence={"n_metric_steps": 3, "metric_value": 0.5, "data_provenance": "synthetic"}; steps=3.'
    assert pc.parse_evidence(prefixed)["data_provenance"] == "synthetic"
    # A real-data finding carries no provenance marker.
    real = 'evidence={"n_metric_steps": 3, "metric_value": 0.87, "verified_true_steps": 3}; steps=4.'
    assert pc.parse_evidence(real)["data_provenance"] is None


def test_synthetic_finding_is_gated_confirmed_and_carries_provenance(monkeypatch) -> None:
    """A confirmed synthetic finding still passes the honesty gate (labelling, not blocking)."""
    rows = [_confirmed_synthetic_row("hsyn", 1)]
    findings = _gated_findings(monkeypatch, rows)
    assert findings["counts"]["confirmed"] == 1
    f = findings["confirmed"][0]
    assert f["id"] == "hsyn"
    # provenance rides on both the finding dict and its parsed stats
    assert f.get("data_provenance") == "synthetic"
    assert (f.get("stats") or {}).get("data_provenance") == "synthetic"


def test_synthetic_finding_labelled_in_table_narrative_and_methods(monkeypatch) -> None:
    from propab.paper_compiler import compile_session_methods_latex

    rows = [_confirmed_synthetic_row("hsyn", 1)]
    findings = _gated_findings(monkeypatch, rows)

    # (1) Findings TABLE: row marked + caption disclosure.
    ftab = findings_table(findings, baseline=None)
    assert _SYNTH_LABEL in ftab
    assert "illustrative" in ftab.lower()

    # (2) NARRATIVE: dedicated synthetic-provenance limitations paragraph.
    narrative = research_narrative_section(findings, reasoning_trace=_REASONING_TRACE, question="Q?")
    assert "Data provenance" in narrative
    assert "synthetic" in narrative.lower()

    # (3) METHODS/limitations: discloses the synthetic dataset limitation.
    async def _fake_fetch(session_factory, session_id):  # noqa: ANN001
        return rows

    async def _fake_agg(session_factory, session_id):  # noqa: ANN001
        return {"tools": {"genomics_verification": 1}, "rounds": 1, "code_steps": 0}

    monkeypatch.setattr(pc, "_fetch_result_rows", _fake_fetch)
    monkeypatch.setattr(pc, "_aggregate_tool_usage", _fake_agg)
    methods = asyncio.run(compile_session_methods_latex(None, "sid"))["combined_latex"]
    assert _SYNTH_LABEL in methods
    assert "Data provenance and limitations" in methods


def test_real_data_finding_is_not_labelled_synthetic(monkeypatch) -> None:
    """A real-data confirmed finding must NOT be tagged synthetic anywhere."""
    from propab.paper_compiler import compile_session_methods_latex, compile_session_results_latex

    rows = [_confirmed_metric_row("hreal", 1)]
    findings = _gated_findings(monkeypatch, rows)
    assert findings["confirmed"][0].get("data_provenance") is None

    ftab = findings_table(findings, baseline=0.80)
    assert _SYNTH_LABEL not in ftab

    narrative = research_narrative_section(findings, reasoning_trace=_REASONING_TRACE, question="Q?")
    assert "Data provenance (synthetic)" not in narrative

    async def _fake_fetch(session_factory, session_id):  # noqa: ANN001
        return rows

    async def _fake_agg(session_factory, session_id):  # noqa: ANN001
        return {"tools": {"train_model": 1}, "rounds": 1, "code_steps": 0}

    monkeypatch.setattr(pc, "_fetch_result_rows", _fake_fetch)
    monkeypatch.setattr(pc, "_aggregate_tool_usage", _fake_agg)
    methods = asyncio.run(compile_session_methods_latex(None, "sid"))["combined_latex"]
    assert _SYNTH_LABEL not in methods

    monkeypatch.setattr(pc, "_fetch_result_rows", _fake_fetch)
    results = asyncio.run(
        compile_session_results_latex(None, "sid", baseline=0.80, findings=findings)
    )
    assert _SYNTH_LABEL not in results


def test_synthetic_label_appears_in_results_prose(monkeypatch) -> None:
    """The Results 'Supported findings' prose sentence marks a synthetic finding."""
    from propab.paper_compiler import compile_session_results_latex

    rows = [_confirmed_synthetic_row("hsyn", 1)]
    findings = _gated_findings(monkeypatch, rows)

    async def _fake_fetch(session_factory, session_id):  # noqa: ANN001
        return rows

    monkeypatch.setattr(pc, "_fetch_result_rows", _fake_fetch)
    results = asyncio.run(
        compile_session_results_latex(None, "sid", baseline=None, findings=findings)
    )
    assert _SYNTH_LABEL in results
