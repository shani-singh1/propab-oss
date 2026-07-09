"""Tests for the B3 biology analysis tools — correctness + honesty invariants.

`differential_expression` and `enrichment_analysis` exist to make the honesty rules
non-optional: significance is decided on the BH FDR q-value (never a raw p), a
significant call always carries an effect size, degenerate inputs are excluded with a
reason (not silently scored), and an implicit/incomplete background is rejected. These
tests assert those invariants, not just the happy path.
"""
from __future__ import annotations

from propab.tools.registry import ToolRegistry

_R = ToolRegistry()


def call(name, **params):
    return _R.call(name, params)


# ── fixtures ─────────────────────────────────────────────────────────────────
_CTRL6 = [10, 11, 10, 12, 10, 11]          # n=6 control samples
_LABELS = ["ctrl"] * 6 + ["case"] * 6       # 6 ctrl then 6 case


def _row(ctrl, case):
    return list(ctrl) + list(case)


def _de_matrix():
    """A planted STRONG up-regulated gene, a WEAK raw-significant-only gene, 30 null
    genes, and a zero-variance FLAT gene."""
    rows, names = [], []
    rows.append(_row(_CTRL6, [c + 15 for c in _CTRL6])); names.append("STRONG")     # huge, real
    rows.append(_row(_CTRL6, [c + 1.2 for c in _CTRL6])); names.append("WEAK")       # raw p<0.05 only
    base = [5, 6, 5, 6, 5, 6]
    for i in range(30):
        rows.append(_row(base, base)); names.append(f"NULL{i}")                       # p == 1
    rows.append(_row([7] * 6, [7] * 6)); names.append("FLAT")                         # zero variance
    return rows, names


# ── differential_expression: planted DE recovered, nulls not significant ─────
def test_de_recovers_planted_gene_and_nulls_not_significant():
    rows, names = _de_matrix()
    r = call("differential_expression", expression=rows, group_labels=_LABELS,
             gene_names=names, test="welch_t", alpha=0.05,
             case_label="case", control_label="ctrl")
    assert r.success
    o = r.output
    by = {row["gene"]: row for row in o["results"]}

    strong = by["STRONG"]
    assert strong["significant"] is True          # recovered after FDR
    assert strong["q_value"] < 0.05
    assert strong["log2_fold_change"] > 0          # correct FC sign (case up vs control)
    assert strong["effect_size"] is not None       # effect size reported alongside p

    # Exactly the one planted gene is significant; the 30 nulls are not.
    assert o["n_significant"] == 1
    assert by["NULL0"]["significant"] is False
    assert by["NULL0"]["q_value"] >= 0.05


# ── honesty invariant: FDR gates significance, not raw p ─────────────────────
def test_de_fdr_gates_significance_not_raw_p():
    rows, names = _de_matrix()
    r = call("differential_expression", expression=rows, group_labels=_LABELS,
             gene_names=names, test="welch_t", alpha=0.05,
             case_label="case", control_label="ctrl")
    o = r.output
    by = {row["gene"]: row for row in o["results"]}

    # The WEAK gene is significant at RAW p but NOT after BH-FDR -> must not be called.
    weak = by["WEAK"]
    assert weak["p_value"] < 0.05                  # would be a "finding" on raw p
    assert weak["q_value"] >= 0.05                  # but not after multiple-testing
    assert weak["significant"] is False             # the tool refuses the raw-p call

    # Global invariants across every tested gene.
    for row in o["results"]:
        if row["significant"]:
            assert row["q_value"] < 0.05                     # never significant on raw p
            assert row["effect_size"] is not None            # never significant without effect size
        if row["q_value"] >= 0.05:
            assert row["significant"] is False               # q gate is authoritative


# ── honesty invariant: degenerate genes handled, effect size present ─────────
def test_de_zero_variance_gene_excluded_with_reason():
    rows, names = _de_matrix()
    r = call("differential_expression", expression=rows, group_labels=_LABELS,
             gene_names=names, test="welch_t", alpha=0.05,
             case_label="case", control_label="ctrl")
    o = r.output
    excluded_names = {e["gene"] for e in o["excluded"]}
    assert "FLAT" in excluded_names                          # zero-variance gene handled
    reason = next(e["reason"] for e in o["excluded"] if e["gene"] == "FLAT")
    assert "zero_variance" in reason
    # The excluded gene is kept OUT of the tested/FDR family.
    assert "FLAT" not in {row["gene"] for row in o["results"]}
    assert o["n_genes_tested"] == len(names) - len(excluded_names)


# ── differential_expression: Mann-Whitney path recovers a planted gene ───────
def test_de_mann_whitney_recovers_and_reports_effect():
    strong = _row(_CTRL6, [c + 15 for c in _CTRL6])
    nulls = [_row(v, v) for v in ([5, 6, 5, 6, 5, 6], [4, 5, 4, 5, 4, 5], [3, 4, 3, 4, 3, 4])]
    r = call("differential_expression", expression=[strong, *nulls], group_labels=_LABELS,
             test="mann_whitney", alpha=0.05, case_label="case", control_label="ctrl")
    assert r.success
    top = r.output["results"][0]
    assert top["significant"] is True
    assert top["q_value"] < 0.05
    assert r.output["effect_size_metric"] == "rank_biserial"
    assert top["effect_size"] is not None
    assert r.output["n_significant"] == 1


# ── differential_expression: bad input -> validation_error (never fabricate) ─
def test_de_one_sample_group_is_validation_error():
    # A 1-sample group cannot yield a variance-based test.
    r = call("differential_expression",
             expression=[[1.0, 2.0, 3.0, 4.0]],
             group_labels=["a", "b", "b", "b"])
    assert not r.success and r.error.type == "validation_error"


def test_de_not_two_groups_is_validation_error():
    r = call("differential_expression",
             expression=[[1.0, 2.0, 3.0, 4.0]],
             group_labels=["a", "a", "a", "a"])          # only one group
    assert not r.success and r.error.type == "validation_error"


def test_de_missing_expression_is_validation_error():
    r = call("differential_expression", group_labels=_LABELS)
    assert not r.success and r.error.type == "validation_error"


# ── enrichment_analysis: planted enriched set recovered ──────────────────────
def _enrichment_inputs():
    bg = [f"G{i}" for i in range(100)]
    query = [f"G{i}" for i in range(10)]                 # subset of bg
    gene_sets = {
        "enriched": [f"G{i}" for i in range(8)] + ["G50", "G51"],   # 8/10 query hit
        "random1": [f"G{i}" for i in range(30, 46)],                # no query overlap
        "random2": ["G60", "G61", "G62"],
        "off_universe": ["X1", "X2"],                               # not in background
    }
    return query, bg, gene_sets


def test_enrichment_recovers_planted_set():
    query, bg, gene_sets = _enrichment_inputs()
    r = call("enrichment_analysis", query_genes=query, background_genes=bg,
             gene_sets=gene_sets, alpha=0.05)
    assert r.success
    o = r.output
    top = o["results"][0]
    assert top["set_name"] == "enriched"
    assert top["significant"] is True
    assert top["q_value"] < 0.05
    assert top["fold_enrichment"] > 1.0                  # over-representation is visible
    assert top["overlap"] == 8
    assert o["n_significant"] == 1

    # A set with no members in the background is skipped with a reason, not scored.
    skipped_names = {s["set_name"] for s in o["skipped"]}
    assert "off_universe" in skipped_names
    assert "off_universe" not in {row["set_name"] for row in o["results"]}


# ── enrichment_analysis: background-subset guard fires ───────────────────────
def test_enrichment_rejects_query_not_subset_of_background():
    _, bg, gene_sets = _enrichment_inputs()
    r = call("enrichment_analysis",
             query_genes=["G0", "NOT_IN_BACKGROUND"],    # leaks outside the universe
             background_genes=bg, gene_sets=gene_sets, alpha=0.05)
    assert not r.success and r.error.type == "validation_error"


# ── enrichment_analysis: degenerate inputs -> validation_error ───────────────
def test_enrichment_empty_query_is_validation_error():
    _, bg, gene_sets = _enrichment_inputs()
    r = call("enrichment_analysis", query_genes=[], background_genes=bg, gene_sets=gene_sets)
    assert not r.success and r.error.type == "validation_error"


def test_enrichment_empty_background_is_validation_error():
    query, _, gene_sets = _enrichment_inputs()
    r = call("enrichment_analysis", query_genes=query, background_genes=[], gene_sets=gene_sets)
    assert not r.success and r.error.type == "validation_error"


def test_enrichment_bad_gene_sets_is_validation_error():
    query, bg, _ = _enrichment_inputs()
    r = call("enrichment_analysis", query_genes=query, background_genes=bg, gene_sets=["not", "a", "dict"])
    assert not r.success and r.error.type == "validation_error"


# ── registry: both tools register and are visible to the worker ──────────────
def test_bio_tools_registered_for_worker():
    names = {s["name"] for s in _R.get_for("worker")}
    assert "differential_expression" in names
    assert "enrichment_analysis" in names
    cluster = {s["name"] for s in _R.get_cluster("biology")}
    assert {"differential_expression", "enrichment_analysis"} <= cluster
