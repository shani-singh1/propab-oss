"""B3 — differential expression (group-vs-group DE with MANDATORY FDR + effect size).

The single biggest source of false biology findings is calling a gene "significant"
off a raw per-gene p-value across thousands of simultaneous tests, and/or off a
p-value with no regard for whether the effect is real-sized. This tool refuses to do
either.

Given an expression matrix (genes x samples) and a two-group label vector, it runs a
per-gene two-group test (Welch's t or Mann-Whitney U, caller-selectable), and returns
for every testable gene: the log2 fold-change, an EFFECT SIZE (Cohen's d for the
t-test, rank-biserial correlation for Mann-Whitney), the raw p-value, and the
**Benjamini-Hochberg FDR-adjusted q-value** computed across all tested genes. Rows are
sorted by q.

Honesty by construction (this is the whole point):
  * **FDR is mandatory.** Significance is decided on the BH q-value, NEVER on raw p.
    ``n_significant`` counts genes with q < alpha only.
  * **No significance without an effect size.** A gene is flagged ``significant`` only
    when ``q < alpha`` AND a finite effect size is reported. A ``small_effect`` flag
    marks a significant-but-tiny effect (|log2FC| below ``min_abs_log2fc``) so the
    "significant but meaningless" trap is visible, never hidden.
  * **Degenerate genes are excluded with a reason, not silently passed.** A
    zero-variance (constant) gene, or a gene with no within-group variance for the
    t-test, cannot yield a valid p-value; it is dropped into ``excluded`` with a
    reason and kept OUT of the FDR family (so it neither borrows nor lends power).
  * **1-sample / missing groups fail loudly.** Fewer than 2 samples in a group, not
    exactly 2 groups, a length mismatch, or NaN/inf in the matrix -> validation_error,
    never a fabricated result.
  * BH is implemented exactly in numpy (sort, ``q_i = p_i*m/rank``, backward running
    minimum, capped at 1) — the same construction as the statistics-cluster tool.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from propab.tools.types import ToolError, ToolResult

_WELCH_ALIASES = {
    "welch_t", "welch", "welch_t_test", "welchs_t", "welch-t", "t", "t_test",
    "ttest", "two_sample_t", "student_t", "students_t", "independent_t",
}
_MWU_ALIASES = {
    "mann_whitney", "mann-whitney", "mannwhitney", "mann_whitney_u", "mannwhitneyu",
    "mwu", "u_test", "wilcoxon_rank_sum", "wilcoxon-rank-sum", "ranksum", "rank_sum",
    "wilcoxon",
}

TOOL_SPEC = {
    "name": "differential_expression",
    "domain": "biology",
    "audience": "worker",
    # Emits per-gene significance calls (p / q / effect size): surfaced to every
    # significance workflow AND never auto-filled from the spec example (that would
    # inject a placeholder expression matrix and manufacture fake DE calls).
    "significance_capable": True,
    "description": (
        "Per-gene differential expression between two sample groups. Provide "
        "'expression' (genes x samples matrix) and 'group_labels' (one label per "
        "sample, exactly two distinct groups). test in {welch_t (default), "
        "mann_whitney}. Returns per-gene log2 fold-change, an effect size (Cohen's d "
        "for welch_t, rank-biserial for mann_whitney), raw p, and Benjamini-Hochberg "
        "FDR q, sorted by q, plus n_significant at q<alpha. FDR is MANDATORY: a gene "
        "is 'significant' only when q<alpha AND a finite effect size is present; a "
        "tiny effect is flagged. Zero-variance / no-within-group-variance genes are "
        "excluded with a reason; <2 samples per group or not-two-groups -> "
        "validation_error. Nothing is ever called significant on a raw p-value."
    ),
    "params": {
        "expression": {
            "type": "list[list[float]]",
            "required": True,
            "description": "Expression matrix, rows = genes, cols = samples. Finite numbers only.",
        },
        "group_labels": {
            "type": "list",
            "required": True,
            "description": "One label per sample (matrix column). Exactly two distinct labels required.",
        },
        "test": {
            "type": "str",
            "required": False,
            "default": "welch_t",
            "description": "welch_t (default, unequal-variance t) | mann_whitney (rank-based, non-parametric).",
        },
        "alpha": {
            "type": "float",
            "required": False,
            "default": 0.05,
            "description": "FDR threshold in (0,1). A gene is significant iff BH q < alpha (never raw p).",
        },
        "gene_names": {
            "type": "list[str]",
            "required": False,
            "description": "Optional gene names, length = number of rows. Defaults to gene_0, gene_1, ...",
        },
        "case_label": {
            "type": "str",
            "required": False,
            "description": "Which group is the numerator of the fold-change (case/treatment). Defaults to the 2nd sorted label.",
        },
        "control_label": {
            "type": "str",
            "required": False,
            "description": "Which group is the denominator (control/reference). Defaults to the 1st sorted label.",
        },
        "data_is_log2": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "True if values are already log2-scale (log2FC = mean_case - mean_control). If False, values must be >=0 and log2FC = log2((mean_case+pseudocount)/(mean_control+pseudocount)).",
        },
        "pseudocount": {
            "type": "float",
            "required": False,
            "default": 1.0,
            "description": "Added to group means before the log2 ratio when data_is_log2 is False (avoids log of 0).",
        },
        "min_abs_log2fc": {
            "type": "float",
            "required": False,
            "default": 1.0,
            "description": "|log2FC| below this flags a significant gene as small_effect (visibility only; does not gate significance).",
        },
    },
    "output": {
        "test": "str — normalized test used (welch_t | mann_whitney)",
        "effect_size_metric": "str — cohens_d (welch_t) or rank_biserial (mann_whitney)",
        "alpha": "float — FDR level applied",
        "case_label": "str — fold-change numerator group",
        "control_label": "str — fold-change denominator group",
        "n_case": "int — samples in the case group",
        "n_control": "int — samples in the control group",
        "n_genes_total": "int — rows in the input matrix",
        "n_genes_tested": "int — genes that entered the FDR family",
        "n_genes_excluded": "int — degenerate genes dropped with a reason",
        "excluded": "list — [{gene, reason}] for degenerate/undefined genes (not FDR-corrected)",
        "results": "list — per tested gene {gene, log2_fold_change, effect_size, effect_size_metric, statistic, p_value, q_value, significant, small_effect, mean_case, mean_control}, sorted by q",
        "n_significant": "int — genes with q < alpha (each also carries a finite effect size)",
    },
    "example": {
        "params": {
            "expression": [[10, 11, 9, 30, 31, 29], [5, 6, 5, 5, 6, 5]],
            "group_labels": ["ctrl", "ctrl", "ctrl", "case", "case", "case"],
            "test": "welch_t",
            "alpha": 0.05,
            "case_label": "case",
            "control_label": "ctrl",
        },
        "output": {
            "n_genes_tested": 2,
            "n_significant": 1,
            "results": [{"gene": "gene_0", "log2_fold_change": 1.6, "significant": True}],
        },
    },
}


def _bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values over a raw p-vector (any order in, same order out).

    Sort ascending, q_i = p_i * m / rank_i, enforce monotone non-decreasing from the
    largest rank down (backward running minimum), cap at 1, map back to input order.
    """
    m = int(p.shape[0])
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = p_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return q


def differential_expression(
    expression: list | None = None,
    group_labels: list | None = None,
    test: str = "welch_t",
    alpha: float = 0.05,
    gene_names: list | None = None,
    case_label=None,
    control_label=None,
    data_is_log2: bool = False,
    pseudocount: float = 1.0,
    min_abs_log2fc: float = 1.0,
) -> ToolResult:
    # ---- Required inputs present? (never fabricate on absence) ----
    if expression is None:
        return ToolResult(success=False, error=ToolError(type="validation_error", message="Parameter 'expression' (genes x samples matrix) is required."))
    if group_labels is None:
        return ToolResult(success=False, error=ToolError(type="validation_error", message="Parameter 'group_labels' (one label per sample) is required."))

    # ---- Normalize the test. ----
    test_key = str(test).strip().lower().replace("-", "_")
    if test_key in _WELCH_ALIASES:
        test_norm, effect_metric = "welch_t", "cohens_d"
    elif test_key in _MWU_ALIASES:
        test_norm, effect_metric = "mann_whitney", "rank_biserial"
    else:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"Unknown test {test!r}. Use 'welch_t' or 'mann_whitney'."))

    # ---- alpha / min_abs_log2fc / pseudocount sane? ----
    try:
        a = float(alpha)
    except (TypeError, ValueError):
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"alpha must be numeric in (0,1); got {alpha!r}."))
    if not np.isfinite(a) or not (0.0 < a < 1.0):
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"alpha must be in (0,1); got {a}."))
    try:
        min_fc = float(min_abs_log2fc)
        pc = float(pseudocount)
    except (TypeError, ValueError):
        return ToolResult(success=False, error=ToolError(type="validation_error", message="min_abs_log2fc and pseudocount must be numeric."))
    if not np.isfinite(min_fc) or min_fc < 0:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"min_abs_log2fc must be a non-negative number; got {min_fc}."))
    if not np.isfinite(pc) or pc < 0:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"pseudocount must be a non-negative number; got {pc}."))

    # ---- Parse the matrix. ----
    try:
        X = np.asarray(expression, dtype=float)
    except (TypeError, ValueError) as exc:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"expression must be a numeric 2-D matrix: {exc}"))
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.size == 0:
        return ToolResult(success=False, error=ToolError(type="validation_error", message="expression must be a non-empty 2-D matrix (genes x samples)."))
    if not np.all(np.isfinite(X)):
        return ToolResult(success=False, error=ToolError(type="validation_error", message="expression contains NaN or inf; every value must be a finite number."))

    n_genes, n_samples = X.shape

    labels = list(group_labels)
    if len(labels) != n_samples:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"group_labels length ({len(labels)}) must equal the number of samples/columns ({n_samples})."))

    labels_arr = np.array([str(v) for v in labels], dtype=object)
    unique = sorted(set(labels_arr.tolist()))
    if len(unique) != 2:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"Exactly two distinct groups are required; got {len(unique)}: {unique}."))

    # ---- Resolve case / control direction. ----
    if control_label is not None or case_label is not None:
        cl = None if control_label is None else str(control_label)
        ca = None if case_label is None else str(case_label)
        # Fill the unspecified side with the remaining label.
        if cl is not None and cl not in unique:
            return ToolResult(success=False, error=ToolError(type="validation_error", message=f"control_label {cl!r} is not one of the groups {unique}."))
        if ca is not None and ca not in unique:
            return ToolResult(success=False, error=ToolError(type="validation_error", message=f"case_label {ca!r} is not one of the groups {unique}."))
        if cl is None:
            cl = [u for u in unique if u != ca][0]
        if ca is None:
            ca = [u for u in unique if u != cl][0]
        if ca == cl:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="case_label and control_label must be different groups."))
        control_lbl, case_lbl = cl, ca
    else:
        control_lbl, case_lbl = unique[0], unique[1]

    case_mask = labels_arr == case_lbl
    control_mask = labels_arr == control_lbl
    n_case = int(case_mask.sum())
    n_control = int(control_mask.sum())
    if n_case < 2 or n_control < 2:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"Each group needs >= 2 samples for a variance-based test; got case={n_case}, control={n_control}."))

    # ---- Gene names. ----
    if gene_names is not None:
        names = [str(g) for g in gene_names]
        if len(names) != n_genes:
            return ToolResult(success=False, error=ToolError(type="validation_error", message=f"gene_names length ({len(names)}) must equal the number of genes/rows ({n_genes})."))
    else:
        names = [f"gene_{i}" for i in range(n_genes)]

    if not data_is_log2 and np.any(X < 0):
        return ToolResult(success=False, error=ToolError(type="validation_error", message="Negative values with data_is_log2=False: expression must be non-negative for a log2 ratio. Set data_is_log2=True if the matrix is already log-scaled."))

    # ---- Per-gene test. Degenerate genes are excluded (kept out of the FDR family). ----
    tested = []  # dicts without q yet
    excluded = []
    for gi in range(n_genes):
        row = X[gi]
        case_vals = row[case_mask]
        ctrl_vals = row[control_mask]
        mean_case = float(np.mean(case_vals))
        mean_ctrl = float(np.mean(ctrl_vals))

        # Zero-variance (constant across ALL samples) can never be differentially expressed.
        if np.ptp(row) == 0.0:
            excluded.append({"gene": names[gi], "reason": "zero_variance (constant across all samples)"})
            continue

        var_case = float(np.var(case_vals, ddof=1)) if n_case > 1 else 0.0
        var_ctrl = float(np.var(ctrl_vals, ddof=1)) if n_control > 1 else 0.0

        if test_norm == "welch_t":
            # No within-group variance in EITHER group -> the t statistic is undefined
            # (0/0 or division by 0). Excluding is the honest call, not a fake p=0.
            if var_case == 0.0 and var_ctrl == 0.0:
                excluded.append({"gene": names[gi], "reason": "no_within_group_variance (t undefined; perfect separation)"})
                continue
            res = stats.ttest_ind(case_vals, ctrl_vals, equal_var=False)
            statistic = float(res.statistic)
            p = float(res.pvalue)
            pooled_sd = np.sqrt(((n_case - 1) * var_case + (n_control - 1) * var_ctrl) / (n_case + n_control - 2))
            effect = float((mean_case - mean_ctrl) / pooled_sd) if pooled_sd > 0 else float("nan")
        else:  # mann_whitney
            res = stats.mannwhitneyu(case_vals, ctrl_vals, alternative="two-sided")
            statistic = float(res.statistic)
            p = float(res.pvalue)
            # Rank-biserial correlation in [-1, 1]; sign matches the case-vs-control direction.
            effect = float(2.0 * statistic / (n_case * n_control) - 1.0)

        if not np.isfinite(p):
            excluded.append({"gene": names[gi], "reason": "test_undefined (non-finite p-value)"})
            continue

        # log2 fold-change (case over control).
        if data_is_log2:
            log2fc = float(mean_case - mean_ctrl)
        else:
            log2fc = float(np.log2((mean_case + pc) / (mean_ctrl + pc)))

        tested.append({
            "gene": names[gi],
            "log2_fold_change": round(log2fc, 6),
            "effect_size": (round(effect, 6) if np.isfinite(effect) else None),
            "effect_size_metric": effect_metric,
            "statistic": round(statistic, 6),
            "p_value": p,
            "mean_case": round(mean_case, 6),
            "mean_control": round(mean_ctrl, 6),
            "_log2fc_abs": abs(log2fc),
            "_effect_finite": bool(np.isfinite(effect)),
        })

    n_tested = len(tested)
    results = []
    n_significant = 0
    if n_tested > 0:
        raw_p = np.array([t["p_value"] for t in tested], dtype=float)
        q = _bh_qvalues(raw_p)
        for t, qv in zip(tested, q):
            qv = float(qv)
            # HONESTY GATE: significant iff q < alpha AND a finite effect size is present.
            significant = bool(qv < a and t["_effect_finite"])
            small_effect = bool(significant and t["_log2fc_abs"] < min_fc)
            if significant:
                n_significant += 1
            results.append({
                "gene": t["gene"],
                "log2_fold_change": t["log2_fold_change"],
                "effect_size": t["effect_size"],
                "effect_size_metric": t["effect_size_metric"],
                "statistic": t["statistic"],
                "p_value": round(t["p_value"], 10),
                "q_value": round(qv, 10),
                "significant": significant,
                "small_effect": small_effect,
                "mean_case": t["mean_case"],
                "mean_control": t["mean_control"],
            })
        # Sort by q ascending, then raw p, then largest |log2FC|.
        results.sort(key=lambda r: (r["q_value"], r["p_value"], -abs(r["log2_fold_change"])))

    return ToolResult(
        success=True,
        output={
            "test": test_norm,
            "effect_size_metric": effect_metric,
            "alpha": a,
            "case_label": case_lbl,
            "control_label": control_lbl,
            "n_case": n_case,
            "n_control": n_control,
            "n_genes_total": int(n_genes),
            "n_genes_tested": int(n_tested),
            "n_genes_excluded": int(len(excluded)),
            "excluded": excluded,
            "results": results,
            "n_significant": int(n_significant),
        },
    )
