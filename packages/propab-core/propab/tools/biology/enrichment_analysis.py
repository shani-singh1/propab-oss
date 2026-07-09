"""B3 — over-representation (enrichment) analysis with an EXPLICIT background + FDR.

Given a query gene list, an explicit background/universe, and caller-supplied gene-set
definitions (no network), this computes for each set a hypergeometric over-representation
p-value, a Benjamini-Hochberg FDR q-value across all tested sets, and the fold-enrichment
(observed overlap vs. the overlap expected by chance). Sets are returned sorted by q.

Honesty by construction:
  * **The background must be explicit.** Enrichment is meaningless without the universe
    it is measured against; the tool REJECTS a query that is not a subset of the
    background (a hidden/implied universe is how enrichment gets faked).
  * **FDR across all tested sets.** Many sets are tested at once, so significance is the
    BH q-value, never the raw hypergeometric p. ``n_significant`` counts q < alpha only.
  * **Fold-enrichment is always reported** so a "significant" result resting on a tiny
    absolute overlap (e.g. 2 genes) is visible; a ``small_overlap`` flag makes it louder.
  * **Degenerate inputs fail loudly.** Empty query, empty background, empty gene_sets,
    or a background smaller than the query -> validation_error, never a fabricated q.
    Sets with no members in the background are skipped with a reason (kept out of the
    FDR family), not silently scored.
  * Gene lists are de-duplicated (a gene counted twice cannot inflate an overlap).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "enrichment_analysis",
    "domain": "biology",
    "audience": "worker",
    # Emits per-set significance calls (hypergeometric p / q): surfaced to significance
    # workflows AND never auto-filled from the spec example (that would inject a
    # placeholder query/background and manufacture fake enrichment).
    "significance_capable": True,
    "description": (
        "Over-representation (enrichment) analysis of a query gene list against an "
        "EXPLICIT background, over caller-supplied gene sets (dict set_name -> member "
        "genes; no network). Per set: hypergeometric over-representation p, "
        "Benjamini-Hochberg FDR q across all tested sets, and fold-enrichment; sorted "
        "by q. The query MUST be a subset of the background or it is rejected (a hidden "
        "universe is how enrichment gets faked). Significance is q<alpha, never raw p; "
        "fold-enrichment and a small_overlap flag keep a tiny-overlap 'hit' visible. "
        "Empty query/background/gene_sets or background<query -> validation_error."
    ),
    "params": {
        "query_genes": {
            "type": "list[str]",
            "required": True,
            "description": "Genes of interest (e.g. the DE hits). Must be a subset of background_genes; de-duplicated.",
        },
        "background_genes": {
            "type": "list[str]",
            "required": True,
            "description": "The universe of genes the query was drawn from (e.g. all measured genes). Must contain every query gene.",
        },
        "gene_sets": {
            "type": "dict[str, list[str]]",
            "required": True,
            "description": "Mapping set_name -> member genes (e.g. GO terms / pathways). Provided by the caller; no network fetch.",
        },
        "alpha": {
            "type": "float",
            "required": False,
            "default": 0.05,
            "description": "FDR threshold in (0,1). A set is significant iff BH q < alpha (never raw p).",
        },
        "min_set_size": {
            "type": "int",
            "required": False,
            "default": 1,
            "description": "Gene sets with fewer than this many members present in the background are skipped (kept out of the FDR family).",
        },
        "min_overlap": {
            "type": "int",
            "required": False,
            "default": 3,
            "description": "A significant set whose query overlap is below this is flagged small_overlap (visibility only).",
        },
    },
    "output": {
        "alpha": "float — FDR level applied",
        "background_size": "int — |background| (universe size N)",
        "query_size": "int — |query ∩ background| (draw size n)",
        "n_sets_total": "int — gene sets supplied",
        "n_sets_tested": "int — sets that entered the FDR family",
        "n_sets_skipped": "int — sets with no/too-few members in the background",
        "skipped": "list — [{set_name, reason}] for untested sets",
        "results": "list — per tested set {set_name, overlap, set_size_in_background, expected_overlap, fold_enrichment, p_value, q_value, significant, small_overlap, overlap_genes}, sorted by q",
        "n_significant": "int — sets with q < alpha",
    },
    "example": {
        "params": {
            "query_genes": ["A", "B", "C", "D"],
            "background_genes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "gene_sets": {"pathway1": ["A", "B", "C", "E"], "pathway2": ["G", "H"]},
            "alpha": 0.05,
        },
        "output": {
            "n_sets_tested": 2,
            "results": [{"set_name": "pathway1", "overlap": 3, "fold_enrichment": 1.875}],
        },
    },
}


def _bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (same order out as in)."""
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


def enrichment_analysis(
    query_genes: list | None = None,
    background_genes: list | None = None,
    gene_sets: dict | None = None,
    alpha: float = 0.05,
    min_set_size: int = 1,
    min_overlap: int = 3,
) -> ToolResult:
    # ---- Required inputs present? ----
    if query_genes is None:
        return ToolResult(success=False, error=ToolError(type="validation_error", message="Parameter 'query_genes' is required."))
    if background_genes is None:
        return ToolResult(success=False, error=ToolError(type="validation_error", message="Parameter 'background_genes' is required — the background/universe must be explicit."))
    if gene_sets is None:
        return ToolResult(success=False, error=ToolError(type="validation_error", message="Parameter 'gene_sets' (dict set_name -> member genes) is required."))
    if not isinstance(gene_sets, dict):
        return ToolResult(success=False, error=ToolError(type="validation_error", message="gene_sets must be a dict mapping set_name -> list of member genes."))

    # ---- alpha / thresholds sane? ----
    try:
        a = float(alpha)
    except (TypeError, ValueError):
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"alpha must be numeric in (0,1); got {alpha!r}."))
    if not np.isfinite(a) or not (0.0 < a < 1.0):
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"alpha must be in (0,1); got {a}."))
    try:
        min_k_set = int(min_set_size)
        min_ov = int(min_overlap)
    except (TypeError, ValueError):
        return ToolResult(success=False, error=ToolError(type="validation_error", message="min_set_size and min_overlap must be integers."))
    if min_k_set < 1:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"min_set_size must be >= 1; got {min_k_set}."))

    # ---- De-duplicate to sets (a gene counted twice cannot inflate an overlap). ----
    bg = {str(g) for g in background_genes}
    query = {str(g) for g in query_genes}

    if len(bg) == 0:
        return ToolResult(success=False, error=ToolError(type="validation_error", message="background_genes is empty — a universe with no genes cannot support enrichment."))
    if len(query) == 0:
        return ToolResult(success=False, error=ToolError(type="validation_error", message="query_genes is empty — nothing to test for enrichment."))
    if len(gene_sets) == 0:
        return ToolResult(success=False, error=ToolError(type="validation_error", message="gene_sets is empty — no sets to test."))

    # ---- The background MUST be explicit and complete: query ⊆ background. ----
    missing = query - bg
    if missing:
        sample = sorted(missing)[:10]
        return ToolResult(
            success=False,
            error=ToolError(
                type="validation_error",
                message=(
                    f"{len(missing)} query gene(s) are not in the background/universe "
                    f"(e.g. {sample}). Enrichment requires the query to be a subset of an "
                    "explicit background; add these genes to background_genes or remove them."
                ),
            ),
        )

    N = len(bg)          # universe size
    n = len(query)       # number of "successes drawn" = query size (all in background)

    # ---- Per-set hypergeometric over-representation. ----
    tested = []
    skipped = []
    for set_name, members in gene_sets.items():
        name = str(set_name)
        if members is None:
            skipped.append({"set_name": name, "reason": "no members"})
            continue
        try:
            members_set = {str(g) for g in members}
        except TypeError:
            skipped.append({"set_name": name, "reason": "members is not a list of genes"})
            continue
        # Restrict the set to the background (only genes that could have been drawn).
        set_in_bg = members_set & bg
        K = len(set_in_bg)  # successes available in the universe
        if K == 0:
            skipped.append({"set_name": name, "reason": "no members present in the background"})
            continue
        if K < min_k_set:
            skipped.append({"set_name": name, "reason": f"set_size_in_background {K} < min_set_size {min_k_set}"})
            continue

        overlap_genes = query & set_in_bg
        k = len(overlap_genes)  # observed successes in the draw
        # Over-representation upper tail: P(X >= k) = sf(k-1).
        p = float(stats.hypergeom.sf(k - 1, N, K, n))
        p = min(max(p, 0.0), 1.0)
        expected = float(n * K / N)
        fold = float((k / expected)) if expected > 0 else float("inf")

        tested.append({
            "set_name": name,
            "overlap": int(k),
            "set_size_in_background": int(K),
            "expected_overlap": round(expected, 6),
            "fold_enrichment": (round(fold, 6) if np.isfinite(fold) else None),
            "p_value": p,
            "overlap_genes": sorted(overlap_genes),
        })

    results = []
    n_significant = 0
    if tested:
        raw_p = np.array([t["p_value"] for t in tested], dtype=float)
        q = _bh_qvalues(raw_p)
        for t, qv in zip(tested, q):
            qv = float(qv)
            significant = bool(qv < a)  # decided on the FDR q, never the raw p
            small_overlap = bool(significant and t["overlap"] < min_ov)
            if significant:
                n_significant += 1
            results.append({
                "set_name": t["set_name"],
                "overlap": t["overlap"],
                "set_size_in_background": t["set_size_in_background"],
                "expected_overlap": t["expected_overlap"],
                "fold_enrichment": t["fold_enrichment"],
                "p_value": round(t["p_value"], 10),
                "q_value": round(qv, 10),
                "significant": significant,
                "small_overlap": small_overlap,
                "overlap_genes": t["overlap_genes"],
            })
        # Sort by q ascending, then raw p, then largest fold-enrichment.
        results.sort(key=lambda r: (
            r["q_value"],
            r["p_value"],
            -(r["fold_enrichment"] if r["fold_enrichment"] is not None else float("inf")),
        ))

    return ToolResult(
        success=True,
        output={
            "alpha": a,
            "background_size": int(N),
            "query_size": int(n),
            "n_sets_total": int(len(gene_sets)),
            "n_sets_tested": int(len(tested)),
            "n_sets_skipped": int(len(skipped)),
            "skipped": skipped,
            "results": results,
            "n_significant": int(n_significant),
        },
    )
