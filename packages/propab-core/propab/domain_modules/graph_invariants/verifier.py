"""Deterministic graph invariant verification across network families."""
from __future__ import annotations

from typing import Any

import numpy as np

from propab.domain_modules.graph_invariants.adapter import GraphInvariantSpec, GraphInvariantsAdapter


def _family_correlation(df, family: str, src: str, tgt: str) -> float:
    sub = df[df["network_family"] == family]
    if len(sub) < 5:
        return 0.0
    x = sub[src].to_numpy(dtype=float)
    y = sub[tgt].to_numpy(dtype=float)
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _label_shuffle_null(
    df, held: str, src: str, tgt: str, observed_abs: float, *, n_perm: int = 200, seed: int = 0
) -> tuple[float, float] | None:
    """Adversarial null for the held-out src->tgt correlation.

    Shuffles the target invariant against the source WITHIN the held-out family
    (breaking any real src->tgt relationship while preserving each column's own
    distribution), recomputes |corr| n_perm times, and returns
    (null_p95, permutation_p). Returns None when a null cannot be built (held
    family too small or degenerate) — the caller then emits NO null stats, so the
    result fails closed to "unknown"/inconclusive rather than a free confirm.
    """
    sub = df[df["network_family"] == held]
    if len(sub) < 5:
        return None
    x = sub[src].to_numpy(dtype=float)
    y = sub[tgt].to_numpy(dtype=float)
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    rng = np.random.default_rng(seed)
    null_abs = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        yp = rng.permutation(y)
        null_abs[i] = 0.0 if np.std(yp) < 1e-9 else abs(float(np.corrcoef(x, yp)[0, 1]))
    p95 = float(np.percentile(null_abs, 95))
    perm_p = float(np.mean(null_abs >= observed_abs))
    return p95, perm_p


def run_graph_invariant_check(spec: GraphInvariantSpec) -> dict[str, Any]:
    df = GraphInvariantsAdapter().load_frame()
    src, tgt = spec.source_invariant, spec.target_invariant
    if src not in df.columns or tgt not in df.columns:
        raise ValueError(f"Unknown invariants: {src}, {tgt}")

    by_family = {fam: _family_correlation(df, fam, src, tgt) for fam in df["network_family"].unique()}
    held = spec.held_out_family or sorted(by_family.keys())[0]
    train_fams = [f for f in by_family if f != held]
    train_corr = float(np.mean([by_family[f] for f in train_fams])) if train_fams else 0.0
    held_corr = by_family.get(held, 0.0)

    if spec.claim_type == "correlation_negative":
        holds_train = train_corr < -0.1
        holds_held = held_corr < -0.05
    elif spec.claim_type == "holds_all_families":
        holds_train = all(abs(c) > 0.15 for c in by_family.values())
        holds_held = abs(held_corr) > 0.1
    else:
        holds_train = train_corr > 0.15
        holds_held = held_corr > 0.05

    # Real adversarial null (replaces the old bare-threshold confirm that, via
    # verification_method="cross_network_lofo", was classified "deterministic" and
    # bypassed the artifact gate entirely). We emit the label-shuffle null in the
    # lofo shape the gate reads (lofo_r2 / label_shuffle_null_p95 /
    # label_shuffle_permutation_p) so classify_evidence_type routes this as "lofo".
    #
    # The null must also GATE this domain's own verdict counters — not just the
    # downstream artifact gate. Previously ``verified`` was the bare
    # train/held threshold alone, so a spurious correlation that clears the
    # threshold but SITS INSIDE the shuffle-null distribution (perm_p not
    # significant) still emitted verified_true_steps=1 / discovery_worthy=True,
    # and classify_graph_verdict returned "confirmed" at 0.90 — a fail-open that
    # every sibling statistical domain (genomics/enzyme/network_diffusion) closes
    # by requiring the null. When the null cannot be built we fail CLOSED
    # (verified=False), consistent with this function's "no free confirm" contract.
    null = _label_shuffle_null(df, held, src, tgt, abs(held_corr))
    survives_null = False
    if null is not None:
        _p95, _perm_p = null
        survives_null = abs(held_corr) > _p95 and _perm_p < 0.05

    verified = holds_train and holds_held and survives_null

    result: dict[str, Any] = {
        "metric_name": "invariant_correlation",
        "metric_value": held_corr,
        "train_correlation": train_corr,
        "held_out_correlation": held_corr,
        "held_out_family": held,
        "by_family": by_family,
        "verification_method": "cross_network_lofo",
        "verified_true_steps": 1 if verified else 0,
        "verified_false_steps": 0 if verified else 1,
        "discovery_worthy": verified,
        "trivial_rediscovery": not verified,
    }
    if null is not None:
        p95, perm_p = null
        result["lofo_r2"] = abs(held_corr)
        result["label_shuffle_null_p95"] = p95
        result["label_shuffle_permutation_p"] = perm_p
    return result


def classify_graph_verdict(hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
    steps = int(result.get("verified_true_steps") or 0)
    held = float(result.get("held_out_correlation") or 0.0)
    if steps >= 1:
        return "confirmed", f"invariant holds on held-out family (r={held:.3f})", 0.90
    if abs(held) < 0.05:
        return "refuted", f"no invariant relationship on holdout (r={held:.3f})", 0.85
    return "inconclusive", f"weak cross-family invariant signal (r={held:.3f})", 0.55
