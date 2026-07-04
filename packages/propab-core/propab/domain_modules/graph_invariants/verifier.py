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

    verified = holds_train and holds_held
    return {
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


def classify_graph_verdict(hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
    steps = int(result.get("verified_true_steps") or 0)
    held = float(result.get("held_out_correlation") or 0.0)
    if steps >= 1:
        return "confirmed", f"invariant holds on held-out family (r={held:.3f})", 0.90
    if abs(held) < 0.05:
        return "refuted", f"no invariant relationship on holdout (r={held:.3f})", 0.85
    return "inconclusive", f"weak cross-family invariant signal (r={held:.3f})", 0.55
