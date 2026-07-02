#!/usr/bin/env python3
"""
fixes.md one-hour test: falsify Propab's best-scoring mechanism via counterfactual experiment.

Mechanism (competing_mechanisms id 2d1828a5):
  t55/t70/t75 correlate with RT activity only because they proxy family-specific chemistry;
  LOFO should degrade with finer family splits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from demo.mandrake.domain import load_frame
from propab.domain_adapters.mandrake_adapter import (
    _leave_one_family_out_r2,
    _make_model,
    _within_family_r2,
)

ART = ROOT / "artifacts"
FEATURES = ["t55_raw", "t70_raw", "t75_raw"]
MECHANISM_ID = "2d1828a5-e415-42d0-abab-8946a49f4b78"


def _lofo_groups(X, y, groups, model):
    return _leave_one_family_out_r2(X, y, groups, model)


def _fine_groups(df: pd.DataFrame, scheme: str) -> np.ndarray:
    if scheme == "family_x_active":
        return (df["rt_family"].astype(str) + "|active=" + df["active"].astype(str)).to_numpy()
    if scheme == "family_x_thermal_quartile":
        q = pd.qcut(df["t70_raw"], q=4, duplicates="drop")
        return (df["rt_family"].astype(str) + "|t70_q=" + q.astype(str)).to_numpy()
    if scheme == "family_x_foldseek_cluster":
        tm = pd.to_numeric(df["foldseek_best_TM"], errors="coerce").fillna(0)
        q = pd.qcut(tm, q=3, duplicates="drop")
        return (df["rt_family"].astype(str) + "|tm_q=" + q.astype(str)).to_numpy()
    if scheme == "per_sequence":
        return df["rt_name"].astype(str).to_numpy()
    raise ValueError(scheme)


def _residualized_r2(X, y, family_dummies: np.ndarray, model) -> float:
    """R² of predicting y from X after removing family-mean effects."""
    y_res = y.copy()
    X_res = X.copy()
    for i in range(family_dummies.shape[1]):
        mask = family_dummies[:, i] > 0.5
        if mask.sum() < 2:
            continue
        y_res[mask] -= y[mask].mean()
        X_res[mask] -= X[mask].mean(axis=0)
    m = clone(model)
    m.fit(X_res, y_res)
    return float(r2_score(y_res, m.predict(X_res)))


def main() -> int:
    df = load_frame()
    cols = [c for c in FEATURES if c in df.columns]
    sub = df[["pe_efficiency_pct", "rt_family", "active", "rt_name", "foldseek_best_TM", *cols]].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    X = sub[cols].to_numpy(float)
    y = sub["pe_efficiency_pct"].to_numpy(float)
    families = sub["rt_family"].astype(str).to_numpy()
    model = _make_model("ridge")

    coarse_lofo = _lofo_groups(X, y, families, model)
    coarse_within = _within_family_r2(X, y, families, model)
    coarse_gap = coarse_within - coarse_lofo

    fine_schemes = ["family_x_active", "family_x_thermal_quartile", "family_x_foldseek_cluster"]
    fine_results = {}
    for scheme in fine_schemes:
        g = _fine_groups(sub, scheme)
        n_groups = len(np.unique(g))
        lofo = _lofo_groups(X, y, g, model)
        within = _within_family_r2(X, y, g, model)
        fine_results[scheme] = {
            "n_groups": n_groups,
            "lofo_r2": round(lofo, 4),
            "within_r2": round(within, 4),
            "gap": round(within - lofo, 4),
            "delta_lofo_vs_coarse": round(lofo - coarse_lofo, 4),
        }

    # Family dummies for partial test
    fam_dummies = pd.get_dummies(sub["rt_family"], drop_first=False).to_numpy(float)
    residual_r2 = _residualized_r2(X, y, fam_dummies, model)

    # Counterfactual prediction from mechanism:
    # If proxy-only, finer splits → LOFO degrades (more negative)
    degrades = all(r["delta_lofo_vs_coarse"] <= 0 for r in fine_results.values())
    any_improves = any(r["delta_lofo_vs_coarse"] > 0.05 for r in fine_results.values())

    if degrades and not any_improves:
        outcome = "supports_proxy_mechanism"
        interpretation = (
            "Finer group splits did not improve LOFO; LOFO same or worse. "
            "Consistent with thermal features tracking group identity, not cross-group biology."
        )
    elif any_improves:
        outcome = "refutes_proxy_mechanism"
        interpretation = (
            "LOFO improved under at least one finer split — thermal signal is not purely "
            "coarse-family proxy; mechanism's counterfactual failed."
        )
    else:
        outcome = "inconclusive"
        interpretation = "Mixed movement in LOFO under fine splits; proxy claim neither clearly confirmed nor refuted."

    report = {
        "fixes_md_test": "one-hour counterfactual on best Propab mechanism",
        "mechanism_id": MECHANISM_ID,
        "mechanism_as_stated": (
            "Features ['t55_raw', 't70_raw', 't75_raw'] correlate with RT activity only because "
            "they proxy unmeasured family-specific chemistry — LOFO should degrade with finer family splits."
        ),
        "source": "propab_competing (structure bench top score 7/20)",
        "counterfactual_constructed": {
            "intervention": "Split grouping finer than rt_family (family×active, family×thermal quartile, family×foldseek TM quartile)",
            "prediction_if_mechanism_true": (
                "LOFO R² should degrade (become more negative) vs coarse rt_family LOFO, "
                "because thermal features increasingly encode subgroup identity rather than transferable biology."
            ),
            "prediction_if_mechanism_false": (
                "Cross-subgroup signal should persist or improve under splits that remove coarse family confounding."
            ),
        },
        "experiment": {
            "n_samples": len(sub),
            "n_families_coarse": len(np.unique(families)),
            "features": cols,
            "coarse": {
                "lofo_r2": round(coarse_lofo, 4),
                "within_r2": round(coarse_within, 4),
                "gap": round(coarse_gap, 4),
            },
            "fine_splits": fine_results,
            "residual_r2_after_family_demean": round(residual_r2, 4),
        },
        "outcome": outcome,
        "interpretation": interpretation,
        "fixes_md_verdict": "",
    }

    if outcome == "supports_proxy_mechanism":
        report["fixes_md_verdict"] = (
            "Mechanism was specific enough to test and the counterfactual held: this is a "
            "family-proxy claim, not cross-family biophysics. Wrong-but-falsifiable / "
            "right-but-shallow — either way, not a real transferable mechanism. "
            "Next problem is upstream generation, not schema v2 alone."
        )
    elif outcome == "refutes_proxy_mechanism":
        report["fixes_md_verdict"] = (
            "Counterfactual failed — mechanism was wrong but checkable. Progress: "
            "first falsifiable Propab mechanism candidate; worth building infra around residual tests."
        )
    else:
        report["fixes_md_verdict"] = (
            "Could construct counterfactual but outcome ambiguous. Mechanism is at least "
            "partially falsifiable; not pure comparative-regression dressed as causal language."
        )

    out_path = ART / "mechanism_counterfactual_test.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "experiment"}, indent=2))
    print("\nexperiment:", json.dumps(report["experiment"], indent=2))
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
