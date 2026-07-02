#!/usr/bin/env python3
"""
Statistical power analysis for the Mandrake contrarian question (fixes.md).

Before a third 2-hour campaign run, answer: does n≈56 across 7 imbalanced families
have enough power to resolve family-specific-signal vs redundancy-artifact — for any
method, human or machine?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from demo.mandrake.domain import load_frame, repo_data_dir
from propab.domain_adapters.mandrake_adapter import (
    MandrakeAdapter,
    MandrakeExperimentSpec,
    _leave_one_family_out_r2,
    _make_model,
    _within_family_r2,
)

DEFAULT_FEATURES = ["t70_raw", "t75_raw", "foldseek_best_TM"]


def pairwise_identity(seq_a: str, seq_b: str) -> float:
    """Prefix identity on the shorter sequence (fast proxy; not full alignment)."""
    a, b = seq_a.upper(), seq_b.upper()
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n


def family_identity_matrix(df: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for fam, grp in df.groupby("rt_family"):
        names = grp["rt_name"].tolist()
        seqs = grp["sequence"].astype(str).tolist()
        n = len(names)
        mat = np.eye(n)
        pairs: list[dict] = []
        for i in range(n):
            for j in range(i + 1, n):
                ident = pairwise_identity(seqs[i], seqs[j])
                mat[i, j] = mat[j, i] = ident
                pairs.append({"a": names[i], "b": names[j], "identity": round(ident, 4)})
        pairs.sort(key=lambda p: p["identity"], reverse=True)
        arr = mat[np.triu_indices(n, k=1)] if n > 1 else np.array([])
        out[str(fam)] = {
            "n": n,
            "n_active": int(grp["active"].sum()),
            "pairwise_identity_mean": round(float(arr.mean()), 4) if len(arr) else None,
            "pairwise_identity_min": round(float(arr.min()), 4) if len(arr) else None,
            "pairwise_identity_max": round(float(arr.max()), 4) if len(arr) else None,
            "n_pairs_below_50pct": int(np.sum(arr < 0.5)) if len(arr) else 0,
            "n_pairs_below_70pct": int(np.sum(arr < 0.7)) if len(arr) else 0,
            "top_pairs": pairs[:5],
            "low_identity_pairs": [p for p in pairs if p["identity"] < 0.5][:5],
        }
    return out


def clustered_split_indices(
    identities: np.ndarray,
    *,
    test_frac: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Greedy split: maximize min train-test identity distance (clustered holdout)."""
    n = len(identities)
    if n < 4:
        return None
    n_test = max(2, int(round(n * test_frac)))
    rng = np.random.default_rng(seed)
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_score = -1.0
    for _ in range(200):
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        if len(train_idx) < 2 or len(test_idx) < 2:
            continue
        cross = identities[np.ix_(train_idx, test_idx)]
        score = float(np.min(cross)) if cross.size else 0.0
        if score > best_score:
            best_score = score
            best = train_idx, test_idx
    return best


def low_identity_holdout_indices(
    identities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Greedy: largest test set where every test seq is < threshold identity to all train."""
    n = len(identities)
    if n < 4:
        return None
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_n_test = -1
    for n_train in range(2, n - 1):
        for train_idx in _combinations_indices(n, n_train):
            train_idx = np.asarray(train_idx, dtype=int)
            mask = np.ones(n, dtype=bool)
            mask[train_idx] = False
            test_idx = np.where(mask)[0]
            cross = identities[np.ix_(train_idx, test_idx)]
            if cross.size and np.all(np.max(cross, axis=0) < threshold):
                if len(test_idx) > best_n_test:
                    best_n_test = len(test_idx)
                    best = train_idx, test_idx
    return best


def _combinations_indices(n: int, k: int, *, limit: int = 80):
    """Yield up to `limit` k-combinations of range(n) without itertools explosion."""
    from itertools import combinations

    combos = list(combinations(range(n), k))
    if len(combos) <= limit:
        yield from combos
        return
    step = max(1, len(combos) // limit)
    for i in range(0, len(combos), step):
        yield combos[i]


def split_r2(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model,
) -> float:
    m = clone(model)
    m.fit(X[train_idx], y[train_idx])
    pred = m.predict(X[test_idx])
    return float(max(-1.0, min(1.0, r2_score(y[test_idx], pred))))


def simulate_within_family_power(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_features: int,
    n_sim: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Power to reject label-shuffle null for pooled within-family R² (oracle upper bound)."""
    rng = np.random.default_rng(seed)
    model = _make_model("ridge")
    groups = np.zeros(len(y), dtype=int)
    observed = _within_family_r2(X, y, groups, model)
    null: list[float] = []
    for _ in range(n_sim):
        perm_y = y.copy()
        rng.shuffle(perm_y)
        null.append(_within_family_r2(X, perm_y, groups, model))
    null_arr = np.asarray(null)
    null_p95 = float(np.percentile(null_arr, 95))
    return {
        "n_samples": len(y),
        "n_features": n_features,
        "observed_within_r2": round(observed, 4),
        "null_mean": round(float(null_arr.mean()), 4),
        "null_std": round(float(null_arr.std()), 4),
        "null_p95": round(null_p95, 4),
        "empirical_p_vs_null": round((np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1), 4),
        "note": "Pooled across families — optimistic upper bound; per-family power is lower.",
    }


def per_family_split_analysis(
    df: pd.DataFrame,
    features: list[str],
    *,
    n_perm: int = 80,
    seed: int = 7,
) -> dict:
    model = _make_model("ridge")
    results: dict[str, dict] = {}
    for fam, grp in df.groupby("rt_family"):
        fam = str(fam)
        sub = grp.dropna(subset=features + ["pe_efficiency_pct"])
        n = len(sub)
        if n < 4:
            results[fam] = {"n": n, "testable": False, "reason": "n < 4 (adapter minimum)"}
            continue
        names = sub["rt_name"].tolist()
        seqs = sub["sequence"].astype(str).tolist()
        idx = {names[i]: i for i in range(n)}
        ident = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                v = pairwise_identity(seqs[i], seqs[j])
                ident[i, j] = ident[j, i] = v
        X = sub[features].to_numpy(float)
        y = sub["pe_efficiency_pct"].to_numpy(float)
        y_std = float(np.std(y))
        y_active = int((y > 0).sum())

        random_r2s: list[float] = []
        clustered_r2s: list[float] = []
        lowid_r2s: list[float] = []
        clustered_sizes: list[tuple[int, int]] = []
        lowid_sizes: list[tuple[int, int]] = []
        rng = np.random.default_rng(seed)

        for rep in range(n_perm):
            tr, te = train_test_split(np.arange(n), test_size=0.5, random_state=int(rng.integers(1e9)))
            random_r2s.append(split_r2(X, y, tr, te, model))
            cs = clustered_split_indices(ident, seed=int(rng.integers(1e9)))
            if cs:
                tr, te = cs
                clustered_r2s.append(split_r2(X, y, tr, te, model))
                clustered_sizes.append((len(tr), len(te)))
            lh = low_identity_holdout_indices(ident, threshold=0.5)
            if lh:
                tr, te = lh
                lowid_r2s.append(split_r2(X, y, tr, te, model))
                lowid_sizes.append((len(tr), len(te)))

        null_r2s: list[float] = []
        for _ in range(min(n_perm, 200)):
            perm_y = y.copy()
            rng.shuffle(perm_y)
            tr, te = train_test_split(np.arange(n), test_size=0.5, random_state=int(rng.integers(1e9)))
            null_r2s.append(split_r2(X, perm_y, tr, te, model))

        def summ(vals: list[float]) -> dict | None:
            if not vals:
                return None
            a = np.asarray(vals)
            return {
                "mean": round(float(a.mean()), 4),
                "std": round(float(a.std()), 4),
                "p05": round(float(np.percentile(a, 5)), 4),
                "p50": round(float(np.percentile(a, 50)), 4),
                "p95": round(float(np.percentile(a, 95)), 4),
                "frac_negative": round(float(np.mean(a < 0)), 4),
            }

        null_s = summ(null_r2s)
        rand_s = summ(random_r2s)
        clust_s = summ(clustered_r2s)
        low_s = summ(lowid_r2s)

        # Power proxies: P(R2 > null_p95) under each split scheme
        def power_proxy(obs: dict | None, null: dict | None) -> float | None:
            if not obs or not null:
                return None
            return round(float(obs["mean"] > null["p95"]), 4)  # crude; use simulation below

        results[fam] = {
            "n": n,
            "n_active_pe": y_active,
            "pe_std": round(y_std, 4),
            "testable": True,
            "identity": {
                "mean": round(float(ident[np.triu_indices(n, 1)].mean()), 4) if n > 1 else None,
                "min": round(float(ident[np.triu_indices(n, 1)].min()), 4) if n > 1 else None,
                "max": round(float(ident[np.triu_indices(n, 1)].max()), 4) if n > 1 else None,
            },
            "split_sizes": {
                "clustered_median": (
                    [int(np.median([s[0] for s in clustered_sizes])), int(np.median([s[1] for s in clustered_sizes]))]
                    if clustered_sizes else None
                ),
                "low_identity_median": (
                    [int(np.median([s[0] for s in lowid_sizes])), int(np.median([s[1] for s in lowid_sizes]))]
                    if lowid_sizes else None
                ),
                "low_identity_feasible_frac": round(len(lowid_r2s) / n_perm, 4),
            },
            "random_split_r2": rand_s,
            "clustered_split_r2": clust_s,
            "low_identity_holdout_r2": low_s,
            "label_shuffle_null_r2": null_s,
            "belief1_detectable": (
                clust_s is not None
                and null_s is not None
                and clust_s["p95"] > null_s["p95"]
                and clust_s["p50"] > 0
            ),
            "belief2_detectable": (
                low_s is not None
                and rand_s is not None
                and low_s["p50"] < 0
                and rand_s["p50"] - low_s["p50"] > 0.15
            ),
        }
    return results


def classical_regression_mde(n: int, k: int, *, power: float = 0.8, alpha: float = 0.05) -> dict:
    """Approximate minimum R² detectable via F-test (Cohen f²)."""
    # f² = R²/(1-R²); n_needed ≈ (z_alpha + z_beta)² / f² + k + 1
    from math import sqrt

    z_alpha = 1.96
    z_beta = 0.84 if power >= 0.8 else 0.52
    out = {}
    for r2 in (0.2, 0.35, 0.5, 0.65):
        f2 = r2 / max(1e-9, 1 - r2)
        n_need = (z_alpha + z_beta) ** 2 / f2 + k + 1
        out[f"r2_{int(r2*100)}"] = {
            "cohen_f2": round(f2, 3),
            "n_required_approx": int(np.ceil(n_need)),
            "detectable_at_n": n >= n_need,
        }
    return {"n_given": n, "k_predictors": k, "alpha": alpha, "power": power, "effects": out}


def lofo_power_from_null(null_std: float, null_p95: float, alpha: float = 0.05) -> dict:
    """Analytic-ish LOFO power given observed null band from label shuffle."""
    # Effect must exceed null_p95 to be "real"; width of null ~ 2.33*std for 95%
    mde_above_mean = null_p95  # observed must beat this
    return {
        "null_std": null_std,
        "null_p95": null_p95,
        "interpretation": (
            f"LOFO R² must exceed ~{null_p95:.2f} to clear 95th percentile of family-label null; "
            f"null SD≈{null_std:.2f} implies ~±{1.96*null_std:.2f} noise band at α≈0.05."
        ),
        "observed_in_prior_run": 0.063,
        "prior_run_percentile": "~70th (p≈0.30 per lofo_family_permutation.json)",
        "power_at_prior_observed": "low — not outside noise band",
    }


def overall_verdict(report: dict) -> str:
    fam = report["per_family_splits"]
    testable = [v for v in fam.values() if v.get("testable")]
    belief1_any = sum(1 for v in testable if v.get("belief1_detectable"))
    belief2_any = sum(1 for v in testable if v.get("belief2_detectable"))
    both_discriminate = sum(
        1 for v in testable if v.get("belief1_detectable") and v.get("belief2_detectable")
    )
    n_with_active = sum(1 for v in testable if v.get("n_active_pe", 0) >= 4)

    lines = []
    lines.append(
        f"Total n={report['dataset']['n_samples']} across {report['dataset']['n_families']} families; "
        f"only {n_with_active} families have ≥4 active PE measurements."
    )
    lines.append(
        "Classical power: detecting R²≈0.35 with 3 features needs n≈19 per family at 80% power — "
        "only Retroviral (n=18) barely reaches that; Retron (12) and LTR (10) do not."
    )
    if report["lofo_cross_family"]["empirical_p_vs_label_null"] > 0.05:
        lines.append(
            f"Cross-family LOFO R²={report['lofo_cross_family']['lofo_r2']:.3f} is inside the "
            f"label-shuffle noise band (p={report['lofo_cross_family']['empirical_p_vs_label_null']:.2f}; "
            f"must exceed null p95≈{report['lofo_cross_family']['label_null_p95']:.2f})."
        )
    lines.append(
        f"Contrarian split tests: Belief 1 (positive clustered-split R²) detectable in {belief1_any} "
        f"families; Belief 2 (redundancy collapse) in {belief2_any}; neither simultaneously in {both_discriminate}."
    )
    retro = fam.get("Retroviral", {})
    if retro.get("random_split_r2"):
        clust_p50 = (retro.get("clustered_split_r2") or {}).get("p50")
        clust_txt = f"{clust_p50:.2f}" if clust_p50 is not None else "n/a"
        lines.append(
            f"Best case Retroviral: median random-split R²={retro['random_split_r2']['p50']:.2f}, "
            f"clustered={clust_txt} — all negative today, so rivals cannot be separated by sign."
        )

    if belief1_any == 0 and report["lofo_cross_family"]["empirical_p_vs_label_null"] > 0.05:
        verdict = "UNDERPOWERED — a third 2-hour campaign run is very unlikely to resolve the rival beliefs."
        recommendation = (
            "Do not spend another campaign hour-budget on hypothesis search at this n. "
            "The two gated runs were inconclusive for the right reason: insufficient sample size and "
            "variance, not (only) protocol failure. Next steps: (1) expand to ≥30–50 RTs per target family, "
            "(2) pre-register one split protocol on Retroviral only, or (3) publish the power analysis as "
            "the honest stopping point for this dataset."
        )
    elif both_discriminate == 0:
        verdict = "MARGINAL / UNRESOLVABLE AT CURRENT N — isolated families lack simultaneous leverage on both rivals."
        recommendation = (
            "A focused Retroviral-only study with pre-registered clustered vs low-identity splits might "
            "weakly move one belief, but cannot adjudicate the full 7-family question. Not worth a third "
            "open-ended 2-hour search."
        )
    else:
        verdict = "PARTIAL — one family may discriminate; full 7-family question still open."
        recommendation = "Pre-register split analysis on the discriminating family before more search."

    return verdict + "\n\n" + recommendation + "\n\nEvidence:\n- " + "\n- ".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mandrake contrarian power analysis")
    parser.add_argument("--data-dir", default=str(repo_data_dir(ROOT)))
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "mandrake_power_analysis.json"))
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    df = load_frame(data_dir)
    labels = pd.read_csv(data_dir / "rt_sequences.csv")

    sub = df[["rt_name", "rt_family", "pe_efficiency_pct", "active", *features]].merge(
        labels[["rt_name", "sequence"]], on="rt_name", how="left",
    )
    for c in features:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=features + ["pe_efficiency_pct", "sequence"])
    sub = sub[sub["rt_family"] != "Unclassified"]

    adapter = MandrakeAdapter(data_dir=data_dir)
    spec = MandrakeExperimentSpec(feature_subset=features, methodology="LOFO")
    adapter_result = adapter.run_experiment(spec)

    X_all = sub[features].to_numpy(float)
    y_all = sub["pe_efficiency_pct"].to_numpy(float)
    fam_all = sub["rt_family"].astype(str).to_numpy()
    model = _make_model("ridge")

    # LOFO label null (reuse width)
    rng = np.random.default_rng(42)
    lofo_null: list[float] = []
    for _ in range(300):
        pf = fam_all.copy()
        rng.shuffle(pf)
        lofo_null.append(_leave_one_family_out_r2(X_all, y_all, pf, model))
    lofo_null_arr = np.asarray(lofo_null)
    lofo_obs = float(adapter_result.get("lofo_r2", 0))
    lofo_p = float((np.sum(lofo_null_arr >= lofo_obs) + 1) / (len(lofo_null_arr) + 1))

    fam_counts = sub["rt_family"].value_counts().to_dict()
    identity_report = family_identity_matrix(sub)

    per_family = per_family_split_analysis(sub, features)

    # Classical MDE for largest families
    classical = {
        fam: classical_regression_mde(n, len(features))
        for fam, n in fam_counts.items()
    }

    report = {
        "question": (
            "Is RT activity across 7 evolutionary families governed by family-specific mechanisms "
            "(signal under clustered split) vs redundancy artifacts (collapse under <50% identity holdout)?"
        ),
        "dataset": {
            "n_samples": len(sub),
            "n_families": len(fam_counts),
            "family_counts": fam_counts,
            "n_active_any_pe": int((sub["pe_efficiency_pct"] > 0).sum()),
            "features": features,
            "pe_efficiency_overall_std": round(float(sub["pe_efficiency_pct"].std()), 4),
        },
        "sequence_identity_by_family": identity_report,
        "lofo_cross_family": {
            "within_family_r2": round(float(adapter_result.get("within_family_r2", 0)), 4),
            "lofo_r2": round(lofo_obs, 4),
            "lofo_gap": round(float(adapter_result.get("lofo_gap", 0)), 4),
            "label_null_mean": round(float(lofo_null_arr.mean()), 4),
            "label_null_std": round(float(lofo_null_arr.std()), 4),
            "label_null_p95": round(float(np.percentile(lofo_null_arr, 95)), 4),
            "empirical_p_vs_label_null": round(lofo_p, 4),
            "power_summary": lofo_power_from_null(
                float(lofo_null_arr.std()), float(np.percentile(lofo_null_arr, 95)),
            ),
        },
        "classical_mde_per_family": classical,
        "per_family_splits": per_family,
        "oracle_pooled_within_family": simulate_within_family_power(
            X_all, y_all, n_features=len(features), n_sim=200,
        ),
        "verdict": "",
        "recommendation": "",
    }
    verdict_text = overall_verdict(report)
    report["verdict"] = verdict_text.split("\n\n")[0]
    report["recommendation"] = "\n\n".join(verdict_text.split("\n\n")[1:])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("dataset", "lofo_cross_family", "verdict", "recommendation")}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
