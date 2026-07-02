#!/usr/bin/env python3
"""
Family-label permutation null for LOFO R² (fixes.md sanity check).

Shuffle rt_family labels N times, recompute LOFO R² each time, compare to observed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from demo.mandrake.domain import load_frame, repo_data_dir
from propab.domain_adapters.mandrake_adapter import (
    MandrakeAdapter,
    MandrakeExperimentSpec,
    _leave_one_family_out_r2,
    _make_model,
)


def family_label_permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    model,
    *,
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[float, list[float]]:
    observed = _leave_one_family_out_r2(X, y, families, model)
    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(n_perm):
        perm_families = families.copy()
        rng.shuffle(perm_families)
        null.append(_leave_one_family_out_r2(X, y, perm_families, model))
    return observed, null


def summarize_null(observed: float, null: list[float]) -> dict:
    arr = np.asarray(null, dtype=float)
    # one-sided: how often null >= observed (LOFO at least this good by chance)
    p_ge = float(np.mean(arr >= observed))
    p_ge_adj = (np.sum(arr >= observed) + 1) / (len(arr) + 1)
    rank = int(np.sum(arr < observed)) + 1
    percentile = 100.0 * rank / len(arr)
    return {
        "observed_lofo_r2": round(observed, 6),
        "n_permutations": len(null),
        "null_mean": round(float(np.mean(arr)), 6),
        "null_std": round(float(np.std(arr)), 6),
        "null_min": round(float(np.min(arr)), 6),
        "null_max": round(float(np.max(arr)), 6),
        "null_p05": round(float(np.percentile(arr, 5)), 6),
        "null_p50": round(float(np.percentile(arr, 50)), 6),
        "null_p95": round(float(np.percentile(arr, 95)), 6),
        "null_p99": round(float(np.percentile(arr, 99)), 6),
        "empirical_p_value_ge": round(p_ge_adj, 4),
        "percentile_rank": round(percentile, 2),
        "outside_noise_band_p95": observed > float(np.percentile(arr, 95)),
        "outside_noise_band_p99": observed > float(np.percentile(arr, 99)),
        "verdict": (
            "signal plausibly real (observed above 95th percentile of label-shuffled null)"
            if observed > float(np.percentile(arr, 95)) and p_ge_adj < 0.05
            else "not clearly outside noise — LOFO may not exist at this n"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Family-label LOFO permutation test")
    parser.add_argument("--features", default="t70_raw,t75_raw,foldseek_best_TM")
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--data-dir", default=str(repo_data_dir(ROOT)))
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "lofo_family_permutation.json"))
    args = parser.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    adapter = MandrakeAdapter(data_dir=Path(args.data_dir))
    spec = MandrakeExperimentSpec(feature_subset=features, methodology="LOFO", baseline_model=args.model)
    result = adapter.run_experiment(spec)

    df = load_frame(Path(args.data_dir))
    cols = [c for c in features if c in df.columns]
    sub = df[["pe_efficiency_pct", "rt_family", *cols]].dropna()
    X = sub[cols].to_numpy(dtype=float)
    y = sub["pe_efficiency_pct"].to_numpy(dtype=float)
    families = sub["rt_family"].astype(str).to_numpy()
    model = _make_model(args.model)

    observed, null = family_label_permutation_null(
        X, y, families, model, n_perm=args.n_perm,
    )
    summary = summarize_null(observed, null)
    report = {
        "features": cols,
        "model": args.model,
        "n_samples": len(y),
        "n_families": len(np.unique(families)),
        "families": sorted(set(families.tolist())),
        "adapter_lofo_r2": result.get("mean_r2"),
        "adapter_permutation_p_y_within_family": result.get("permutation_p"),
        **summary,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
