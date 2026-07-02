#!/usr/bin/env python3
"""
Option 1 preflight (fixes.md): within-cubic dielectric LOFO.

Restricts to cubic crystal system (n≈599) and LOFO on composition-complexity
families (n_elements) — avoids cross-crystal-system tensor mismatch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.domain_adapters.mandrake_adapter import (  # noqa: E402
    _family_label_shuffle_null,
    _leave_one_family_out_r2,
    _make_model,
)
from propab.domain_adapters.materials_adapter import MaterialsAdapter  # noqa: E402

COMBOS: list[list[str]] = [
    ["mp_bandgap"],
    ["mp_bandgap", "mean_ionicity"],
    ["mp_bandgap", "mass_density"],
    ["mp_bandgap", "mean_ionicity", "mass_density"],
    ["mp_bandgap", "mean_electronegativity", "mean_ionicity"],
]

TARGET = "dielectric"
FAMILY = "n_elements"


def _run_lofo(df: pd.DataFrame, features: list[str]) -> dict:
    cols = [c for c in features if c in df.columns]
    sub = df[[TARGET, FAMILY, *cols]].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub[FAMILY] = sub[FAMILY].astype(str)
    sub = sub.dropna()
    X = sub[cols].to_numpy(dtype=float)
    y = sub[TARGET].to_numpy(dtype=float)
    families = sub[FAMILY].to_numpy()
    model = _make_model("ridge")
    lofo = _leave_one_family_out_r2(X, y, families, model)
    _, label_p, label_null = _family_label_shuffle_null(X, y, families, model, n_perm=100)
    p95 = float(np.percentile(label_null, 95)) if label_null else 0.0
    return {
        "lofo_r2": lofo,
        "label_shuffle_null_p95": p95,
        "n_families": int(len(np.unique(families))),
        "n_samples": len(y),
        "family_counts": sub[FAMILY].value_counts().to_dict(),
    }


def main() -> int:
    df = MaterialsAdapter().load_frame()
    cubic = df[df["crystal_system"] == "cubic"].copy()
    counts = cubic[FAMILY].value_counts()
    print(f"cubic dielectric: n={len(cubic)} families={len(counts)} ({FAMILY})", flush=True)
    print(f"family counts: {counts.to_dict()}\n", flush=True)

    results: list[dict] = []
    any_pass = False
    best = None
    for i, features in enumerate(COMBOS, 1):
        print(f"[{i}/{len(COMBOS)}] {features} ...", flush=True)
        r = _run_lofo(cubic, features)
        lofo = float(r["lofo_r2"])
        p95 = float(r["label_shuffle_null_p95"])
        passed = lofo > p95
        any_pass = any_pass or passed
        row = {
            "features": features,
            "lofo_r2": round(lofo, 6),
            "label_shuffle_null_p95": round(p95, 6),
            "gap_to_null": round(lofo - p95, 6),
            "beats_null": passed,
            **{k: r[k] for k in ("n_families", "n_samples", "family_counts")},
        }
        results.append(row)
        if best is None or lofo > best["lofo_r2"]:
            best = row
        verdict = "PASS" if passed else "FAIL"
        print(f"{features}: lofo_r2={lofo:.3f} vs p95={p95:.3f} -> {verdict}", flush=True)

    out = ROOT / "artifacts" / "cubic_dielectric_preflight_lofo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "matbench_dielectric",
        "subset": "cubic",
        "target": TARGET,
        "family": FAMILY,
        "n_samples": len(cubic),
        "combos": results,
        "any_beats_null": any_pass,
        "best_combo": best,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)
    print(f"any_beats_null: {any_pass}", flush=True)
    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
