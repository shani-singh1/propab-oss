#!/usr/bin/env python3
"""
Preflight LOFO on matbench perovskites (fixes.md Option 2).

A-site element group as family; formation energy (e_form) as target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.domain_adapters.perovskites_adapter import (  # noqa: E402
    PerovskitesAdapter,
    run_perovskites_lofo,
)

COMBOS: list[list[str]] = [
    ["mean_ionicity", "mean_electronegativity"],
    ["mean_atomic_mass", "mass_density", "mean_ionicity"],
    ["n_elements", "mean_Z", "mean_ionicity", "mass_density"],
    ["mean_electronegativity", "mean_ionicity", "std_Z"],
    ["mean_coordination", "mass_density", "mean_ionicity"],
    ["mean_atomic_mass", "mean_ionicity", "mean_electronegativity", "std_principal_quantum_n"],
]


def main() -> int:
    adapter = PerovskitesAdapter()
    print("Loading/featurizing matbench perovskites (cached after first run)...", flush=True)
    df = adapter.load_frame()
    counts = df["a_site_group"].value_counts()
    print(f"perovskites: n={len(df)} a_site_families={len(counts)} min_family_n={counts.min()}", flush=True)
    print(f"top families: {counts.head(8).to_dict()}\n", flush=True)

    results: list[dict] = []
    any_pass = False
    best = None
    for i, features in enumerate(COMBOS, 1):
        print(f"[{i}/{len(COMBOS)}] {features} ...", flush=True)
        r = run_perovskites_lofo(features=features, df=df, preflight=True)
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
            "n_families": r.get("n_families"),
            "n_samples": r.get("n_samples"),
            "family_leakage_confirmed": r.get("family_leakage_confirmed"),
        }
        results.append(row)
        if best is None or lofo > best["lofo_r2"]:
            best = row
        verdict = "PASS" if passed else "FAIL"
        print(f"{features}: lofo_r2={lofo:.3f} vs p95={p95:.3f} -> {verdict}", flush=True)

    out = ROOT / "artifacts" / "perovskites_preflight_lofo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "matbench_perovskites",
        "target": "e_form",
        "family": "a_site_group",
        "n_samples": len(df),
        "n_a_site_families": int(len(counts)),
        "family_counts_top10": counts.head(10).to_dict(),
        "combos": results,
        "any_beats_null": any_pass,
        "best_combo": best,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"any_beats_null: {any_pass}")
    if best:
        print(f"best: {best['features']} lofo={best['lofo_r2']} p95={best['label_shuffle_null_p95']}")
    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
