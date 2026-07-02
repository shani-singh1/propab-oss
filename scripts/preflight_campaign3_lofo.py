#!/usr/bin/env python3
"""Pre-flight multi-feature LOFO check before campaign 3 (fixes.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.domain_adapters.materials_adapter import run_materials_lofo  # noqa: E402

COMBOS: list[list[str]] = [
    ["mp_bandgap"],
    ["mp_bandgap", "mean_ionicity"],
    ["mp_bandgap", "mass_density"],
    ["mp_bandgap", "mean_ionicity", "mass_density"],
    ["mp_bandgap", "mean_electronegativity", "mean_ionicity"],
]


def main() -> int:
    results: list[dict] = []
    any_pass = False
    for features in COMBOS:
        r = run_materials_lofo(features=features)
        lofo = float(r["lofo_r2"])
        p95 = float(r["label_shuffle_null_p95"])
        passed = lofo > p95
        any_pass = any_pass or passed
        row = {
            "features": features,
            "lofo_r2": round(lofo, 6),
            "label_shuffle_null_p95": round(p95, 6),
            "beats_null": passed,
            "n_families": r.get("n_families"),
            "family_leakage_confirmed": r.get("family_leakage_confirmed"),
        }
        results.append(row)
        verdict = "PASS" if passed else "FAIL"
        print(f"{features}: lofo_r2={lofo:.3f} vs p95={p95:.3f} -> {verdict}")

    out = ROOT / "artifacts" / "campaign3_preflight_lofo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"combos": results, "any_beats_null": any_pass}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    print(f"any_beats_null: {any_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
