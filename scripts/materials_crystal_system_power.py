#!/usr/bin/env python3
"""LOFO power check on real crystal-system families (fixes.md Task A)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.domain_adapters.materials_adapter import MaterialsAdapter  # noqa: E402

# Reuse power block from evaluate_v1_domain_candidates
sys.path.insert(0, str(ROOT))
from scripts.evaluate_v1_domain_candidates import _lofo_power_block  # noqa: E402


def main() -> int:
    df = MaterialsAdapter().load_frame()
    counts = df["crystal_system"].value_counts().to_dict()
    features = [c for c in ("mp_bandgap", "mean_ionicity", "mass_density", "mean_atomic_mass") if c in df.columns]
    if not features:
        features = ["mean_ionicity", "mass_density", "mean_atomic_mass"]
    sub = df.dropna(subset=features + ["dielectric", "crystal_system"])
    X = sub[features].to_numpy(float)
    y = sub["dielectric"].to_numpy(float)
    groups = sub["crystal_system"].astype(str).to_numpy()
    lofo = _lofo_power_block(X, y, groups, n_perm=40)
    lofo["has_adequate_power"] = (
        lofo["smallest_group_n"] >= 50
        and lofo["n_groups"] >= 5
        and lofo["classical_mde_smallest_group"]["effects"].get("r2_25", {}).get("detectable_at_n", False)
    )
    report = {
        "n_samples": len(sub),
        "crystal_system_counts": counts,
        "n_crystal_systems": len(counts),
        "features_used": features,
        "has_mp_bandgap": bool("mp_bandgap" in df.columns and df["mp_bandgap"].notna().any()),
        "mp_bandgap_coverage": float(sub["mp_bandgap"].notna().mean()) if "mp_bandgap" in sub.columns else 0.0,
        "lofo_power": lofo,
    }
    report["lofo_power"]["has_adequate_power"] = bool(report["lofo_power"].get("has_adequate_power"))
    out = ROOT / "artifacts" / "materials_crystal_system_power.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if lofo["has_adequate_power"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
