#!/usr/bin/env python3
"""2-minute smoke check: materials LOFO on real matbench dielectric data."""
from __future__ import annotations

import math
import sys

from propab.domain_adapters.materials_adapter import (
    MaterialsAdapter,
    MaterialsExperimentSpec,
    resolve_materials_features,
)
from propab.tools.materials.materials_verification import materials_verification


def _check_result(result: dict, *, label: str) -> bool:
    lofo = result.get("lofo_r2")
    p95 = result.get("label_shuffle_null_p95")
    leakage = result.get("family_leakage_confirmed")

    print(f"\n=== {label} ===")
    print(f"  lofo_r2:                  {lofo}")
    print(f"  label_shuffle_null_p95:   {p95}")
    print(f"  family_leakage_confirmed: {leakage}")
    print(f"  lofo_gap:                 {result.get('lofo_gap')}")
    print(f"  n_samples:                {result.get('n_samples')}")
    print(f"  n_families:               {result.get('n_families')}")
    print(f"  feature_subset:           {result.get('feature_subset')}")

    ok = True
    if lofo is None or (isinstance(lofo, float) and (math.isnan(lofo) or math.isinf(lofo))):
        print("  FAIL: lofo_r2 missing or non-finite")
        ok = False
    if p95 is None or (isinstance(p95, float) and (math.isnan(p95) or math.isinf(p95))):
        print("  FAIL: label_shuffle_null_p95 missing or non-finite")
        ok = False
    if leakage is None or not isinstance(leakage, bool):
        print("  FAIL: family_leakage_confirmed must be a bool")
        ok = False
    if ok:
        print("  PASS")
    return ok


def main() -> int:
    adapter = MaterialsAdapter()
    cache = adapter.ensure_dataset()
    print(f"matbench cache: {cache}")

    # Direct adapter path (what the worker calls under the hood)
    direct = adapter.run_experiment(
        MaterialsExperimentSpec(
            feature_subset=["n_sites", "n_elements", "mean_Z"],
            methodology="LOFO",
            target_column="dielectric",
        )
    )
    ok_direct = _check_result(direct, label="MaterialsAdapter.run_experiment")

    # Registered tool path (what sub_agent_loop invokes)
    tool = materials_verification(feature_subset=["n_sites", "n_elements", "mean_Z"])
    if not tool.success:
        print(f"\n=== materials_verification tool ===\n  FAIL: {tool.error}")
        ok_tool = False
    else:
        ok_tool = _check_result(tool.output, label="materials_verification tool")

    # Magpie-style hypothesis text resolves to usable structural features
    magpie_feats = resolve_materials_features(
        "MagpieData mean Number and composition predict dielectric constant"
    )
    print(f"\n=== magpie alias resolution ===\n  resolved features: {magpie_feats}")
    if not magpie_feats:
        print("  FAIL: magpie alias did not resolve to features")
        ok_magpie = False
    else:
        magpie = adapter.run_experiment(
            MaterialsExperimentSpec(feature_subset=magpie_feats, methodology="LOFO")
        )
        ok_magpie = _check_result(magpie, label="magpie-resolved features")

    # Electronic / ionic literature-backed features
    electronic = adapter.run_experiment(
        MaterialsExperimentSpec(
            feature_subset=[
                "mean_atomic_mass",
                "mass_density",
                "mean_ionicity",
                "mean_coordination",
                "std_principal_quantum_n",
            ],
            methodology="LOFO",
        )
    )
    ok_electronic = _check_result(electronic, label="electronic/ionic features")

    # Real crystal-system families (not site-count quintiles)
    csys_counts = adapter.load_frame()["crystal_system"].value_counts()
    print(f"\n=== crystal_system families ===\n  {csys_counts.to_dict()}")
    ok_csys = len(csys_counts) >= 5 and csys_counts.min() >= 50
    if not ok_csys:
        print(f"  FAIL: need ≥5 crystal systems with n≥50 (min={csys_counts.min()})")
    else:
        print("  PASS")

    passed = ok_direct and ok_tool and ok_magpie and ok_electronic and ok_csys
    print("\n" + ("SMOKE PASS" if passed else "SMOKE FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
