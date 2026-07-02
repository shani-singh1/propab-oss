#!/usr/bin/env python3
"""Print one matbench dielectric structure row and derived feature vector."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.domain_adapters.materials_adapter import MaterialsAdapter, _KNOWN_FEATURES  # noqa: E402
from propab.domain_adapters.materials_crystal_system import symmetry_from_structure_dict  # noqa: E402
from propab.domain_adapters.materials_featurizer import featurize_structure  # noqa: E402


def main() -> int:
    adapter = MaterialsAdapter()
    cache = adapter.ensure_dataset()
    raw = json.loads(gzip.decompress(cache.read_bytes()).decode("utf-8"))
    entry = raw["data"][0]
    struct, target = entry[0], entry[1]

    print(f"cache: {cache}")
    print(f"target (dielectric): {target}")
    print("\n--- structure top-level keys ---")
    print(sorted(struct.keys()))
    print("\n--- lattice ---")
    print(json.dumps(struct.get("lattice"), indent=2)[:1200])
    print("\n--- first site ---")
    sites = struct.get("sites") or []
    if sites:
        print(json.dumps(sites[0], indent=2)[:800])
    print(f"\n--- n_sites: {len(sites)} ---")

    feats = featurize_structure(struct)
    sym = symmetry_from_structure_dict(struct)
    print("\n--- pymatgen symmetry ---")
    print(json.dumps(sym, indent=2))
    print("\n--- derived features ---")
    for name in _KNOWN_FEATURES:
        print(f"  {name}: {feats.get(name)}")

    df = adapter.load_frame()
    print(f"\n--- frame: {len(df)} rows, columns: {list(df.columns)} ---")
    print("crystal_system counts:")
    print(df["crystal_system"].value_counts().to_string())
    print(df[list(_KNOWN_FEATURES)].describe().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
