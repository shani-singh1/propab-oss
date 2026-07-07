#!/usr/bin/env python3
"""Pre-populate the real enzyme_kinetics (DLKcat) and genomics (GTEx) caches.

The adapters fetch and cache these on first use, but this script lets you build
them ahead of time (e.g. before an offline campaign run) and prints whether real
data was served or the synthetic fallback was used.

Usage:
    PYTHONPATH="packages/propab-core;." python scripts/build_real_domain_datasets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.domain_modules.enzyme_kinetics import adapter as enzyme  # noqa: E402
from propab.domain_modules.genomics import adapter as genomics  # noqa: E402


def _build(name: str, adapter, meta_path_fn) -> None:
    print(f"[{name}] building cache ...")
    path = adapter.ensure_cache()
    meta = json.loads(meta_path_fn().read_text(encoding="utf-8"))
    kind = "SYNTHETIC FALLBACK" if meta.get("synthetic") else "REAL DATA"
    print(f"[{name}] {kind}")
    print(f"[{name}]   file:   {path}")
    print(f"[{name}]   source: {meta.get('source')}")
    counts = meta.get("per_ec_counts") or {
        "n_genes": meta.get("n_genes"),
        "n_tissues": meta.get("n_tissues"),
    }
    print(f"[{name}]   size:   {counts}")


def main() -> int:
    _build("enzyme_kinetics", enzyme.EnzymeKineticsAdapter(), enzyme.meta_path)
    _build("genomics", genomics.GenomicsAdapter(), genomics.meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
