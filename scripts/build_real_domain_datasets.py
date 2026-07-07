#!/usr/bin/env python3
"""Pre-populate the real biology-domain caches (genomics, enzyme_kinetics, mandrake).

The adapters fetch and cache the public datasets on first use, but this script
lets you build them ahead of time (e.g. before an offline campaign run) and
prints whether real data was served or the synthetic fallback was used. It
mirrors ``scripts/fetch_graph_datasets.py`` for the graph domains.

Datasets
--------
- genomics        — GTEx v8 median gene-level TPM (public; auto-fetched + cached).
- enzyme_kinetics — DLKcat kcat compilation, BRENDA + SABIO-RK derived
                    (public GitHub; auto-fetched + cached).
- mandrake        — 56 retroviral RT sequences + handcrafted biophysical
                    features. This is a PRIVATE dataset (``mandrake-data/``,
                    git-ignored): there is no public URL to fetch it from, so we
                    can only REPORT whether it is present. It must be provided
                    out-of-band; we never fabricate it.

Usage:
    PYTHONPATH="packages/propab-core;." python scripts/build_real_domain_datasets.py

Exit code is non-zero only on an unexpected error; a missing private mandrake
dataset or a synthetic fallback is reported, not treated as a failure (the
adapters keep an honest, clearly-labelled synthetic fallback so preflight and CI
never hard-fail offline).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))
sys.path.insert(0, str(ROOT))

from propab.domain_modules.enzyme_kinetics import adapter as enzyme  # noqa: E402
from propab.domain_modules.genomics import adapter as genomics  # noqa: E402


def _build(name: str, adapter, meta_path_fn) -> bool:
    """Build one auto-fetching cache. Returns True if REAL data was served."""
    print(f"[{name}] building cache ...")
    path = adapter.ensure_cache()
    meta = json.loads(meta_path_fn().read_text(encoding="utf-8"))
    is_synthetic = bool(meta.get("synthetic"))
    kind = "SYNTHETIC FALLBACK" if is_synthetic else "REAL DATA"
    print(f"[{name}] {kind}")
    print(f"[{name}]   file:   {path}")
    print(f"[{name}]   source: {meta.get('source')}")
    counts = meta.get("per_ec_counts") or {
        "n_genes": meta.get("n_genes"),
        "n_tissues": meta.get("n_tissues"),
    }
    print(f"[{name}]   size:   {counts}")
    if is_synthetic:
        print(
            f"[{name}]   NOTE:   real fetch unavailable (offline?); the honest "
            "synthetic fallback is cached. Re-run with network access to fetch "
            "the real dataset."
        )
    return not is_synthetic


def _report_mandrake() -> bool:
    """Report mandrake RT-set presence. Returns True if the real data is present.

    Mandrake is a private, non-public dataset: there is no URL to fetch it from,
    so this only inspects whether ``mandrake-data/`` (or ``/app/mandrake-data``)
    is already on disk. We never synthesize or fabricate it.
    """
    name = "mandrake"
    print(f"[{name}] checking private RT dataset ...")
    from propab.domain_adapters.mandrake_adapter import MandrakeAdapter

    data_dir = MandrakeAdapter.resolve_data_dir()
    feat = data_dir / "handcrafted_features.csv"
    seqs = data_dir / "rt_sequences.csv"
    if feat.is_file() and seqs.is_file():
        try:
            df = MandrakeAdapter(data_dir=data_dir).load_frame()
            n = int(len(df))
            fams = int(df["rt_family"].nunique())
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] PRESENT but unreadable: {exc}")
            return False
        print(f"[{name}] REAL DATA (private)")
        print(f"[{name}]   dir:     {data_dir}")
        print(f"[{name}]   samples: {n} RT sequences across {fams} families")
        return True
    print(f"[{name}] ABSENT — private dataset, no public URL to fetch from.")
    print(f"[{name}]   expected: {feat}")
    print(f"[{name}]             {seqs}")
    print(
        f"[{name}]   Provide the private mandrake-data/ out-of-band. It is "
        "git-ignored and cannot be fabricated; the mandrake tests skip cleanly "
        "when it is absent."
    )
    return False


def main() -> int:
    real = {}
    real["enzyme_kinetics"] = _build(
        "enzyme_kinetics", enzyme.EnzymeKineticsAdapter(), enzyme.meta_path
    )
    real["genomics"] = _build("genomics", genomics.GenomicsAdapter(), genomics.meta_path)
    real["mandrake"] = _report_mandrake()

    print("\nsummary:")
    for domain, is_real in real.items():
        status = "REAL" if is_real else "synthetic/absent"
        print(f"  {domain:16s} {status}")
    if not all(real.values()):
        print(
            "\nSome datasets are synthetic or absent. This is a non-fatal, honest "
            "state (offline or private data not provided). Real-data tests skip "
            "cleanly; re-run with network access / the private mandrake-data/ to "
            "exercise them."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
