"""Immunology / peptide-MHC epitope dataset adapter (leave-one-MHC-allele-out).

Each row is a candidate T-cell epitope: a short peptide described by
physicochemical features (length, mean hydrophobicity, net charge, aromatic
fraction, anchor-position hydrophobicity at P2 and the C-terminus, molecular
weight, proline fraction), grouped by its **MHC allele**, with a binding target
``binding_score`` (higher = stronger binder; a -log10(IC50 nM) style scale).

Real upgrade path: ``ensure_cache`` serves a real IEDB / NetMHCpan-style binding
export (peptide, allele, measured IC50) dropped at ``data/epitope/iedb_subset_v1.csv``.
With no such file present it generates a clearly-labelled SYNTHETIC epitope table
so the domain runs offline without private data; ``dataset_is_synthetic()``
reports which is on disk and the plugin's ``uses_synthetic_data()`` reads it.

The synthetic table carries a genuine (planted) allele-shared binding rule (an
anchor-hydrophobicity / charge law) plus a per-allele offset, so leave-one-allele-
out is a real generalization test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from propab.config import settings

# Common HLA class-I alleles as the leave-one-allele-out groups.
ALLELES: tuple[str, ...] = (
    "HLA-A*02:01", "HLA-A*01:01", "HLA-A*03:01", "HLA-A*24:02",
    "HLA-B*07:02", "HLA-B*08:01", "HLA-B*44:02", "HLA-C*07:01",
)
N_PER_ALLELE = 90
RANDOM_SEED = 42

KNOWN_FEATURES: tuple[str, ...] = (
    "peptide_length",
    "mean_hydrophobicity",
    "net_charge",
    "aromatic_fraction",
    "anchor2_hydrophobicity",
    "anchorC_hydrophobicity",
    "mol_weight",
    "proline_fraction",
)


def cache_dir() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "epitope"
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path() -> Path:
    return cache_dir() / "iedb_subset_v1.csv"


def meta_path() -> Path:
    return cache_dir() / "iedb_subset_v1.meta.json"


def dataset_is_synthetic() -> bool:
    mp = meta_path()
    if not mp.is_file():
        return True
    try:
        return bool(json.loads(mp.read_text(encoding="utf-8")).get("synthetic", True))
    except Exception:  # noqa: BLE001
        return True


def real_data_cached() -> bool:
    return cache_path().is_file() and not dataset_is_synthetic()


def _synthetic_frame(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Labelled synthetic epitope table with a planted allele-shared binding rule."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for ai, allele in enumerate(ALLELES):
        allele_offset = rng.normal(0.0, 0.4)
        for j in range(N_PER_ALLELE):
            length = float(rng.integers(8, 12))
            mean_hyd = float(rng.normal(0.0, 0.9))
            net_charge = float(rng.normal(0.0, 1.3))
            arom = float(np.clip(rng.normal(0.15, 0.08), 0.0, 1.0))
            anchor2 = float(rng.normal(0.2, 1.0))
            anchorC = float(rng.normal(0.5, 1.0))
            mw = float(rng.normal(1050 + 20 * ai, 130))
            pro = float(np.clip(rng.normal(0.08, 0.05), 0.0, 1.0))
            # Global binding rule: anchor hydrophobicity favourable, extreme net
            # charge unfavourable, plus per-allele offset + measurement noise.
            binding = (
                6.0
                + 0.7 * anchorC
                + 0.5 * anchor2
                + 0.3 * mean_hyd
                - 0.45 * abs(net_charge)
                - 0.6 * (length - 9.0) ** 2 * 0.1
                + allele_offset
                + rng.normal(0, 0.5)
            )
            rows.append({
                "peptide_id": f"IEDB_SYN_{ai:02d}_{j:03d}",
                "allele": allele,
                "peptide_length": length,
                "mean_hydrophobicity": mean_hyd,
                "net_charge": net_charge,
                "aromatic_fraction": arom,
                "anchor2_hydrophobicity": anchor2,
                "anchorC_hydrophobicity": anchorC,
                "mol_weight": mw,
                "proline_fraction": pro,
                "binding_score": float(binding),
            })
    return pd.DataFrame(rows)


def _write_readme(*, synthetic: bool, source: str) -> None:
    kind = "SYNTHETIC FALLBACK" if synthetic else "REAL DATA"
    cache_dir().joinpath("README.md").write_text(
        f"# epitope cache — {kind}\n\n"
        f"Source: {source}\n\n"
        "Real upgrade: drop an IEDB / NetMHCpan binding export (peptide, allele, "
        "measured IC50 -> binding_score) at `iedb_subset_v1.csv` with a matching "
        "`.meta.json` (`synthetic: false`).\n\n"
        "This directory is git-ignored; the adapter regenerates it on first use.\n",
        encoding="utf-8",
    )


@dataclass
class EpitopeExperimentSpec:
    feature_subset: list[str]
    target_column: str = "binding_score"
    group_column: str = "allele"

    @classmethod
    def from_hypothesis(cls, hypothesis: dict[str, Any]) -> "EpitopeExperimentSpec":
        text = str(hypothesis.get("text") or hypothesis.get("statement") or "").lower()
        features = list(hypothesis.get("feature_subset") or [])
        if not features:
            if "anchor" in text:
                features = ["anchor2_hydrophobicity", "anchorC_hydrophobicity", "mean_hydrophobicity"]
            elif "charge" in text or "electrostatic" in text:
                features = ["net_charge", "mean_hydrophobicity", "aromatic_fraction"]
            elif "length" in text or "size" in text or "weight" in text:
                features = ["peptide_length", "mol_weight", "proline_fraction"]
            else:
                features = ["anchorC_hydrophobicity", "net_charge", "mean_hydrophobicity", "peptide_length"]
        return cls(feature_subset=features[:5], target_column="binding_score")


class EpitopeAdapter:
    def _write(self, df: pd.DataFrame, *, synthetic: bool, source: str) -> Path:
        path = cache_path()
        df.to_csv(path, index=False)
        meta = {
            "synthetic": synthetic,
            "source": source,
            "n_peptides": int(len(df)),
            "alleles": sorted(df["allele"].unique().tolist()) if not df.empty else list(ALLELES),
            "target": "binding_score",
            "features": list(KNOWN_FEATURES),
        }
        meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _write_readme(synthetic=synthetic, source=source)
        return path

    def ensure_cache(self) -> Path:
        path = cache_path()
        if path.is_file():
            return path
        return self._write(
            _synthetic_frame(),
            synthetic=True,
            source="synthetic-fallback (planted binding rule; real path: IEDB/NetMHCpan export)",
        )

    def load_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.ensure_cache())
