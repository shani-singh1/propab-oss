"""Proteomics / protein-stability dataset adapter (leave-one-protein-family-out).

Each row is a protein described by sequence-derived features (length, molecular
weight, GRAVY hydropathy, fraction charged / helix-propensity / aromatic /
proline residues, and an instability-index proxy), grouped by its **protein
family (fold)**, with a thermostability target ``tm_celsius`` (melting
temperature).

Real upgrade path: ``ensure_cache`` serves a real melting-temperature export
(e.g. Meltome Atlas / ProThermDB with sequence-derived features and a Pfam family
column) dropped at ``data/proteomics/meltome_subset_v1.csv``. With no such file
present it generates a clearly-labelled SYNTHETIC protein table so the domain runs
offline; ``dataset_is_synthetic()`` reports which is on disk and the plugin's
``uses_synthetic_data()`` reads it.

The synthetic table carries a genuine (planted) family-shared stability rule (a
charge / hydropathy / proline law) plus a per-family offset, so leave-one-family-
out is a real generalization test rather than a within-row identity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from propab.config import settings

# Structural / fold families as the leave-one-family-out groups.
FAMILIES: tuple[str, ...] = (
    "TIM_barrel", "Rossmann_fold", "immunoglobulin", "ferredoxin",
    "beta_propeller", "ferritin_like", "OB_fold", "SH3_barrel",
)
N_PER_FAMILY = 80
RANDOM_SEED = 42

KNOWN_FEATURES: tuple[str, ...] = (
    "sequence_length",
    "molecular_weight",
    "gravy_hydropathy",
    "frac_charged",
    "frac_helix_propensity",
    "frac_aromatic",
    "frac_proline",
    "instability_index",
)


def cache_dir() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "proteomics"
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path() -> Path:
    return cache_dir() / "meltome_subset_v1.csv"


def meta_path() -> Path:
    return cache_dir() / "meltome_subset_v1.meta.json"


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
    """Labelled synthetic protein table with a planted family-shared stability rule."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for fi, fam in enumerate(FAMILIES):
        family_offset = rng.normal(0.0, 3.0)
        for j in range(N_PER_FAMILY):
            length = float(rng.normal(280 + 10 * fi, 60))
            mw = float(length * 110.0 + rng.normal(0, 800))
            gravy = float(rng.normal(-0.2, 0.35))
            charged = float(np.clip(rng.normal(0.24, 0.05), 0.0, 1.0))
            helix = float(np.clip(rng.normal(0.33, 0.06), 0.0, 1.0))
            arom = float(np.clip(rng.normal(0.08, 0.03), 0.0, 1.0))
            proline = float(np.clip(rng.normal(0.05, 0.02), 0.0, 1.0))
            instab = float(rng.normal(38, 9))
            # Global stability rule (higher charge/proline/hydropathy -> higher Tm)
            # + per-family offset + measurement noise. Units ~ Celsius.
            tm = (
                58.0
                + 60.0 * (charged - 0.24)
                + 90.0 * (proline - 0.05)
                + 6.0 * gravy
                - 0.25 * (instab - 38)
                + family_offset
                + rng.normal(0, 3.5)
            )
            rows.append({
                "protein_id": f"MELT_SYN_{fi:02d}_{j:03d}",
                "family": fam,
                "sequence_length": length,
                "molecular_weight": mw,
                "gravy_hydropathy": gravy,
                "frac_charged": charged,
                "frac_helix_propensity": helix,
                "frac_aromatic": arom,
                "frac_proline": proline,
                "instability_index": instab,
                "tm_celsius": float(tm),
            })
    return pd.DataFrame(rows)


def _write_readme(*, synthetic: bool, source: str) -> None:
    kind = "SYNTHETIC FALLBACK" if synthetic else "REAL DATA"
    cache_dir().joinpath("README.md").write_text(
        f"# proteomics cache — {kind}\n\n"
        f"Source: {source}\n\n"
        "Real upgrade: drop a Meltome Atlas / ProThermDB export (sequence-derived "
        "features + Pfam family + Tm) at `meltome_subset_v1.csv` with a matching "
        "`.meta.json` (`synthetic: false`).\n\n"
        "This directory is git-ignored; the adapter regenerates it on first use.\n",
        encoding="utf-8",
    )


@dataclass
class ProteomicsExperimentSpec:
    feature_subset: list[str]
    target_column: str = "tm_celsius"
    group_column: str = "family"

    @classmethod
    def from_hypothesis(cls, hypothesis: dict[str, Any]) -> "ProteomicsExperimentSpec":
        text = str(hypothesis.get("text") or hypothesis.get("statement") or "").lower()
        features = list(hypothesis.get("feature_subset") or [])
        if not features:
            if "charge" in text or "electrostatic" in text or "ionic" in text:
                features = ["frac_charged", "gravy_hydropathy", "frac_proline"]
            elif "proline" in text or "rigid" in text or "instability" in text:
                features = ["frac_proline", "instability_index", "frac_helix_propensity"]
            elif "hydropath" in text or "gravy" in text or "hydrophob" in text:
                features = ["gravy_hydropathy", "frac_aromatic", "frac_charged"]
            elif "length" in text or "size" in text or "weight" in text:
                features = ["sequence_length", "molecular_weight", "frac_charged"]
            else:
                features = ["frac_charged", "frac_proline", "gravy_hydropathy", "instability_index"]
        return cls(feature_subset=features[:5], target_column="tm_celsius")


class ProteomicsAdapter:
    def _write(self, df: pd.DataFrame, *, synthetic: bool, source: str) -> Path:
        path = cache_path()
        df.to_csv(path, index=False)
        meta = {
            "synthetic": synthetic,
            "source": source,
            "n_proteins": int(len(df)),
            "families": sorted(df["family"].unique().tolist()) if not df.empty else list(FAMILIES),
            "target": "tm_celsius",
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
            source="synthetic-fallback (planted stability rule; real path: Meltome Atlas / ProThermDB)",
        )

    def load_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.ensure_cache())
