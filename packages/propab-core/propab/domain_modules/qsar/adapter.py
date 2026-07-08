"""QSAR / drug-target bioactivity dataset adapter (leave-one-scaffold-out).

The frame is a compound-activity table: each row is a small molecule described by
standard 2D molecular descriptors (molecular weight, cLogP, H-bond donors /
acceptors, topological polar surface area, rotatable bonds, aromatic rings,
fraction sp3), grouped by its **Bemis-Murcko chemical scaffold**, with a
``pactivity`` (pIC50 = -log10(IC50 in molar)) target.

Real upgrade path: this adapter's ``ensure_cache`` will serve a real ChEMBL
bioactivity export (activities filtered to a single target / assay, RDKit
descriptors, RDKit ``GetScaffoldForMol`` scaffolds) when one is dropped at
``data/qsar/chembl_subset_v1.csv``. With no such file present it generates a
clearly-labelled SYNTHETIC compound table so the domain runs offline without any
private or licensed data; ``dataset_is_synthetic()`` reports which is on disk and
the plugin's ``uses_synthetic_data()`` reads it, so a finding is never passed off
as real when it is not.

The synthetic table carries a genuine (planted) structure-activity relationship —
a global descriptor -> potency law shared across scaffolds plus a small per-
scaffold offset — so the leave-one-scaffold-out verifier exercises real
generalization rather than a tautology.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from propab.config import settings

# Bemis-Murcko-style scaffold buckets (leave-one-scaffold-out groups).
SCAFFOLDS: tuple[str, ...] = (
    "benzimidazole", "quinazoline", "indole", "pyrimidine",
    "piperazine", "benzothiophene", "naphthalene", "pyrazole",
)
N_PER_SCAFFOLD = 90
RANDOM_SEED = 42

KNOWN_FEATURES: tuple[str, ...] = (
    "mol_weight",
    "clogp",
    "num_h_donors",
    "num_h_acceptors",
    "tpsa",
    "num_rotatable_bonds",
    "num_aromatic_rings",
    "fraction_csp3",
)


def cache_dir() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "qsar"
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path() -> Path:
    return cache_dir() / "chembl_subset_v1.csv"


def meta_path() -> Path:
    return cache_dir() / "chembl_subset_v1.meta.json"


def dataset_is_synthetic() -> bool:
    """True when the on-disk cache was produced by the synthetic fallback."""
    mp = meta_path()
    if not mp.is_file():
        return True
    try:
        return bool(json.loads(mp.read_text(encoding="utf-8")).get("synthetic", True))
    except Exception:  # noqa: BLE001
        return True


def real_data_cached() -> bool:
    """True only when a REAL (non-synthetic) ChEMBL cache is already on disk.

    Never triggers a fetch: real-data tests use it to ``pytest.skip`` cleanly
    rather than assert on the synthetic frame.
    """
    return cache_path().is_file() and not dataset_is_synthetic()


def _synthetic_frame(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Labelled synthetic compound-activity table with a planted SAR.

    A global descriptor->potency law (shared coefficients) holds across every
    scaffold; each scaffold adds a small potency offset and its descriptor
    distribution is centred slightly differently, so leave-one-scaffold-out is a
    genuine generalization test (not a within-row identity).
    """
    rng = np.random.default_rng(seed)
    # Global structure-activity coefficients on standardized-ish descriptors.
    rows: list[dict[str, Any]] = []
    for si, scaf in enumerate(SCAFFOLDS):
        scaffold_offset = rng.normal(0.0, 0.3)
        for j in range(N_PER_SCAFFOLD):
            mw = float(rng.normal(360 + 8 * si, 55))
            clogp = float(rng.normal(2.8, 1.1))
            hbd = float(max(0, round(rng.normal(2.0, 1.2))))
            hba = float(max(0, round(rng.normal(4.5, 1.6))))
            tpsa = float(max(0.0, rng.normal(75, 22)))
            rotb = float(max(0, round(rng.normal(5.0, 2.2))))
            arom = float(max(0, round(rng.normal(2.4, 1.0))))
            fsp3 = float(np.clip(rng.normal(0.38, 0.12), 0.0, 1.0))
            # Global QSAR law (loosely lipophilicity/size driven potency) +
            # scaffold offset + measurement noise.
            pactivity = (
                5.4
                + 0.55 * (clogp - 2.8)
                - 0.004 * (mw - 360)
                - 0.010 * (tpsa - 75)
                + 0.18 * (arom - 2.4)
                - 0.9 * (fsp3 - 0.38)
                + scaffold_offset
                + rng.normal(0, 0.45)
            )
            rows.append({
                "compound_id": f"CHEMBL_SYN_{si:02d}_{j:03d}",
                "scaffold": scaf,
                "mol_weight": mw,
                "clogp": clogp,
                "num_h_donors": hbd,
                "num_h_acceptors": hba,
                "tpsa": tpsa,
                "num_rotatable_bonds": rotb,
                "num_aromatic_rings": arom,
                "fraction_csp3": fsp3,
                "pactivity": float(pactivity),
            })
    return pd.DataFrame(rows)


def _write_readme(*, synthetic: bool, source: str) -> None:
    kind = "SYNTHETIC FALLBACK" if synthetic else "REAL DATA"
    cache_dir().joinpath("README.md").write_text(
        f"# qsar cache — {kind}\n\n"
        f"Source: {source}\n\n"
        "Real upgrade: drop a ChEMBL single-target activity export (RDKit "
        "descriptors + Bemis-Murcko scaffold column) at `chembl_subset_v1.csv` "
        "with a matching `.meta.json` (`synthetic: false`).\n\n"
        "This directory is git-ignored; the adapter regenerates it on first use.\n",
        encoding="utf-8",
    )


@dataclass
class QSARExperimentSpec:
    feature_subset: list[str]
    target_column: str = "pactivity"
    group_column: str = "scaffold"

    @classmethod
    def from_hypothesis(cls, hypothesis: dict[str, Any]) -> "QSARExperimentSpec":
        text = str(hypothesis.get("text") or hypothesis.get("statement") or "").lower()
        features = list(hypothesis.get("feature_subset") or [])
        if not features:
            if "lipophil" in text or "logp" in text or "clogp" in text:
                features = ["clogp", "num_aromatic_rings", "fraction_csp3", "mol_weight"]
            elif "polar" in text or "tpsa" in text or "hydrogen" in text or "h-bond" in text:
                features = ["tpsa", "num_h_donors", "num_h_acceptors", "mol_weight"]
            elif "size" in text or "weight" in text or "flexib" in text or "rotatable" in text:
                features = ["mol_weight", "num_rotatable_bonds", "num_aromatic_rings"]
            else:
                features = ["clogp", "mol_weight", "tpsa", "num_aromatic_rings"]
        return cls(feature_subset=features[:5], target_column="pactivity")


class QSARAdapter:
    def _write(self, df: pd.DataFrame, *, synthetic: bool, source: str) -> Path:
        path = cache_path()
        df.to_csv(path, index=False)
        meta = {
            "synthetic": synthetic,
            "source": source,
            "n_compounds": int(len(df)),
            "scaffolds": sorted(df["scaffold"].unique().tolist()) if not df.empty else list(SCAFFOLDS),
            "target": "pactivity",
            "features": list(KNOWN_FEATURES),
        }
        meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _write_readme(synthetic=synthetic, source=source)
        return path

    def ensure_cache(self) -> Path:
        path = cache_path()
        if path.is_file():
            return path
        # No trivially-fetchable single-file public source; a real ChEMBL export
        # dropped in place is served verbatim, else a labelled synthetic table.
        return self._write(
            _synthetic_frame(),
            synthetic=True,
            source="synthetic-fallback (planted SAR; real path: ChEMBL single-target export)",
        )

    def load_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.ensure_cache())
