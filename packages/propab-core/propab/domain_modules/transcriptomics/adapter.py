"""Transcriptomics / gene-regulation dataset adapter (leave-one-condition-out).

Each row is a gene measured under one experimental **condition** (perturbation),
described by promoter / regulatory features (GC content, CpG ratio, TATA-box
score, promoter length, TF-motif count, upstream conservation, and a chromatin-
accessibility proxy), with a regulatory target ``log2_fold_change`` (expression
response). Rows are grouped by condition so the verifier's leave-one-condition-out
(LOCO) holdout runs across perturbations.

Real upgrade path: ``ensure_cache`` serves a real differential-expression export
(e.g. GEO / a perturbation atlas with promoter-derived features and a condition
column) dropped at ``data/transcriptomics/geo_subset_v1.csv``. With no such file
present it generates a clearly-labelled SYNTHETIC table so the domain runs offline;
``dataset_is_synthetic()`` reports which is on disk and ``uses_synthetic_data()``
reads it.

The synthetic table carries a genuine (planted) condition-shared regulatory rule
(a promoter-feature -> response law) plus a per-condition offset, so leave-one-
condition-out is a real generalization test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from propab.config import settings

# Experimental perturbation conditions as the leave-one-condition-out groups.
CONDITIONS: tuple[str, ...] = (
    "heat_shock", "hypoxia", "oxidative_stress", "serum_starvation",
    "LPS_stimulation", "glucose_deprivation", "DNA_damage", "cold_shock",
)
N_PER_CONDITION = 110
RANDOM_SEED = 42

KNOWN_FEATURES: tuple[str, ...] = (
    "gc_content",
    "cpg_ratio",
    "tata_score",
    "promoter_length",
    "tf_motif_count",
    "conservation_score",
    "chromatin_accessibility",
)


def cache_dir() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "transcriptomics"
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path() -> Path:
    return cache_dir() / "geo_subset_v1.csv"


def meta_path() -> Path:
    return cache_dir() / "geo_subset_v1.meta.json"


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
    """Labelled synthetic gene-regulation table with a planted regulatory rule."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for ci, cond in enumerate(CONDITIONS):
        condition_offset = rng.normal(0.0, 0.4)
        for j in range(N_PER_CONDITION):
            gc = float(np.clip(rng.normal(0.5, 0.08), 0.2, 0.8))
            cpg = float(np.clip(rng.normal(0.6, 0.2), 0.0, 2.0))
            tata = float(rng.normal(0.0, 1.0))
            plen = float(rng.normal(1000, 250))
            motif = float(max(0, round(rng.normal(6.0, 2.5))))
            cons = float(np.clip(rng.normal(0.5, 0.15), 0.0, 1.0))
            atac = float(rng.normal(0.0, 1.0))
            # Global regulatory rule: motif density + accessibility drive the
            # magnitude of the expression response; CpG modulates; per-condition
            # offset + biological noise.
            lfc = (
                0.35 * (motif - 6.0)
                + 0.8 * atac
                + 0.6 * (cpg - 0.6)
                + 0.5 * tata
                + condition_offset
                + rng.normal(0, 0.6)
            )
            rows.append({
                "gene_id": f"GENE_SYN_{ci:02d}_{j:04d}",
                "condition": cond,
                "gc_content": gc,
                "cpg_ratio": cpg,
                "tata_score": tata,
                "promoter_length": plen,
                "tf_motif_count": motif,
                "conservation_score": cons,
                "chromatin_accessibility": atac,
                "log2_fold_change": float(lfc),
            })
    return pd.DataFrame(rows)


def _write_readme(*, synthetic: bool, source: str) -> None:
    kind = "SYNTHETIC FALLBACK" if synthetic else "REAL DATA"
    cache_dir().joinpath("README.md").write_text(
        f"# transcriptomics cache — {kind}\n\n"
        f"Source: {source}\n\n"
        "Real upgrade: drop a GEO / perturbation-atlas differential-expression "
        "export (promoter features + condition + log2 fold change) at "
        "`geo_subset_v1.csv` with a matching `.meta.json` (`synthetic: false`).\n\n"
        "This directory is git-ignored; the adapter regenerates it on first use.\n",
        encoding="utf-8",
    )


@dataclass
class TranscriptomicsExperimentSpec:
    feature_subset: list[str]
    target_column: str = "log2_fold_change"
    group_column: str = "condition"

    @classmethod
    def from_hypothesis(cls, hypothesis: dict[str, Any]) -> "TranscriptomicsExperimentSpec":
        text = str(hypothesis.get("text") or hypothesis.get("statement") or "").lower()
        features = list(hypothesis.get("feature_subset") or [])
        if not features:
            if "motif" in text or "transcription factor" in text or "tf " in text:
                features = ["tf_motif_count", "chromatin_accessibility", "tata_score"]
            elif "accessib" in text or "chromatin" in text or "atac" in text:
                features = ["chromatin_accessibility", "tf_motif_count", "cpg_ratio"]
            elif "cpg" in text or "methyl" in text or "gc content" in text or "gc-content" in text:
                features = ["cpg_ratio", "gc_content", "conservation_score"]
            elif "conservation" in text or "length" in text or "promoter" in text:
                features = ["conservation_score", "promoter_length", "gc_content"]
            else:
                features = ["tf_motif_count", "chromatin_accessibility", "cpg_ratio", "tata_score"]
        return cls(feature_subset=features[:5], target_column="log2_fold_change")


class TranscriptomicsAdapter:
    def _write(self, df: pd.DataFrame, *, synthetic: bool, source: str) -> Path:
        path = cache_path()
        df.to_csv(path, index=False)
        meta = {
            "synthetic": synthetic,
            "source": source,
            "n_rows": int(len(df)),
            "conditions": sorted(df["condition"].unique().tolist()) if not df.empty else list(CONDITIONS),
            "target": "log2_fold_change",
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
            source="synthetic-fallback (planted regulatory rule; real path: GEO / perturbation atlas)",
        )

    def load_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.ensure_cache())
