"""GTEx-style gene expression subset — synthetic fallback, cache on disk."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from propab.config import settings

TISSUES: tuple[str, ...] = (
    "Brain", "Heart", "Liver", "Lung", "Muscle",
    "Skin", "Blood", "Adipose", "Thyroid", "Colon",
)
N_GENES = 1000
SAMPLES_PER_TISSUE = 10
RANDOM_SEED = 42

KNOWN_FEATURES: tuple[str, ...] = (
    "expression_variance",
    "tissue_specificity_tau",
    "mean_expression",
    "cv_across_tissues",
)


def cache_dir() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "genomics"
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path() -> Path:
    return cache_dir() / "gtex_subset_v1.csv"


def _synthetic_gtex_frame() -> pd.DataFrame:
    """10 tissues × 1000 genes — ~50MB class, generated in seconds."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    gene_ids = [f"ENSG{i:011d}" for i in range(N_GENES)]
    # Housekeeping-like genes: low tau, high cross-tissue correlation
    for gi, gid in enumerate(gene_ids):
        base = rng.lognormal(mean=2.0 + 0.3 * (gi % 7), sigma=0.4)
        tissue_effects = rng.normal(0, 0.15 if gi % 5 == 0 else 0.8, len(TISSUES))
        for ti, tissue in enumerate(TISSUES):
            for _ in range(SAMPLES_PER_TISSUE):
                expr = max(0.01, base + tissue_effects[ti] + rng.normal(0, 0.25))
                rows.append({
                    "gene_id": gid,
                    "tissue": tissue,
                    "expression": float(expr),
                })
    df = pd.DataFrame(rows)
    features = compute_gene_features(df)
    return df.merge(features, on="gene_id", how="left")


def compute_gene_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gene-level features from tissue expression matrix."""
    pivot = df.pivot_table(index="gene_id", columns="tissue", values="expression", aggfunc="mean")
    means = pivot.mean(axis=1)
    stds = pivot.std(axis=1).replace(0, np.nan)
    cv = (stds / means.replace(0, np.nan)).fillna(0.0)
    # Tau index (Yanai et al.) — 0 = housekeeping, 1 = tissue-specific
    ranked = pivot.rank(axis=1, method="average")
    n = pivot.shape[1]
    tau = ((ranked.sum(axis=1) - n) / (n - 1) / n).clip(0, 1)
    out = pd.DataFrame({
        "gene_id": pivot.index,
        "expression_variance": pivot.var(axis=1).values,
        "tissue_specificity_tau": tau.values,
        "mean_expression": means.values,
        "cv_across_tissues": cv.values,
    })
    return out


@dataclass
class GenomicsExperimentSpec:
    feature_subset: list[str]
    target_column: str = "mean_expression"
    tissue_column: str = "tissue"
    methodology: str = "LOFO"

    @classmethod
    def from_hypothesis(cls, hypothesis: dict[str, Any]) -> GenomicsExperimentSpec:
        text = str(hypothesis.get("text") or "").lower()
        features = list(hypothesis.get("feature_subset") or [])
        if not features:
            if "tau" in text or "specificity" in text:
                features = ["tissue_specificity_tau", "expression_variance"]
            elif "housekeeping" in text or "constitutive" in text:
                features = ["mean_expression", "cv_across_tissues"]
            else:
                features = ["expression_variance", "mean_expression"]
        target = "tissue_specificity_tau" if "tau" in text else "mean_expression"
        return cls(feature_subset=features[:4], target_column=target)


class GenomicsAdapter:
    def ensure_cache(self) -> Path:
        path = cache_path()
        if path.is_file():
            return path
        df = _synthetic_gtex_frame()
        df.to_csv(path, index=False)
        meta = {"n_genes": N_GENES, "n_tissues": len(TISSUES), "synthetic": True}
        cache_dir().joinpath("gtex_subset_v1.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )
        return path

    def load_frame(self) -> pd.DataFrame:
        path = self.ensure_cache()
        return pd.read_csv(path)
