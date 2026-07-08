"""Real GTEx v8 cross-tissue gene-expression subset (median TPM).

The frame is built from the GTEx v8 *median gene-level TPM* release
(``GTEx_Analysis_2017-06-05_v8_...gene_median_tpm.gct.gz``), a real public
transcriptomic atlas. We take one representative tissue column for each of 10
organs, log2-transform, keep the 1000 most-variable genes, and derive the
standard gene-level features (mean, variance, coefficient of variation, and the
Yanai tau tissue-specificity index) from the **real** expression matrix.

Grouping is by each gene's dominant (max-expression) tissue so the verifier's
leave-one-tissue-out (LOFO) + tissue-label-shuffle null runs on real tissues.

Provenance / license: see ``data/genomics/README.md``. If GTEx cannot be
reached and no real cache exists, a clearly-labelled synthetic fallback is
generated; ``dataset_is_synthetic()`` reports which one is on disk and the
plugin's ``uses_synthetic_data()`` reads it.
"""
from __future__ import annotations

import gzip
import io
import json
import urllib.request
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
SAMPLES_PER_TISSUE = 10  # synthetic-fallback only
RANDOM_SEED = 42

GTEX_URL = (
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
    "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
)
# Representative detailed GTEx tissue column per organ label above.
_GTEX_TISSUE_MAP: dict[str, str] = {
    "Brain": "Brain - Cortex",
    "Heart": "Heart - Left Ventricle",
    "Liver": "Liver",
    "Lung": "Lung",
    "Muscle": "Muscle - Skeletal",
    "Skin": "Skin - Not Sun Exposed (Suprapubic)",
    "Blood": "Whole Blood",
    "Adipose": "Adipose - Subcutaneous",
    "Thyroid": "Thyroid",
    "Colon": "Colon - Sigmoid",
}

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


def meta_path() -> Path:
    return cache_dir() / "gtex_subset_v1.meta.json"


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
    """True only when a REAL (non-synthetic) GTEx cache is already on disk.

    Unlike ``dataset_is_synthetic()`` this NEVER triggers a network fetch: it
    reads the existing cache meta and returns False when the cache is absent or
    was produced by the synthetic fallback. Real-data tests use this to
    ``pytest.skip`` cleanly (green-with-skips) instead of downloading GTEx in CI
    or vacuously passing on the synthetic frame. Run
    ``scripts/build_real_domain_datasets.py`` to populate the real cache.
    """
    return cache_path().is_file() and not dataset_is_synthetic()


def compute_gene_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gene-level features from a (gene_id, tissue, expression) long frame."""
    pivot = df.pivot_table(index="gene_id", columns="tissue", values="expression", aggfunc="mean")
    means = pivot.mean(axis=1)
    stds = pivot.std(axis=1).replace(0, np.nan)
    cv = (stds / means.replace(0, np.nan)).fillna(0.0)
    # Tau index (Yanai et al.) — 0 = housekeeping, 1 = tissue-specific.
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


def _build_real_long_frame(gz_bytes: bytes) -> pd.DataFrame:
    """Parse GTEx median-TPM .gct.gz into a (gene_id, tissue, expression) frame."""
    with gzip.open(io.BytesIO(gz_bytes), "rt") as fh:
        fh.readline()  # '#1.2'
        fh.readline()  # dims
        raw = pd.read_csv(fh, sep="\t")
    cols: dict[str, str] = {}
    for label, detailed in _GTEX_TISSUE_MAP.items():
        if detailed in raw.columns:
            cols[label] = detailed
            continue
        organ = detailed.split(" - ")[0]
        candidates = [c for c in raw.columns if c.startswith(organ)]
        if candidates:
            cols[label] = candidates[0]
    if len(cols) < 3:
        return pd.DataFrame()
    mat = raw[["Name", *cols.values()]].copy()
    mat.columns = ["gene_id", *cols.keys()]
    mat["gene_id"] = mat["gene_id"].astype(str).str.split(".").str[0]
    mat = mat.drop_duplicates(subset="gene_id").set_index("gene_id")
    expr = mat[(mat > 0.1).sum(axis=1) >= 5]
    logexpr = np.log2(expr + 1.0)
    top = logexpr.var(axis=1).sort_values(ascending=False).head(N_GENES).index
    sub = logexpr.loc[top]
    long = (
        sub.reset_index()
        .melt(id_vars="gene_id", var_name="tissue", value_name="expression")
    )
    features = compute_gene_features(long)
    return long.merge(features, on="gene_id", how="left")


def _write_readme(*, synthetic: bool, source: str) -> None:
    kind = "SYNTHETIC FALLBACK" if synthetic else "REAL DATA"
    cache_dir().joinpath("README.md").write_text(
        f"# genomics cache — {kind}\n\n"
        f"Source: {source}\n\n"
        "Full provenance, license and column documentation:\n"
        "`packages/propab-core/propab/domain_modules/genomics/DATA_PROVENANCE.md`\n\n"
        "This directory is git-ignored; the adapter regenerates it from the source "
        "on first use. Delete the CSV + meta to force a refresh.\n",
        encoding="utf-8",
    )


def _fetch_gtex(timeout: float = 180.0) -> bytes:
    req = urllib.request.Request(GTEX_URL, headers={"User-Agent": "propab-genomics-adapter/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted GTEx host)
        return resp.read()


def _synthetic_gtex_frame() -> pd.DataFrame:
    """Labelled synthetic fallback (used only if GTEx is unavailable)."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    gene_ids = [f"ENSG{i:011d}" for i in range(N_GENES)]
    for gi, gid in enumerate(gene_ids):
        base = rng.lognormal(mean=2.0 + 0.3 * (gi % 7), sigma=0.4)
        tissue_effects = rng.normal(0, 0.15 if gi % 5 == 0 else 0.8, len(TISSUES))
        for ti, tissue in enumerate(TISSUES):
            for _ in range(SAMPLES_PER_TISSUE):
                expr = max(0.01, base + tissue_effects[ti] + rng.normal(0, 0.25))
                rows.append({"gene_id": gid, "tissue": tissue, "expression": float(expr)})
    df = pd.DataFrame(rows)
    return df.merge(compute_gene_features(df), on="gene_id", how="left")


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
        # Never let the target column leak in as its own predictor: with the target
        # among the features the Ridge model reads off the answer and the
        # leave-one-tissue-out R² is trivially 1.0 regardless of the data.
        features = [f for f in features if f != target]
        if not features:
            features = [f for f in KNOWN_FEATURES if f != target][:2]
        return cls(feature_subset=features[:4], target_column=target)


class GenomicsAdapter:
    def _write(self, df: pd.DataFrame, *, synthetic: bool, source: str) -> Path:
        path = cache_path()
        df.to_csv(path, index=False)
        meta = {
            "synthetic": synthetic,
            "source": source,
            "n_genes": int(df["gene_id"].nunique()) if not df.empty else 0,
            "n_tissues": int(df["tissue"].nunique()) if not df.empty else 0,
            "tissues": sorted(df["tissue"].unique().tolist()) if not df.empty else list(TISSUES),
            "features": list(KNOWN_FEATURES),
        }
        meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _write_readme(synthetic=synthetic, source=source)
        return path

    def ensure_cache(self) -> Path:
        path = cache_path()
        if path.is_file():
            return path
        try:
            gz = _fetch_gtex()
            df = _build_real_long_frame(gz)
            if not df.empty and df["tissue"].nunique() >= 3 and df["gene_id"].nunique() >= 30:
                return self._write(
                    df,
                    synthetic=False,
                    source=(
                        "GTEx v8 median gene-level TPM "
                        "(GTEx_Analysis_2017-06-05_v8 gene_median_tpm)"
                    ),
                )
        except Exception:  # noqa: BLE001 (network/parse failures fall back)
            pass
        return self._write(_synthetic_gtex_frame(), synthetic=True, source="synthetic-fallback")

    def load_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.ensure_cache())
