"""Matbench perovskites adapter — A-site LOFO on formation energy."""
from __future__ import annotations

import gzip
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from propab.domain_adapters.mandrake_adapter import (
    _bootstrap_lofo_ci,
    _family_baseline_r2,
    _family_label_shuffle_null,
    _family_lofo_breakdown,
    _global_r2,
    _leave_one_family_out_r2,
    _make_model,
    _permutation_p_value,
    _within_family_r2,
)
from propab.domain_adapters.materials_featurizer import featurize_structure
from propab.domain_adapters.perovskites_a_site import a_site_group
from propab.artifact_verification import _survives_label_shuffle_lofo

_MATBENCH_URL = "https://ml.materialsproject.org/projects/matbench_perovskites.json.gz"

KNOWN_FEATURES: tuple[str, ...] = (
    "n_sites",
    "n_elements",
    "mean_Z",
    "std_Z",
    "std_principal_quantum_n",
    "mean_atomic_mass",
    "mass_density",
    "mean_electronegativity",
    "mean_ionicity",
    "mean_coordination",
)

DEFAULT_TARGET = "e_form"
DEFAULT_FAMILY = "a_site_group"
DEFAULT_DATA_DIRS = (
    Path("/app/data/v1_candidates"),
    Path("data/v1_candidates"),
)
_FEAT_CACHE = "matbench_perovskites_featurized.parquet"


@dataclass
class PerovskitesExperimentSpec:
    feature_subset: list[str]
    methodology: str = "LOFO"
    target_column: str = DEFAULT_TARGET
    family_column: str = DEFAULT_FAMILY
    metric: str = "lofo_r2"
    baseline_model: str = "ridge"

    def to_tool_params(self) -> dict[str, Any]:
        return {
            "feature_subset": self.feature_subset,
            "methodology": self.methodology,
            "target_column": self.target_column,
            "family_column": self.family_column,
            "metric": self.metric,
            "baseline_model": self.baseline_model,
        }


class PerovskitesAdapter:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or self.resolve_data_dir()

    @staticmethod
    def resolve_data_dir() -> Path:
        for candidate in DEFAULT_DATA_DIRS:
            if (candidate / "matbench_perovskites.json.gz").is_file():
                return candidate
        return DEFAULT_DATA_DIRS[-1]

    def _cache_path(self) -> Path:
        return self.data_dir / "matbench_perovskites.json.gz"

    def _featurized_cache_path(self) -> Path:
        return self.data_dir / _FEAT_CACHE

    def ensure_dataset(self) -> Path:
        path = self._cache_path()
        if path.is_file():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            _MATBENCH_URL,
            headers={"User-Agent": "propab-oss/1.0 (perovskites adapter)"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            path.write_bytes(resp.read())
        return path

    def load_frame(self, *, rebuild_cache: bool = False) -> pd.DataFrame:
        raw_path = self.ensure_dataset()
        feat_path = self._featurized_cache_path()
        if not rebuild_cache and feat_path.is_file():
            if feat_path.stat().st_mtime >= raw_path.stat().st_mtime:
                df = pd.read_parquet(feat_path)
                if len(df) >= 500 and df[DEFAULT_FAMILY].nunique() >= 5:
                    return df

        raw = json.loads(gzip.decompress(raw_path.read_bytes()).decode("utf-8"))
        rows: list[dict[str, Any]] = []
        for entry in raw.get("data", []):
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            struct_dict, target = entry[0], entry[1]
            if not isinstance(struct_dict, dict):
                continue
            struct = Structure.from_dict(struct_dict)
            row = featurize_structure(struct_dict)
            row[DEFAULT_TARGET] = float(target)
            row[DEFAULT_FAMILY] = a_site_group(struct)
            rows.append(row)
        df = pd.DataFrame(rows).dropna(subset=[DEFAULT_TARGET, DEFAULT_FAMILY])
        if len(df) < 500:
            raise ValueError(f"insufficient_perovskites_rows: {len(df)}")
        if df[DEFAULT_FAMILY].nunique() < 5:
            raise ValueError("insufficient_a_site_families")
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(feat_path, index=False)
        return df

    def run_experiment(
        self,
        spec: PerovskitesExperimentSpec,
        *,
        df: pd.DataFrame | None = None,
        preflight: bool = False,
    ) -> dict[str, Any]:
        if df is None:
            df = self.load_frame()
        cols = [c for c in spec.feature_subset if c in df.columns]
        if not cols:
            raise ValueError(f"No usable features: {spec.feature_subset}")
        sub = df[[spec.target_column, spec.family_column, *cols]].copy()
        for c in cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.dropna()
        if len(sub) < 50:
            raise ValueError(f"Too few rows: {len(sub)}")
        X = sub[cols].to_numpy(dtype=float)
        y = sub[spec.target_column].to_numpy(dtype=float)
        families = sub[spec.family_column].astype(str).to_numpy()
        model = _make_model(spec.baseline_model)
        fam_base = _family_baseline_r2(y, families)
        within = _within_family_r2(X, y, families, model)
        lofo = _leave_one_family_out_r2(X, y, families, model)
        lofo_gap = within - lofo
        n_perm = 100 if preflight else 300
        if preflight:
            family_breakdown = {}
            bootstrap_ci = [0.0, 0.0]
            permutation_p = 1.0
        else:
            family_breakdown = _family_lofo_breakdown(X, y, families, model)
            bootstrap_ci = _bootstrap_lofo_ci(X, y, families, model)
            permutation_p = _permutation_p_value(X, y, families, model, observed=lofo)
        _, label_shuffle_p, label_null = _family_label_shuffle_null(
            X, y, families, model, n_perm=n_perm,
        )
        label_shuffle_p95 = float(np.percentile(label_null, 95)) if label_null else None
        exp_blob = {
            "lofo_r2": lofo,
            "mean_r2": lofo,
            "lofo_gap": lofo_gap,
            "label_shuffle_null_p95": label_shuffle_p95,
            "label_shuffle_permutation_p": label_shuffle_p,
            "permutation_p": permutation_p,
        }
        lofo_check = _survives_label_shuffle_lofo(exp_blob)
        return {
            "mean_r2": lofo,
            "lofo_r2": lofo,
            "within_family_r2": within,
            "global_r2": _global_r2(X, y, model),
            "family_baseline_r2": fam_base,
            "lofo_gap": round(lofo_gap, 4),
            "surprise_score": round(lofo - fam_base, 4),
            "bootstrap_ci": bootstrap_ci,
            "permutation_p": permutation_p,
            "label_shuffle_permutation_p": round(label_shuffle_p, 4),
            "label_shuffle_null_p95": round(label_shuffle_p95, 6) if label_shuffle_p95 is not None else None,
            "family_leakage_confirmed": bool(lofo_check.survived),
            "family_leakage_rationale": lofo_check.rationale,
            "family_breakdown": family_breakdown,
            "feature_subset": cols,
            "methodology": spec.methodology,
            "metric": spec.metric,
            "baseline_model": spec.baseline_model,
            "n_samples": len(y),
            "n_families": len(np.unique(families)),
            "group_column": spec.family_column,
            "target_column": spec.target_column,
            "data_source": str(self._cache_path()),
            "verified": True,
            "metric_value": lofo,
            "p_value": permutation_p,
            "confidence_interval": bootstrap_ci,
        }


def run_perovskites_lofo(
    *,
    features: list[str],
    df: pd.DataFrame | None = None,
    preflight: bool = False,
) -> dict[str, Any]:
    return PerovskitesAdapter().run_experiment(
        PerovskitesExperimentSpec(feature_subset=features, methodology="LOFO"),
        df=df,
        preflight=preflight,
    )
