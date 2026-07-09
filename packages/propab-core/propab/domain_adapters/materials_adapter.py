"""Materials (matbench dielectric) domain adapter — crystal-system LOFO verification."""
from __future__ import annotations

import gzip
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from propab.domain_adapters.materials_crystal_system import symmetry_from_structure_dict
from propab.domain_adapters.materials_featurizer import featurize_structure
from propab.domain_adapters.materials_frame_cache import (
    load_symmetry_cache,
    save_symmetry_cache,
)
from propab.domain_adapters.materials_mp_bandgap import load_bandgap_cache
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
from propab.artifact_verification import _survives_label_shuffle_lofo

_MATBENCH_URL = "https://ml.materialsproject.org/projects/matbench_dielectric.json.gz"

_KNOWN_FEATURES: tuple[str, ...] = (
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
    "mp_bandgap",
    "space_group_number",
)

_FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "magpie": _KNOWN_FEATURES,
    "composition": ("n_elements", "mean_Z", "std_Z", "mean_atomic_mass", "std_principal_quantum_n"),
    "structural": ("n_sites", "mean_coordination", "mass_density"),
    "compositional": ("n_elements", "mean_Z", "std_Z", "mean_atomic_mass", "std_principal_quantum_n"),
    "dielectric": ("mean_ionicity", "mean_electronegativity", "mass_density", "mean_atomic_mass"),
    "electronic": ("mean_electronegativity", "std_principal_quantum_n", "mean_Z", "std_Z"),
    "ionic": ("mean_ionicity", "mean_electronegativity", "mean_coordination"),
    "takahashi": ("mean_atomic_mass", "mass_density", "std_principal_quantum_n"),
    "kang": ("mean_coordination", "mean_ionicity", "mean_electronegativity"),
    "noda": ("mean_ionicity", "mean_electronegativity"),
    "polarizability": ("mean_electronegativity", "mean_atomic_mass", "mean_ionicity"),
    "coordination": ("mean_coordination",),
    "density": ("mass_density",),
    "atomic mass": ("mean_atomic_mass",),
    "ionicity": ("mean_ionicity",),
    "bandgap": ("mp_bandgap",),
    "penn": ("mp_bandgap", "mean_electronegativity"),
    "electronic structure": ("mp_bandgap", "mean_electronegativity"),
}

DEFAULT_TARGET = "dielectric"
DEFAULT_FAMILY = "crystal_system"
DEFAULT_DATA_DIRS = (
    Path("/app/data/v1_candidates"),
    Path("data/v1_candidates"),
)


@dataclass
class MaterialsExperimentSpec:
    feature_subset: list[str]
    methodology: str
    target_column: str = DEFAULT_TARGET
    family_column: str = DEFAULT_FAMILY
    metric: str = "lofo_r2"
    baseline_model: str = "ridge"
    compare_features: list[str] = field(default_factory=list)

    def to_tool_params(self) -> dict[str, Any]:
        return {
            "feature_subset": self.feature_subset,
            "methodology": self.methodology,
            "target_column": self.target_column,
            "family_column": self.family_column,
            "metric": self.metric,
            "baseline_model": self.baseline_model,
            "compare_features": self.compare_features or None,
        }

    @classmethod
    def from_hypothesis(cls, hypothesis: dict[str, Any], *, question: str = "") -> MaterialsExperimentSpec:
        text = str(hypothesis.get("text") or "")
        methodology = "LOFO"
        features: list[str] = list(hypothesis.get("feature_subset") or [])
        compare: list[str] = list(hypothesis.get("compare_features") or [])
        model = "ridge"
        tm = hypothesis.get("test_methodology") or ""
        if isinstance(tm, str) and tm.strip().startswith("{"):
            try:
                plan = json.loads(tm)
                if isinstance(plan, dict):
                    methodology = str(plan.get("methodology") or plan.get("method") or methodology)
                    features = list(plan.get("features") or plan.get("feature_subset") or features)
                    compare = list(plan.get("compare_features") or compare)
                    model = str(plan.get("model") or plan.get("baseline_model") or model)
            except json.JSONDecodeError:
                pass
        if not features:
            features = resolve_materials_features(text)
        if not features:
            features = list(_KNOWN_FEATURES)
        return cls(
            feature_subset=features[:6],
            methodology=methodology.upper(),
            baseline_model=model,
            compare_features=compare[:6],
        )


class MaterialsAdapter:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or self.resolve_data_dir()

    @staticmethod
    def resolve_data_dir() -> Path:
        for candidate in DEFAULT_DATA_DIRS:
            if (candidate / "matbench_dielectric.json.gz").is_file():
                return candidate
        return DEFAULT_DATA_DIRS[-1]

    def _cache_path(self) -> Path:
        return self.data_dir / "matbench_dielectric.json.gz"

    def ensure_dataset(self) -> Path:
        path = self._cache_path()
        if path.is_file():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            _MATBENCH_URL,
            headers={"User-Agent": "propab-oss/1.0 (materials adapter)"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
        return path

    def load_frame(self) -> pd.DataFrame:
        cache = self.ensure_dataset()
        raw = json.loads(gzip.decompress(cache.read_bytes()).decode("utf-8"))
        symmetry_cached = load_symmetry_cache(self.data_dir) or {}
        bandgap_cached = load_bandgap_cache(self.data_dir)
        rows: list[dict[str, Any]] = []
        symmetry_to_save: dict[int, dict[str, Any]] = dict(
            (int(k), v) for k, v in symmetry_cached.items() if str(k).isdigit()
        )

        for idx, entry in enumerate(raw.get("data", [])[:8000]):
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            struct, target = entry[0], entry[1]
            if not isinstance(struct, dict):
                continue
            row = featurize_structure(struct)
            row["dielectric"] = float(target)

            sym_key = str(idx)
            if sym_key in symmetry_cached:
                sym = symmetry_cached[sym_key]
            elif idx in symmetry_to_save:
                sym = symmetry_to_save[idx]
            else:
                sym = symmetry_from_structure_dict(struct)
                symmetry_to_save[idx] = sym
            row["crystal_system"] = sym["crystal_system"]
            row["space_group_number"] = float(sym["space_group_number"])

            if bandgap_cached and sym_key in bandgap_cached:
                row["mp_bandgap"] = float(bandgap_cached[sym_key])

            rows.append(row)

        if len(symmetry_to_save) > len(symmetry_cached):
            save_symmetry_cache(self.data_dir, symmetry_to_save)

        df = pd.DataFrame(rows).dropna(subset=["dielectric", "crystal_system"])
        if len(df) < 200:
            raise ValueError(f"insufficient_matbench_rows: {len(df)}")
        if df["crystal_system"].nunique() < 3:
            raise ValueError(
                f"insufficient_crystal_system_families: {df['crystal_system'].value_counts().to_dict()}"
            )
        return df

    def run_experiment(self, spec: MaterialsExperimentSpec) -> dict[str, Any]:
        df = self.load_frame()
        # Never let the target leak in as its own predictor (trivial R²=1.0). Guard
        # against a hypothesis/plan that lists the target in feature_subset.
        cols = [c for c in spec.feature_subset if c in df.columns and c != spec.target_column]
        if not cols:
            raise ValueError(f"No usable features: {spec.feature_subset}")
        sub = df[[spec.target_column, spec.family_column, *cols]].copy()
        for c in cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.dropna()
        if len(sub) < 8:
            raise ValueError(f"Too few rows: {len(sub)}")
        X = sub[cols].to_numpy(dtype=float)
        y = sub[spec.target_column].to_numpy(dtype=float)
        families = sub[spec.family_column].astype(str).to_numpy()
        model = _make_model(spec.baseline_model)
        fam_base = _family_baseline_r2(y, families)
        within = _within_family_r2(X, y, families, model)
        lofo = _leave_one_family_out_r2(X, y, families, model)
        lofo_gap = within - lofo
        family_breakdown = _family_lofo_breakdown(X, y, families, model)
        bootstrap_ci = _bootstrap_lofo_ci(X, y, families, model)
        permutation_p = _permutation_p_value(X, y, families, model, observed=lofo)
        _, label_shuffle_p, label_null = _family_label_shuffle_null(
            X, y, families, model, n_perm=300,
        )
        label_shuffle_p95 = float(np.percentile(label_null, 95)) if label_null else None
        compare_result = None
        if spec.compare_features:
            cmp_cols = [c for c in spec.compare_features if c in df.columns]
            if cmp_cols:
                Xc = sub[cmp_cols].to_numpy(dtype=float)
                cmp_lofo = _leave_one_family_out_r2(Xc, y, families, model)
                compare_result = {
                    "features": cmp_cols,
                    "mean_r2": cmp_lofo,
                    "delta_vs_primary": lofo - cmp_lofo,
                }
        exp_blob = {
            "lofo_r2": lofo,
            "mean_r2": lofo,
            "lofo_gap": lofo_gap,
            "label_shuffle_null_p95": label_shuffle_p95,
            "label_shuffle_permutation_p": label_shuffle_p,
            "permutation_p": permutation_p,
        }
        lofo_check = _survives_label_shuffle_lofo(exp_blob)
        result = {
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
            "compare_result": compare_result,
            "executed_code": (
                f"# Materials LOFO (matbench dielectric)\n"
                f"features = {cols!r}\n"
                f"group = {spec.family_column!r}\n"
                f"target = {spec.target_column!r}\n"
            ),
            "data_source": str(self._cache_path()),
            "verified": True,
            "metric_value": lofo,
            "p_value": permutation_p,
            "confidence_interval": bootstrap_ci,
        }
        return result

    def write_artifacts(self, out_dir: Path, spec: MaterialsExperimentSpec, result: dict[str, Any]) -> dict[str, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        (out_dir / "verification_results.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8",
        )
        paths["verification_results"] = str(out_dir / "verification_results.json")
        return paths


def is_materials_campaign(*, question: str = "", payload: dict[str, Any] | None = None) -> bool:
    q = (question or "").lower()
    if "domain_profile:materials" in q:
        return True
    if payload:
        if str(payload.get("domain_profile") or "") == "materials":
            return True
        if str(payload.get("domain") or "") == "materials":
            return True
    return False


def resolve_materials_features(text: str) -> list[str]:
    found: list[str] = []
    blob = (text or "").lower()
    for feat in _KNOWN_FEATURES:
        if feat in blob:
            found.append(feat)
    for alias, mapped in _FEATURE_ALIASES.items():
        if alias in blob:
            found.extend(mapped)
    for block in re.findall(r"\[([^\]]+)\]", text or ""):
        for part in re.split(r"[,|\s]+", block):
            p = part.strip().strip("'\"")
            if p in _KNOWN_FEATURES:
                found.append(p)
    # Underscore-free variants (e.g. "mean atomic mass")
    for feat in _KNOWN_FEATURES:
        spaced = feat.replace("_", " ")
        if spaced in blob:
            found.append(feat)
    return list(dict.fromkeys(found))


def run_materials_lofo(*, features: list[str]) -> dict[str, Any]:
    """Thin LOFO wrapper for pre-flight scripts (fixes.md)."""
    return MaterialsAdapter().run_experiment(
        MaterialsExperimentSpec(feature_subset=features, methodology="LOFO")
    )


def classify_materials_verdict(hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
    text = (hypothesis_text or "").lower()
    lofo = float(result.get("mean_r2", result.get("lofo_r2", 0.0)))
    fam_base = float(result.get("family_baseline_r2", 0.0))
    gap = float(result.get("lofo_gap", 0.0))
    p_perm = float(result.get("permutation_p", 1.0))
    p95 = result.get("label_shuffle_null_p95")
    label_p = result.get("label_shuffle_permutation_p")
    claims_generalization = any(
        m in text for m in ("cross-family", "cross-crystal", "generaliz", "lofo", "holdout", "crystal system")
    )
    if result.get("family_leakage_confirmed"):
        return (
            "confirmed",
            f"LOFO generalizes: lofo_r2={lofo:.3f} beats label-shuffle null p95={p95}",
            0.88,
        )
    if lofo > 0.05 and p95 is not None and lofo > float(p95) and label_p is not None and float(label_p) < 0.05:
        return "confirmed", f"LOFO={lofo:.3f} vs null p95={float(p95):.3f}", 0.85
    if claims_generalization and gap >= 0.25 and lofo < fam_base - 0.05:
        return "refuted", f"family leakage: LOFO={lofo:.3f} gap={gap:.2f}", 0.82
    if lofo < -0.20 and gap < 0.15:
        return "refuted", f"no cross-family signal LOFO={lofo:.3f}", 0.80
    if p_perm < 0.05 and lofo > fam_base - 0.2:
        return "inconclusive", f"in-sample signal LOFO={lofo:.3f} p_perm={p_perm:.3f} — need label-shuffle null", 0.55
    return "inconclusive", f"LOFO={lofo:.3f} gap={gap:.2f} p_perm={p_perm:.3f}", 0.50
