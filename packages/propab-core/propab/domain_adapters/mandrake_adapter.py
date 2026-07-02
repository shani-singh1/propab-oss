"""Mandrake Retroviral Wall domain adapter — verification & LOFO experiments (fixes.md P0/P1)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_KNOWN_FEATURES: tuple[str, ...] = (
    "t40_raw", "t45_raw", "t50_raw", "t55_raw", "t60_raw", "t65_raw", "t70_raw", "t75_raw", "t80_raw",
    "thermophilicity_num", "D1_D2_dist", "D2_D3_dist", "triad_best_rmsd", "triad_found_bin",
    "ramachandran_favoured", "g_factor_overall", "native_net_charge", "isoelectric_point",
    "pocket_salt_bridges", "thumb_surface_net_charge", "salt_per_res", "pocket_hbonds_per_res",
    "camsol_score", "camsol_mean_profile", "camsol_frac_good", "sasa_total", "sasa_avg",
    "sasa_high_pct", "whole_hydrophobic_fraction", "foldseek_best_TM", "foldseek_best_LDDT",
    "foldseek_best_fident", "foldseek_TM_HIV1", "foldseek_TM_MMLV", "foldseek_TM_MMLVPE",
    "foldseek_TM_Retron", "foldseek_TM_LTRRetrotransposon", "foldseek_TM_Group2Intron",
    "foldseek_TM_Telomerase", "has_dgr_motif", "qg_motif_found", "sp_motif_found",
    "yxdd_y_is_strand", "yxdd_d2_is_strand", "yxdd_hydrophobic_fraction",
)

_BIOLOGY_THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "thermal_stability": ("t40_raw", "t45_raw", "t50_raw", "t55_raw", "t60_raw", "t65_raw", "t70_raw", "t75_raw", "t80_raw", "thermal", "thermophilicity"),
    "catalytic_geometry": ("triad_best_rmsd", "d1_d2_dist", "d2_d3_dist", "ramachandran", "g_factor", "geometry", "yxdd"),
    "electrostatics": ("mean_pot", "net_charge", "isoelectric", "salt_bridge", "electrostatic", "pocket_hbond"),
    "fold_similarity": ("foldseek", "tm_score", "lddt", "fident"),
    "surface_properties": ("camsol", "sasa", "hydrophobic", "surface"),
    "motif_structure": ("dgr_motif", "qg_motif", "sp_motif", "motif", "strand"),
}

DEFAULT_TARGET = "pe_efficiency_pct"
DEFAULT_FAMILY = "rt_family"
DEFAULT_DATA_DIRS = (Path("/app/mandrake-data"), Path("mandrake-data"))


@dataclass
class MandrakeExperimentSpec:
    feature_subset: list[str]
    methodology: str
    target_column: str = DEFAULT_TARGET
    family_column: str = DEFAULT_FAMILY
    metric: str = "lofo_r2"
    baseline_model: str = "ridge"
    compare_features: list[str] = field(default_factory=list)
    mechanism_id: str | None = None

    def to_tool_params(self) -> dict[str, Any]:
        return {
            "feature_subset": self.feature_subset,
            "methodology": self.methodology,
            "target_column": self.target_column,
            "family_column": self.family_column,
            "metric": self.metric,
            "baseline_model": self.baseline_model,
            "compare_features": self.compare_features or None,
            "mechanism_id": self.mechanism_id,
        }

    @classmethod
    def from_hypothesis(cls, hypothesis: dict[str, Any], *, question: str = "") -> MandrakeExperimentSpec:
        text = str(hypothesis.get("text") or "")
        methodology = "LOFO"
        mechanism_id = hypothesis.get("mechanism_id")
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
                    mechanism_id = plan.get("mechanism_id") or mechanism_id
            except json.JSONDecodeError:
                pass
        if not features:
            features = resolve_mandrake_features(text)
        if not features:
            features = ["t70_raw", "t75_raw"]
        metric = "logo_mae" if "mae" in text.lower() else "lofo_r2"
        return cls(
            feature_subset=features[:6],
            methodology=methodology.upper(),
            metric=metric,
            baseline_model=model,
            compare_features=compare[:6],
            mechanism_id=str(mechanism_id) if mechanism_id else None,
        )


class MandrakeAdapter:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or self.resolve_data_dir()

    @staticmethod
    def resolve_data_dir() -> Path:
        for candidate in DEFAULT_DATA_DIRS:
            if (candidate / "handcrafted_features.csv").is_file():
                return candidate
        return DEFAULT_DATA_DIRS[0]

    def load_frame(self) -> pd.DataFrame:
        root = self.data_dir
        feat = pd.read_csv(root / "handcrafted_features.csv")
        labels = pd.read_csv(root / "rt_sequences.csv")[["rt_name", "active", "pe_efficiency_pct", "rt_family"]]
        df = feat.merge(labels, on="rt_name", how="inner")
        df[DEFAULT_FAMILY] = df[DEFAULT_FAMILY].astype(str).str.strip()
        df[DEFAULT_TARGET] = pd.to_numeric(df[DEFAULT_TARGET], errors="coerce").fillna(0.0)
        return df

    def run_experiment(self, spec: MandrakeExperimentSpec) -> dict[str, Any]:
        df = self.load_frame()
        cols = [c for c in spec.feature_subset if c in df.columns]
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
                compare_result = {"features": cmp_cols, "mean_r2": cmp_lofo, "delta_vs_primary": lofo - cmp_lofo}
        return {
            "mean_r2": lofo, "lofo_r2": lofo, "within_family_r2": within,
            "global_r2": _global_r2(X, y, model), "family_baseline_r2": fam_base,
            "lofo_gap": round(lofo_gap, 4), "surprise_score": round(lofo - fam_base, 4),
            "bootstrap_ci": bootstrap_ci, "permutation_p": permutation_p,
            "label_shuffle_permutation_p": round(label_shuffle_p, 4),
            "label_shuffle_null_p95": round(label_shuffle_p95, 6) if label_shuffle_p95 is not None else None,
            "family_breakdown": family_breakdown, "feature_subset": cols,
            "methodology": spec.methodology, "metric": spec.metric,
            "baseline_model": spec.baseline_model, "n_samples": len(y),
            "n_families": len(np.unique(families)), "compare_result": compare_result,
            "executed_code": _render_executed_code(spec, cols), "data_dir": str(self.data_dir),
            "verified": True, "metric_value": lofo, "p_value": permutation_p,
            "confidence_interval": bootstrap_ci,
        }

    def write_artifacts(self, out_dir: Path, spec: MandrakeExperimentSpec, result: dict[str, Any]) -> dict[str, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        (out_dir / "experiment_plans.json").write_text(json.dumps({
            "methodology": spec.methodology, "features": spec.feature_subset,
            "model": spec.baseline_model, "metric": spec.metric, "mechanism_id": spec.mechanism_id,
        }, indent=2), encoding="utf-8")
        paths["experiment_plans"] = str(out_dir / "experiment_plans.json")
        (out_dir / "verification_results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        paths["verification_results"] = str(out_dir / "verification_results.json")
        (out_dir / "family_breakdowns.json").write_text(json.dumps(result.get("family_breakdown") or {}, indent=2), encoding="utf-8")
        paths["family_breakdowns"] = str(out_dir / "family_breakdowns.json")
        (out_dir / "executed_code.py").write_text(str(result.get("executed_code") or ""), encoding="utf-8")
        paths["executed_code"] = str(out_dir / "executed_code.py")
        (out_dir / "bootstrap_distributions.json").write_text(json.dumps({
            "bootstrap_ci": result.get("bootstrap_ci"), "permutation_p": result.get("permutation_p"),
        }, indent=2), encoding="utf-8")
        paths["bootstrap_distributions"] = str(out_dir / "bootstrap_distributions.json")
        return paths


def is_mandrake_campaign(*, question: str = "", payload: dict[str, Any] | None = None) -> bool:
    q = (question or "").lower()
    if payload and str(payload.get("seed_source") or "") == "anomaly":
        return True
    if payload and str(payload.get("domain") or "") == "mandrake":
        return True
    markers = ("rt activity", "reverse transcriptase", "evolutionary family", "biophysical propert")
    return sum(1 for m in markers if m in q) >= 2


def resolve_mandrake_features(text: str) -> list[str]:
    found: list[str] = []
    for feat in _KNOWN_FEATURES:
        if feat in text:
            found.append(feat)
    for block in re.findall(r"\[([^\]]+)\]", text or ""):
        for part in re.split(r"[,|\s]+", block):
            p = part.strip().strip("'\"")
            if p in _KNOWN_FEATURES:
                found.append(p)
    return list(dict.fromkeys(found))


def infer_biology_theme(text: str, features: list[str] | None = None) -> str:
    blob = f"{text} {' '.join(features or [])}".lower()
    best, best_hits = "general", 0
    for theme, keys in _BIOLOGY_THEME_KEYWORDS.items():
        hits = sum(1 for k in keys if k in blob)
        if hits > best_hits:
            best_hits, best = hits, theme
    return best


def classify_mandrake_verdict(hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
    text = (hypothesis_text or "").lower()
    lofo = float(result.get("mean_r2", result.get("lofo_r2", 0.0)))
    fam_base = float(result.get("family_baseline_r2", 0.0))
    gap = float(result.get("lofo_gap", 0.0))
    p_perm = float(result.get("permutation_p", 1.0))
    ci = result.get("bootstrap_ci") or result.get("confidence_interval")
    ci_lo = float(ci[0]) if isinstance(ci, (list, tuple)) and len(ci) >= 2 else lofo - 0.15
    is_null = any(m in text for m in ("null hypothesis", "no falsifiable pattern", "group identity fully", "artifact", "vanish", "converge to zero"))
    claims_survival = any(m in text for m in ("cross-group", "independent of", "universal", "retain", "surviv", "lofo", "logo"))
    claims_collapse = any(m in text for m in ("collapse", "group-specific", "degrade under leave-one"))
    if is_null and not claims_survival:
        if gap >= 0.35 and lofo < fam_base - 0.1:
            return "confirmed", f"null: group-specific signal (gap={gap:.2f}, LOFO={lofo:.3f})", 0.88
        if lofo > -0.05 and p_perm < 0.05:
            return "refuted", f"null refuted: LOFO={lofo:.3f} p_perm={p_perm:.4f}", 0.85
        return "inconclusive", f"null: LOFO={lofo:.3f} gap={gap:.2f}", 0.55
    if claims_collapse and gap >= 0.30:
        return "confirmed", f"collapse confirmed gap={gap:.2f}", 0.86
    if lofo > fam_base - 0.25 or ci_lo > fam_base - 0.30:
        if p_perm < 0.10 or lofo > -0.15:
            return "confirmed", f"LOFO={lofo:.3f} vs baseline {fam_base:.3f}", 0.82
    if lofo < -0.35 and gap < 0.20:
        return "refuted", f"no cross-group signal LOFO={lofo:.3f}", 0.80
    cmp = result.get("compare_result")
    if isinstance(cmp, dict) and cmp.get("delta_vs_primary") is not None:
        delta = float(cmp["delta_vs_primary"])
        if delta > 0.03:
            return "confirmed", f"primary beats comparator by {delta:.3f}", 0.78
        if delta < -0.03:
            return "refuted", f"comparator beats primary by {-delta:.3f}", 0.78
    return "inconclusive", f"LOFO={lofo:.3f} gap={gap:.2f} p_perm={p_perm:.3f}", 0.50


def _make_model(name: str) -> Any:
    if (name or "ridge").lower() in {"linear", "linearregression", "lr"}:
        return Pipeline([("scale", StandardScaler()), ("reg", LinearRegression())])
    return Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])


def _clip_r2(v: float) -> float:
    return float(max(-1.0, min(1.0, v)))


def _family_baseline_r2(y: np.ndarray, families: np.ndarray) -> float:
    preds = np.zeros_like(y, dtype=float)
    for fam in np.unique(families):
        mask = families == fam
        preds[mask] = float(np.mean(y[mask]))
    return _clip_r2(float(r2_score(y, preds))) if len(y) > 1 else 0.0


def _within_family_r2(X: np.ndarray, y: np.ndarray, families: np.ndarray, model: Any) -> float:
    scores: list[float] = []
    for fam in np.unique(families):
        mask = families == fam
        if int(mask.sum()) < 4:
            continue
        m = clone(model)
        m.fit(X[mask], y[mask])
        scores.append(_clip_r2(float(r2_score(y[mask], m.predict(X[mask])))))
    return float(np.mean(scores)) if scores else 0.0


def _leave_one_family_out_r2(X: np.ndarray, y: np.ndarray, families: np.ndarray, model: Any) -> float:
    scores: list[float] = []
    for fam in np.unique(families):
        test, train = families == fam, families != fam
        if int(test.sum()) < 2 or int(train.sum()) < 4:
            continue
        m = clone(model)
        m.fit(X[train], y[train])
        scores.append(_clip_r2(float(r2_score(y[test], m.predict(X[test])))))
    return float(np.mean(scores)) if scores else 0.0


def _global_r2(X: np.ndarray, y: np.ndarray, model: Any) -> float:
    m = clone(model)
    m.fit(X, y)
    return _clip_r2(float(r2_score(y, m.predict(X))))


def _family_lofo_breakdown(X: np.ndarray, y: np.ndarray, families: np.ndarray, model: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for fam in np.unique(families):
        test, train = families == fam, families != fam
        if int(test.sum()) < 2 or int(train.sum()) < 4:
            continue
        m = clone(model)
        m.fit(X[train], y[train])
        pred = m.predict(X[test])
        out[str(fam)] = round(_clip_r2(float(r2_score(y[test], pred))), 4)
        out[f"{fam}_mae"] = round(float(mean_absolute_error(y[test], pred)), 4)
    return out


def _bootstrap_lofo_ci(X: np.ndarray, y: np.ndarray, families: np.ndarray, model: Any, *, n_boot: int = 200) -> list[float]:
    rng = np.random.default_rng(42)
    scores = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(_leave_one_family_out_r2(X[idx], y[idx], families[idx], model))
    if not scores:
        return [0.0, 0.0]
    return [round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)]


def _permutation_p_value(X: np.ndarray, y: np.ndarray, families: np.ndarray, model: Any, *, observed: float, n_perm: int = 300) -> float:
    rng = np.random.default_rng(7)
    count = 0
    for _ in range(n_perm):
        perm_y = y.copy()
        for fam in np.unique(families):
            mask = families == fam
            perm_y[mask] = rng.permutation(perm_y[mask])
        if _leave_one_family_out_r2(X, perm_y, families, model) >= observed:
            count += 1
    return round((count + 1) / (n_perm + 1), 4)


def _family_label_shuffle_null(
    X: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    model: Any,
    *,
    n_perm: int = 300,
    seed: int = 42,
) -> tuple[float, float, list[float]]:
    """Label-shuffle LOFO null (fixes.md artifact gate). Returns (observed, p_ge, null_samples)."""
    observed = _leave_one_family_out_r2(X, y, families, model)
    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(n_perm):
        perm_families = families.copy()
        rng.shuffle(perm_families)
        null.append(_leave_one_family_out_r2(X, y, perm_families, model))
    arr = np.asarray(null, dtype=float)
    p_ge = float((np.sum(arr >= observed) + 1) / (len(arr) + 1))
    p95 = float(np.percentile(arr, 95))
    return observed, p_ge, null


def _render_executed_code(spec: MandrakeExperimentSpec, cols: list[str]) -> str:
    return (
        f"# Mandrake LOFO experiment\nfeatures = {cols!r}\n"
        f"methodology = {spec.methodology!r}\nmodel = {spec.baseline_model!r}\n"
    )
