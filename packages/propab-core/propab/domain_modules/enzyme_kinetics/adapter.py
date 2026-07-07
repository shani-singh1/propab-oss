"""Real enzyme kinetics subset (DLKcat / BRENDA + SABIO-RK derived).

The frame is built from the DLKcat training compilation (Li et al., *Nature
Catalysis* 2022; SysBioChalmers/DLKcat), which aggregates **real, measured**
turnover numbers (kcat) from BRENDA and SABIO-RK together with each enzyme's
EC number, organism, substrate and protein sequence. From the real sequence we
derive physicochemical features (molecular weight, length, hydropathy, residue
composition); the target ``log_kcat`` is the log10 of the real measured kcat.

Data is grouped by top-level EC class (EC1..EC6) so the verifier's
leave-one-EC-class-out (LOFO) + EC-label-shuffle null runs on real families.

Provenance / license: see ``data/enzyme_kinetics/README.md``. If the source
cannot be reached and no real cache exists, a clearly-labelled synthetic
fallback is generated so preflight never hard-fails; ``dataset_is_synthetic()``
reports which one is actually on disk and the plugin's ``uses_synthetic_data()``
reads it.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from propab.config import settings

EC_CLASSES = ("EC1", "EC2", "EC3", "EC4", "EC5", "EC6")
N_ENZYMES = 400  # synthetic-fallback size only
RANDOM_SEED = 42

# Real DLKcat kcat compilation (BRENDA + SABIO-RK derived).
DLKCAT_URL = (
    "https://raw.githubusercontent.com/SysBioChalmers/DLKcat/master/"
    "DeeplearningApproach/Data/database/Kcat_combination_0918.json"
)
# Balanced per-EC cap keeps the cached CSV small and LOFO fast while staying real.
MAX_PER_EC = 600
MIN_SEQ_LEN = 20

KNOWN_FEATURES: tuple[str, ...] = (
    "molecular_weight",
    "sequence_length",
    "gravy_hydropathy",
    "frac_charged",
    "frac_hydrophobic",
    "frac_aromatic",
    "frac_polar",
    "frac_glycine",
    "frac_proline",
)

# Kyte-Doolittle hydropathy index.
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Average residue masses (Da) for molecular-weight estimate from sequence.
_MW = {
    "A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14, "Q": 128.13,
    "E": 129.12, "G": 57.05, "H": 137.14, "I": 113.16, "L": 113.16, "K": 128.17,
    "M": 131.19, "F": 147.18, "P": 97.12, "S": 87.08, "T": 101.10, "W": 186.21,
    "Y": 163.18, "V": 99.13,
}
_AA = "ACDEFGHIKLMNPQRSTVWY"


def cache_dir() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "enzyme_kinetics"
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path() -> Path:
    return cache_dir() / "brenda_subset_v1.csv"


def meta_path() -> Path:
    return cache_dir() / "brenda_subset_v1.meta.json"


def dataset_is_synthetic() -> bool:
    """True when the on-disk cache was produced by the synthetic fallback."""
    mp = meta_path()
    if not mp.is_file():
        return True
    try:
        return bool(json.loads(mp.read_text(encoding="utf-8")).get("synthetic", True))
    except Exception:  # noqa: BLE001
        return True


def _ec_top(ec: Any) -> str | None:
    ec = str(ec or "").strip()
    if "." not in ec:
        return None
    head = ec.split(".")[0]
    return f"EC{head}" if head in "123456" else None


def _sequence_features(seq: str) -> dict[str, float] | None:
    seq = "".join(ch for ch in seq.upper() if ch in _AA)
    if len(seq) < MIN_SEQ_LEN:
        return None
    length = len(seq)
    comp = {a: seq.count(a) / length for a in _AA}
    return {
        "sequence_length": float(length),
        "molecular_weight": float(sum(_MW[a] * seq.count(a) for a in _AA) + 18.02),
        "gravy_hydropathy": float(sum(_KD[a] * seq.count(a) for a in _AA) / length),
        "frac_charged": float(comp["D"] + comp["E"] + comp["K"] + comp["R"]),
        "frac_hydrophobic": float(comp["A"] + comp["I"] + comp["L"] + comp["M"] + comp["F"] + comp["V"]),
        "frac_aromatic": float(comp["F"] + comp["W"] + comp["Y"]),
        "frac_polar": float(comp["S"] + comp["T"] + comp["N"] + comp["Q"]),
        "frac_glycine": float(comp["G"]),
        "frac_proline": float(comp["P"]),
    }


def _build_real_frame(raw_records: list[dict[str, Any]]) -> pd.DataFrame:
    """Parse DLKcat records into a real, EC-keyed, sequence-featured frame."""
    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(raw_records):
        ec = _ec_top(rec.get("ECNumber"))
        if ec is None:
            continue
        if "s^(-1)" not in str(rec.get("Unit") or ""):
            continue
        try:
            kcat = float(rec.get("Value"))
        except (TypeError, ValueError):
            continue
        if not (kcat > 0):
            continue
        feats = _sequence_features(str(rec.get("Sequence") or ""))
        if feats is None:
            continue
        rows.append({
            "enzyme_id": f"DLK{i:06d}",
            "ec_class": ec,
            "ec_number": str(rec.get("ECNumber")).strip(),
            "organism": str(rec.get("Organism") or "").strip(),
            "substrate": str(rec.get("Substrate") or "").strip(),
            "kcat": kcat,
            "log_kcat": float(np.log10(kcat)),
            **feats,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(
        subset=["ec_number", "organism", "substrate", "kcat", "sequence_length"]
    )
    # Balance per EC class (deterministic) so no family dominates LOFO. Use a
    # per-group cumulative rank rather than groupby.apply (which can drop the
    # grouping column across pandas versions).
    df = df.sort_values(["ec_class", "enzyme_id"]).reset_index(drop=True)
    within = df.groupby("ec_class").cumcount()
    df = df.loc[within < MAX_PER_EC].reset_index(drop=True)
    return df


def _write_readme(*, synthetic: bool, source: str) -> None:
    kind = "SYNTHETIC FALLBACK" if synthetic else "REAL DATA"
    cache_dir().joinpath("README.md").write_text(
        f"# enzyme_kinetics cache — {kind}\n\n"
        f"Source: {source}\n\n"
        "Full provenance, license and column documentation:\n"
        "`packages/propab-core/propab/domain_modules/enzyme_kinetics/DATA_PROVENANCE.md`\n\n"
        "This directory is git-ignored; the adapter regenerates it from the source "
        "on first use. Delete the CSV + meta to force a refresh.\n",
        encoding="utf-8",
    )


def _fetch_dlkcat(timeout: float = 120.0) -> list[dict[str, Any]]:
    req = urllib.request.Request(DLKCAT_URL, headers={"User-Agent": "propab-enzyme-adapter/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted GitHub raw)
        return json.loads(resp.read())


def _synthetic_frame() -> pd.DataFrame:
    """Labelled synthetic fallback (used only if real data is unavailable)."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for i in range(N_ENZYMES):
        ec = EC_CLASSES[i % len(EC_CLASSES)]
        base_kcat = rng.lognormal(mean=1.5 + 0.3 * (i % 6), sigma=0.35)
        family_shift = {"EC1": 0.2, "EC2": -0.1, "EC3": 0.0, "EC4": 0.15, "EC5": -0.2, "EC6": 0.05}[ec]
        kcat = max(0.01, base_kcat + family_shift + rng.normal(0, 0.1))
        seq_len = int(rng.integers(200, 800))
        rows.append({
            "enzyme_id": f"ENZ{i:05d}",
            "ec_class": ec,
            "ec_number": f"{ec[-1]}.-.-.-",
            "organism": "synthetic",
            "substrate": "synthetic",
            "kcat": float(kcat),
            "log_kcat": float(np.log10(max(kcat, 1e-6))),
            "sequence_length": float(seq_len),
            "molecular_weight": float(rng.uniform(25000, 120000)),
            "gravy_hydropathy": float(rng.uniform(-1.0, 0.5)),
            "frac_charged": float(rng.uniform(0.15, 0.30)),
            "frac_hydrophobic": float(rng.uniform(0.30, 0.45)),
            "frac_aromatic": float(rng.uniform(0.05, 0.12)),
            "frac_polar": float(rng.uniform(0.15, 0.30)),
            "frac_glycine": float(rng.uniform(0.05, 0.10)),
            "frac_proline": float(rng.uniform(0.03, 0.08)),
        })
    return pd.DataFrame(rows)


@dataclass
class EnzymeExperimentSpec:
    feature_subset: list[str]
    target_column: str = "log_kcat"
    group_column: str = "ec_class"

    @classmethod
    def from_hypothesis(cls, hypothesis: dict[str, Any]) -> EnzymeExperimentSpec:
        text = str(hypothesis.get("text") or hypothesis.get("statement") or "").lower()
        features = list(hypothesis.get("feature_subset") or [])
        if not features:
            if "aromatic" in text or "hydrophob" in text or "composition" in text:
                features = ["frac_aromatic", "frac_hydrophobic", "frac_charged", "molecular_weight"]
            elif "hydropath" in text or "gravy" in text or "thermal" in text or "temperature" in text:
                features = ["gravy_hydropathy", "frac_charged", "molecular_weight"]
            elif "length" in text or "sequence" in text or "size" in text or "molecular weight" in text:
                features = ["sequence_length", "molecular_weight", "frac_charged"]
            else:
                features = ["molecular_weight", "sequence_length", "frac_charged", "frac_aromatic"]
        # Only ``log_kcat`` is a real measured target in this dataset.
        return cls(feature_subset=features[:4], target_column="log_kcat")


class EnzymeKineticsAdapter:
    def _write(self, df: pd.DataFrame, *, synthetic: bool, source: str) -> Path:
        path = cache_path()
        df.to_csv(path, index=False)
        meta = {
            "synthetic": synthetic,
            "source": source,
            "n_enzymes": int(len(df)),
            "ec_classes": sorted(df["ec_class"].unique().tolist()) if not df.empty else list(EC_CLASSES),
            "per_ec_counts": (
                df["ec_class"].value_counts().sort_index().to_dict() if not df.empty else {}
            ),
            "target": "log_kcat",
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
            records = _fetch_dlkcat()
            df = _build_real_frame(records)
            if not df.empty and df["ec_class"].nunique() >= 3 and len(df) >= 40:
                return self._write(
                    df,
                    synthetic=False,
                    source=(
                        "DLKcat (Li et al., Nature Catalysis 2022; SysBioChalmers/DLKcat) "
                        "kcat compilation derived from BRENDA + SABIO-RK"
                    ),
                )
        except Exception:  # noqa: BLE001 (network/parse failures fall back)
            pass
        return self._write(_synthetic_frame(), synthetic=True, source="synthetic-fallback")

    def load_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.ensure_cache())
