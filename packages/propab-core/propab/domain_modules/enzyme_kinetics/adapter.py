"""Synthetic BRENDA-style enzyme kinetics subset."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from propab.config import settings

EC_CLASSES = ("EC1", "EC2", "EC3", "EC4", "EC5", "EC6")
N_ENZYMES = 400
RANDOM_SEED = 42

KNOWN_FEATURES: tuple[str, ...] = (
    "log_kcat",
    "log_km",
    "molecular_weight",
    "temperature_opt",
    "ph_opt",
    "sequence_length",
)


def cache_dir() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "enzyme_kinetics"
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path() -> Path:
    return cache_dir() / "brenda_subset_v1.csv"


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for i in range(N_ENZYMES):
        ec = EC_CLASSES[i % len(EC_CLASSES)]
        base_kcat = rng.lognormal(mean=1.5 + 0.3 * (i % 6), sigma=0.35)
        family_shift = {"EC1": 0.2, "EC2": -0.1, "EC3": 0.0, "EC4": 0.15, "EC5": -0.2, "EC6": 0.05}[ec]
        kcat = max(0.01, base_kcat + family_shift + rng.normal(0, 0.1))
        km = max(0.001, rng.lognormal(mean=-1.0, sigma=0.4))
        mw = float(rng.uniform(25000, 120000))
        rows.append({
            "enzyme_id": f"ENZ{i:05d}",
            "ec_class": ec,
            "log_kcat": float(np.log(kcat)),
            "log_km": float(np.log(km)),
            "molecular_weight": mw,
            "temperature_opt": float(rng.uniform(25, 75)),
            "ph_opt": float(rng.uniform(5.5, 9.0)),
            "sequence_length": int(rng.integers(200, 800)),
            "kcat": float(kcat),
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
            if "km" in text:
                features = ["log_km", "molecular_weight", "ph_opt"]
            elif "temperature" in text or "thermal" in text:
                features = ["temperature_opt", "ph_opt", "molecular_weight"]
            else:
                features = ["log_km", "molecular_weight", "sequence_length"]
        target = "log_km" if "km" in text and "kcat" not in text else "log_kcat"
        return cls(feature_subset=features[:4], target_column=target)


class EnzymeKineticsAdapter:
    def ensure_cache(self) -> Path:
        path = cache_path()
        if path.is_file():
            return path
        df = _synthetic_frame()
        df.to_csv(path, index=False)
        cache_dir().joinpath("brenda_subset_v1.meta.json").write_text(
            json.dumps({"n_enzymes": N_ENZYMES, "ec_classes": list(EC_CLASSES), "synthetic": True}, indent=2),
            encoding="utf-8",
        )
        return path

    def load_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.ensure_cache())
