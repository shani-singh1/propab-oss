"""Tests for generic anomaly engine + Mandrake demo adapter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propab.anomaly_engine.anomaly_detector import detect_anomalies, summarize_anomalies
from propab.anomaly_engine.anomaly_objects import AnomalyObject
from propab.anomaly_engine.artifacts import (
    read_anomalies,
    read_mechanisms,
    write_anomalies,
    write_mechanisms,
    write_sweep_parquet,
)
from propab.anomaly_engine.detector_config import ANOMALY_BUCKETS, DetectorConfig, default_bucket_slots
from propab.anomaly_engine.mechanism_inducer import induce_mechanisms_sync
from propab.anomaly_engine.subset_generator import generate_feature_subsets
from propab.anomaly_engine.sweep_engine import SweepConfig, run_sweep
from propab.seed_source import SeedSource

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_frame(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    families = rng.choice(["A", "B", "C"], size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 * x1 + rng.normal(scale=0.3, size=n)
    for fam in np.unique(families):
        mask = families == fam
        y[mask] += {"A": 2.0, "B": -1.0, "C": 0.5}[fam]
    return pd.DataFrame({
        "group": families,
        "target": y,
        "feat_a": x1,
        "feat_b": x2,
        "feat_c": x1 + rng.normal(scale=0.1, size=n),
    })


def test_feature_subsets_max_size():
    cols = ["a", "b", "c", "d"]
    subs = generate_feature_subsets(cols, max_subset_size=3)
    assert len(subs) == 4 + 6 + 4
    assert max(len(s) for s in subs) == 3


def test_synthetic_sweep_and_detect():
    df = _synthetic_frame()
    config = SweepConfig(
        target_column="target",
        family_column="group",
        feature_columns=["feat_a", "feat_b", "feat_c"],
        max_subset_size=2,
        model_names=("Ridge",),
    )
    results = run_sweep(df, config)
    assert len(results) > 0

    anomalies = detect_anomalies(results, DetectorConfig(top_k=10))
    assert 1 <= len(anomalies) <= 10
    assert all(isinstance(a, AnomalyObject) for a in anomalies)


def test_mechanism_inducer_and_artifacts(tmp_path):
    df = _synthetic_frame()
    config = SweepConfig(
        target_column="target",
        family_column="group",
        feature_columns=["feat_a", "feat_b", "feat_c"],
        max_subset_size=2,
        model_names=("Ridge",),
    )
    results = run_sweep(df, config)
    anomalies = detect_anomalies(results, DetectorConfig(top_k=5))
    mechanisms = induce_mechanisms_sync(anomalies)
    assert 1 <= len(mechanisms) <= 5

    write_sweep_parquet(results, tmp_path / "sweep_results.parquet")
    write_anomalies(anomalies, tmp_path / "anomaly_objects.json")
    write_mechanisms(mechanisms, tmp_path / "mechanism_objects.json")

    assert read_mechanisms(tmp_path / "mechanism_objects.json")
    assert isinstance(read_anomalies(tmp_path / "anomaly_objects.json"), list)


def test_seed_source_enum():
    assert SeedSource.ANOMALY.value == "anomaly"


def test_default_bucket_slots():
    slots = default_bucket_slots(12)
    assert sum(slots.values()) == 12
    assert slots["survivor"] >= 1
    assert slots["collapse"] >= 1


def test_bucket_selection_respects_group_cap():
    from propab.anomaly_engine.sweep_engine import SweepResult

    rows: list[SweepResult] = []
    for i in range(12):
        rows.append(SweepResult(
            feature_subset=[f"t{i}_raw"],
            model_name="Ridge",
            within_family_r2=0.5,
            leave_one_family_out_r2=0.1 + i * 0.01,
            global_r2=0.4,
            family_baseline_r2=0.2,
            surprise_score=0.05,
            metadata={"lofo_gap": 0.4, "lofo_family_std": 0.2, "per_family_lofo": {"A": 0.2, "B": -0.1}},
        ))
    for i in range(8):
        rows.append(SweepResult(
            feature_subset=[f"foldseek_{i}"],
            model_name="Ridge",
            within_family_r2=0.6,
            leave_one_family_out_r2=-0.3,
            global_r2=0.5,
            family_baseline_r2=0.2,
            surprise_score=-0.4,
            metadata={"lofo_gap": 0.9, "lofo_family_std": 0.05, "per_family_lofo": {"A": -0.3, "B": -0.3}},
        ))
    groups = {
        "thermal": [f"t{i}_raw" for i in range(12)],
        "foldseek": [f"foldseek_{i}" for i in range(8)],
    }
    cfg = DetectorConfig(
        top_k=10,
        bucket_slots={"survivor": 3, "collapse": 3, "cross_family": 2, "threshold": 2},
        feature_groups=groups,
        max_group_fraction=0.4,
    )
    anomalies = detect_anomalies(rows, cfg)
    summary = summarize_anomalies(anomalies)
    assert summary.get("by_bucket")
    group_counts = summary.get("by_feature_group") or {}
    assert max(group_counts.values(), default=0) <= cfg.group_cap()


def test_mechanism_inducer_uses_buckets():
    from propab.anomaly_engine.mechanism_inducer import _deterministic_mechanisms

    anomalies = [
        AnomalyObject(
            feature_subset=["t70_raw"],
            metric_name="leave_one_family_out_r2",
            expected_score=0.2,
            observed_score=0.1,
            surprise_score=0.05,
            anomaly_type="family_violation",
            affected_families=[],
            neighboring_subsets=[],
            metadata={"bucket": "survivor", "lofo_gap": 0.1, "within_family_r2": 0.3},
        ),
        AnomalyObject(
            feature_subset=["foldseek_best_TM"],
            metric_name="leave_one_family_out_r2",
            expected_score=0.2,
            observed_score=-0.4,
            surprise_score=-0.5,
            anomaly_type="prediction_failure",
            affected_families=[],
            neighboring_subsets=[],
            metadata={"bucket": "collapse", "lofo_gap": 0.8, "within_family_r2": 0.5},
        ),
    ]
    mechs = _deterministic_mechanisms(anomalies)
    assert len(mechs) >= 2
    texts = " ".join(m.explanation for m in mechs).lower()
    assert "foldseek" in texts or "collapse" in texts or "within" in texts


# --- Mandrake adapter (optional local data) ---

try:
    from demo.mandrake.domain import FEATURE_GROUPS, load_frame, repo_data_dir, sweep_feature_columns

    MANDRAKE = repo_data_dir(ROOT)
    _HAS_MANDRAKE = (MANDRAKE / "handcrafted_features.csv").is_file()
except ImportError:
    _HAS_MANDRAKE = False


@pytest.mark.skipif(not _HAS_MANDRAKE, reason="no mandrake-data")
def test_load_mandrake_frame():
    df = load_frame(MANDRAKE)
    assert len(df) >= 50
    assert "pe_efficiency_pct" in df.columns
    assert "rt_family" in df.columns


@pytest.mark.skipif(not _HAS_MANDRAKE, reason="no mandrake-data")
def test_mandrake_sweep_and_detect():
    df = load_frame(MANDRAKE)
    features = sweep_feature_columns()
    config = SweepConfig(
        target_column="pe_efficiency_pct",
        family_column="rt_family",
        feature_columns=features,
        feature_groups=FEATURE_GROUPS,
        max_subset_size=2,
        model_names=("Ridge",),
    )
    results = run_sweep(df, config)
    assert len(results) > 0
    anomalies = detect_anomalies(results, DetectorConfig(top_k=15))
    assert 1 <= len(anomalies) <= 20
