"""Tests for the QSAR domain plugin (leave-one-scaffold-out + within-scaffold null)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propab import config
from propab.domain_modules.qsar import verifier as QV
from propab.domain_modules.qsar.adapter import QSARAdapter, QSARExperimentSpec, SCAFFOLDS
from propab.domain_modules.qsar.plugin import QSARPlugin
from propab.domain_modules.qsar.routing_inspector import inspect_corpus
from propab.domain_modules.qsar.verifier import classify_qsar_verdict, run_qsar_experiment
from propab.domain_modules.registry import get_domain_plugin, resolve_domain_plugin


@pytest.fixture
def tmp_qsar_data(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))


FEATS = ["mol_weight", "clogp", "tpsa", "num_aromatic_rings"]


def _planted_frame(seed: int = 0, n_per: int = 90) -> pd.DataFrame:
    """Global descriptor->potency law shared across scaffolds + small offsets."""
    rng = np.random.default_rng(seed)
    rows = []
    for si, scaf in enumerate(SCAFFOLDS):
        offset = rng.normal(0, 0.25)
        for _ in range(n_per):
            f = rng.normal(0, 1, size=4)
            y = float(1.2 * f[0] - 0.7 * f[1] + 0.5 * f[2] + offset + rng.normal(0, 0.4))
            rows.append({
                "scaffold": scaf, "pactivity": y,
                "mol_weight": f[0], "clogp": f[1], "tpsa": f[2], "num_aromatic_rings": f[3],
            })
    return pd.DataFrame(rows)


def _noise_frame(seed: int = 0, n_per: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for si, scaf in enumerate(SCAFFOLDS):
        for _ in range(n_per):
            rows.append({
                "scaffold": scaf, "pactivity": float(rng.normal()),
                "mol_weight": float(rng.normal()), "clogp": float(rng.normal()),
                "tpsa": float(rng.normal()), "num_aromatic_rings": float(rng.normal()),
            })
    return pd.DataFrame(rows)


def test_registered_and_routes():
    assert get_domain_plugin("qsar") is not None
    routed = resolve_domain_plugin(
        question="Does a QSAR structure-activity relationship for pIC50 survive leave-one-scaffold-out? [domain_profile:qsar]"
    )
    assert routed is not None and routed.domain_id == "qsar"


def test_objective_spec_is_non_ml():
    obj = QSARPlugin().objective_spec()
    assert obj["is_ml"] is False
    assert obj["metric_name"] == "loso_r2"
    assert obj["baseline_kind"] == "measured"


def test_preflight_passes(tmp_qsar_data):
    r = QSARPlugin().preflight()
    assert r.passed, r.reason


def test_planted_signal_confirms(monkeypatch):
    df = _planted_frame(seed=3)
    monkeypatch.setattr(QSARAdapter, "load_frame", lambda self: df)
    result = run_qsar_experiment(QSARExperimentSpec(feature_subset=FEATS))
    assert result["lofo_r2"] > 0.12
    assert result["label_shuffle_null_p"] < 0.05
    assert result["verified_true_steps"] == 1
    verdict, _, conf = classify_qsar_verdict("planted SAR", result)
    assert verdict == "confirmed"
    assert conf > 0.8


def test_pure_noise_does_not_confirm(monkeypatch):
    df = _noise_frame(seed=1)
    monkeypatch.setattr(QSARAdapter, "load_frame", lambda self: df)
    result = run_qsar_experiment(QSARExperimentSpec(feature_subset=FEATS))
    p = result["label_shuffle_null_p"]
    assert not np.isnan(p) and p >= 0.05
    assert result["verified_true_steps"] == 0
    verdict, _, _ = classify_qsar_verdict("pure noise", result)
    assert verdict != "confirmed"


def test_shuffled_target_not_confirmed(monkeypatch):
    """Breaking the descriptor->activity pairing must not confirm."""
    df = _planted_frame(seed=2)
    rng = np.random.default_rng(7)
    df["pactivity"] = rng.permutation(df["pactivity"].to_numpy())
    monkeypatch.setattr(QSARAdapter, "load_frame", lambda self: df)
    result = run_qsar_experiment(QSARExperimentSpec(feature_subset=FEATS))
    verdict, _, _ = classify_qsar_verdict("shuffled control", result)
    assert verdict != "confirmed"


def test_confirm_gate_refuses_without_null_stats():
    verdict, _, _ = classify_qsar_verdict("suspicious high R2", {"lofo_r2": 0.9})
    assert verdict != "confirmed"


def test_routing_corpus_zero_mismatches(tmp_qsar_data):
    report = inspect_corpus()
    assert report["routing_mismatches"] == 0
    assert report["total"] >= 20
