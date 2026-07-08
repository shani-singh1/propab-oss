"""Tests for the epitope domain plugin (leave-one-MHC-allele-out + within-allele null)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propab import config
from propab.domain_modules.epitope.adapter import ALLELES, EpitopeAdapter, EpitopeExperimentSpec
from propab.domain_modules.epitope.plugin import EpitopePlugin
from propab.domain_modules.epitope.routing_inspector import inspect_corpus
from propab.domain_modules.epitope.verifier import classify_epitope_verdict, run_epitope_experiment
from propab.domain_modules.registry import get_domain_plugin, resolve_domain_plugin

FEATS = ["anchorC_hydrophobicity", "anchor2_hydrophobicity", "net_charge", "mean_hydrophobicity"]


@pytest.fixture
def tmp_epitope_data(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))


def _planted_frame(seed: int = 0, n_per: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for allele in ALLELES:
        offset = rng.normal(0, 0.3)
        for _ in range(n_per):
            f = rng.normal(0, 1, size=4)
            y = float(0.9 * f[0] + 0.6 * f[1] - 0.5 * abs(f[2]) + offset + rng.normal(0, 0.4))
            rows.append({
                "allele": allele, "binding_score": y,
                "anchorC_hydrophobicity": f[0], "anchor2_hydrophobicity": f[1],
                "net_charge": f[2], "mean_hydrophobicity": f[3],
            })
    return pd.DataFrame(rows)


def _noise_frame(seed: int = 0, n_per: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for allele in ALLELES:
        for _ in range(n_per):
            rows.append({
                "allele": allele, "binding_score": float(rng.normal()),
                "anchorC_hydrophobicity": float(rng.normal()), "anchor2_hydrophobicity": float(rng.normal()),
                "net_charge": float(rng.normal()), "mean_hydrophobicity": float(rng.normal()),
            })
    return pd.DataFrame(rows)


def test_registered_and_routes():
    assert get_domain_plugin("epitope") is not None
    routed = resolve_domain_plugin(
        question="Does a peptide-MHC binding rule survive leave-one-allele-out across HLA alleles? [domain_profile:epitope]"
    )
    assert routed is not None and routed.domain_id == "epitope"


def test_objective_spec_is_non_ml():
    obj = EpitopePlugin().objective_spec()
    assert obj["is_ml"] is False
    assert obj["metric_name"] == "laoo_r2"
    assert obj["baseline_kind"] == "measured"


def test_preflight_passes(tmp_epitope_data):
    r = EpitopePlugin().preflight()
    assert r.passed, r.reason


def test_planted_signal_confirms(monkeypatch):
    df = _planted_frame(seed=3)
    monkeypatch.setattr(EpitopeAdapter, "load_frame", lambda self: df)
    result = run_epitope_experiment(EpitopeExperimentSpec(feature_subset=FEATS))
    assert result["lofo_r2"] > 0.12
    assert result["label_shuffle_null_p"] < 0.05
    assert result["verified_true_steps"] == 1
    verdict, _, conf = classify_epitope_verdict("planted binding rule", result)
    assert verdict == "confirmed"
    assert conf > 0.8


def test_pure_noise_does_not_confirm(monkeypatch):
    df = _noise_frame(seed=1)
    monkeypatch.setattr(EpitopeAdapter, "load_frame", lambda self: df)
    result = run_epitope_experiment(EpitopeExperimentSpec(feature_subset=FEATS))
    p = result["label_shuffle_null_p"]
    assert not np.isnan(p) and p >= 0.05
    assert result["verified_true_steps"] == 0
    assert classify_epitope_verdict("pure noise", result)[0] != "confirmed"


def test_shuffled_target_not_confirmed(monkeypatch):
    df = _planted_frame(seed=2)
    rng = np.random.default_rng(7)
    df["binding_score"] = rng.permutation(df["binding_score"].to_numpy())
    monkeypatch.setattr(EpitopeAdapter, "load_frame", lambda self: df)
    result = run_epitope_experiment(EpitopeExperimentSpec(feature_subset=FEATS))
    assert classify_epitope_verdict("shuffled control", result)[0] != "confirmed"


def test_confirm_gate_refuses_without_null_stats():
    assert classify_epitope_verdict("suspicious", {"lofo_r2": 0.9})[0] != "confirmed"


def test_routing_corpus_zero_mismatches(tmp_epitope_data):
    report = inspect_corpus()
    assert report["routing_mismatches"] == 0
    assert report["total"] >= 20
