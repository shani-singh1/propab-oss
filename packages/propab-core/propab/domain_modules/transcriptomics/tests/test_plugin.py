"""Tests for the transcriptomics domain plugin (leave-one-condition-out + within-condition null)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propab import config
from propab.domain_modules.registry import get_domain_plugin, resolve_domain_plugin
from propab.domain_modules.transcriptomics.adapter import (
    CONDITIONS,
    TranscriptomicsAdapter,
    TranscriptomicsExperimentSpec,
)
from propab.domain_modules.transcriptomics.plugin import TranscriptomicsPlugin
from propab.domain_modules.transcriptomics.routing_inspector import inspect_corpus
from propab.domain_modules.transcriptomics.verifier import (
    classify_transcriptomics_verdict,
    run_transcriptomics_experiment,
)

FEATS = ["tf_motif_count", "chromatin_accessibility", "cpg_ratio", "tata_score"]


@pytest.fixture
def tmp_transcriptomics_data(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))


def _planted_frame(seed: int = 0, n_per: int = 110) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cond in CONDITIONS:
        offset = rng.normal(0, 0.35)
        for _ in range(n_per):
            f = rng.normal(0, 1, size=4)
            y = float(0.9 * f[0] + 0.8 * f[1] + 0.5 * f[2] + offset + rng.normal(0, 0.5))
            rows.append({
                "condition": cond, "log2_fold_change": y,
                "tf_motif_count": f[0], "chromatin_accessibility": f[1],
                "cpg_ratio": f[2], "tata_score": f[3],
            })
    return pd.DataFrame(rows)


def _noise_frame(seed: int = 0, n_per: int = 110) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cond in CONDITIONS:
        for _ in range(n_per):
            rows.append({
                "condition": cond, "log2_fold_change": float(rng.normal()),
                "tf_motif_count": float(rng.normal()), "chromatin_accessibility": float(rng.normal()),
                "cpg_ratio": float(rng.normal()), "tata_score": float(rng.normal()),
            })
    return pd.DataFrame(rows)


def test_registered_and_routes():
    assert get_domain_plugin("transcriptomics") is not None
    routed = resolve_domain_plugin(
        question="Does a promoter-feature gene-regulation rule survive leave-one-condition-out across perturbations? [domain_profile:transcriptomics]"
    )
    assert routed is not None and routed.domain_id == "transcriptomics"


def test_objective_spec_is_non_ml():
    obj = TranscriptomicsPlugin().objective_spec()
    assert obj["is_ml"] is False
    assert obj["metric_name"] == "loco_r2"
    assert obj["baseline_kind"] == "measured"


def test_preflight_passes(tmp_transcriptomics_data):
    r = TranscriptomicsPlugin().preflight()
    assert r.passed, r.reason


def test_planted_signal_confirms(monkeypatch):
    df = _planted_frame(seed=3)
    monkeypatch.setattr(TranscriptomicsAdapter, "load_frame", lambda self: df)
    result = run_transcriptomics_experiment(TranscriptomicsExperimentSpec(feature_subset=FEATS))
    assert result["lofo_r2"] > 0.12
    assert result["label_shuffle_null_p"] < 0.05
    assert result["verified_true_steps"] == 1
    verdict, _, conf = classify_transcriptomics_verdict("planted regulatory rule", result)
    assert verdict == "confirmed"
    assert conf > 0.8


def test_pure_noise_does_not_confirm(monkeypatch):
    df = _noise_frame(seed=1)
    monkeypatch.setattr(TranscriptomicsAdapter, "load_frame", lambda self: df)
    result = run_transcriptomics_experiment(TranscriptomicsExperimentSpec(feature_subset=FEATS))
    p = result["label_shuffle_null_p"]
    assert not np.isnan(p) and p >= 0.05
    assert result["verified_true_steps"] == 0
    assert classify_transcriptomics_verdict("pure noise", result)[0] != "confirmed"


def test_shuffled_target_not_confirmed(monkeypatch):
    df = _planted_frame(seed=2)
    rng = np.random.default_rng(7)
    df["log2_fold_change"] = rng.permutation(df["log2_fold_change"].to_numpy())
    monkeypatch.setattr(TranscriptomicsAdapter, "load_frame", lambda self: df)
    result = run_transcriptomics_experiment(TranscriptomicsExperimentSpec(feature_subset=FEATS))
    assert classify_transcriptomics_verdict("shuffled control", result)[0] != "confirmed"


def test_confirm_gate_refuses_without_null_stats():
    assert classify_transcriptomics_verdict("suspicious", {"lofo_r2": 0.9})[0] != "confirmed"


def test_routing_corpus_zero_mismatches(tmp_transcriptomics_data):
    report = inspect_corpus()
    assert report["routing_mismatches"] == 0
    assert report["total"] >= 20
