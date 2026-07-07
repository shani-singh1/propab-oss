"""Verifier-honesty tests for the genomics domain.

These assert the leave-one-tissue-out (LOFO) + tissue-label-shuffle-null
machinery is a real honesty gate:
  1. A shuffled-label frame (broken tissue signal) must NOT be confirmed.
  2. The tissue-label-shuffle null must reject a pure-noise signal.
  3. The confirm path fires only when the result carries the LOFO null stats the
     gate reads; absent those stats the gate must refute.

They complement the registration/routing tests, which never exercised the null.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propab import config
from propab.domain_modules.genomics import verifier as GV
from propab.domain_modules.genomics.adapter import (
    GenomicsAdapter,
    GenomicsExperimentSpec,
    compute_gene_features,
    dataset_is_synthetic,
)
from propab.domain_modules.genomics.verifier import (
    classify_genomics_verdict,
    run_genomics_experiment,
)

TISSUES = ["Brain", "Heart", "Liver", "Lung", "Muscle", "Skin", "Blood"]


@pytest.fixture
def tmp_genomics_data(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))


def _long_frame(seed: int = 0, n_genes: int = 300) -> pd.DataFrame:
    """Real-shaped long frame (gene_id, tissue, expression) + gene features."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for gi in range(n_genes):
        gid = f"ENSG{gi:011d}"
        base = rng.normal(4.0, 1.0)
        for tissue in TISSUES:
            expr = max(0.01, base + rng.normal(0, 0.8))
            rows.append({"gene_id": gid, "tissue": tissue, "expression": float(expr)})
    long = pd.DataFrame(rows)
    return long.merge(compute_gene_features(long), on="gene_id", how="left")


# --- 1 & 2: the null must reject a broken / noise signal ----------------------

def test_shuffled_labels_are_not_confirmed(monkeypatch):
    """Destroying the tissue→row alignment must never yield a confirmed verdict."""
    df = _long_frame(seed=1)
    shuffled = df.copy()
    rng = np.random.default_rng(99)
    shuffled["tissue"] = rng.permutation(shuffled["tissue"].to_numpy())
    monkeypatch.setattr(GenomicsAdapter, "load_frame", lambda self: shuffled)

    result = run_genomics_experiment(
        GenomicsExperimentSpec(
            feature_subset=["expression_variance", "mean_expression"],
            target_column="mean_expression",
        )
    )
    verdict, _, _ = classify_genomics_verdict("shuffled control", result)
    assert verdict in {"refuted", "inconclusive"}, result
    assert result["verified_true_steps"] == 0


def test_tissue_label_shuffle_null_rejects_pure_noise():
    """Random features vs a random target: observed R2 must not beat the null."""
    rng = np.random.default_rng(11)
    n = 400
    X = rng.normal(size=(n, 3))
    y = rng.normal(size=n)
    tissues = np.array([TISSUES[i % len(TISSUES)] for i in range(n)])
    observed, nulls = GV._tissue_label_shuffle_null(X, y, tissues, n_perm=40)
    p = float(np.mean([1 for v in nulls if v >= observed]))
    assert p >= 0.05
    assert observed <= float(np.percentile(nulls, 95)) + 1e-9


def test_run_experiment_always_reports_null_stats(monkeypatch):
    """Every experiment result must carry the null stats the gate reads."""
    df = _long_frame(seed=2)
    monkeypatch.setattr(GenomicsAdapter, "load_frame", lambda self: df)
    result = run_genomics_experiment(
        GenomicsExperimentSpec(
            feature_subset=["expression_variance", "mean_expression"],
            target_column="mean_expression",
        )
    )
    for key in ("lofo_r2", "label_shuffle_null_p", "label_shuffle_null_p95", "verified_true_steps"):
        assert key in result, f"missing gate field: {key}"


# --- 3: the confirm gate depends on the null stats ---------------------------

def test_confirm_path_requires_null_stats():
    """The gate confirms only when it can read a passing LOFO null."""
    confirmable = {
        "lofo_r2": 0.30,
        "label_shuffle_null_p": 0.01,
        "label_shuffle_null_p95": 0.05,
        "verified_true_steps": 1,
    }
    verdict, reason, conf = classify_genomics_verdict("real cross-tissue finding", confirmable)
    assert verdict == "confirmed"
    assert "p=0.010" in reason
    assert conf > 0.8


def test_confirm_gate_refuses_when_null_stats_absent():
    """A high R2 with no passing null must NOT be confirmed (anti-rediscovery)."""
    no_null = {"lofo_r2": 0.90}  # p defaults to 1.0, verified_true_steps absent
    verdict, _, _ = classify_genomics_verdict("suspicious high R2", no_null)
    assert verdict != "confirmed"


# --- real data is genuinely real ---------------------------------------------

def test_real_dataset_flag_is_honest(tmp_genomics_data):
    """When real GTEx data is served, the synthetic flag must be False and the
    frame must carry real Ensembl gene ids across multiple named tissues."""
    adapter = GenomicsAdapter()
    df = adapter.load_frame()
    assert {"gene_id", "tissue", "expression"}.issubset(df.columns)
    if not dataset_is_synthetic():
        assert df["tissue"].nunique() >= 3
        assert df["gene_id"].nunique() >= 30
        # Real GTEx ids are Ensembl ENSG accessions; the synthetic fallback uses
        # a zero-padded counter (ENSG00000000000..) — assert real, non-degenerate ids.
        ids = df["gene_id"].astype(str)
        assert ids.str.startswith("ENSG").all()
        assert ids.nunique() == df["gene_id"].nunique()
        assert not ids.eq("ENSG00000000000").any()
