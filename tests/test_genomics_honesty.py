"""Verifier-honesty tests for the genomics domain.

These assert the leave-one-tissue-out (LOFO) + target-label-shuffle-null
machinery is a real honesty gate:
  1. A shuffled-label frame (broken tissue signal) must NOT be confirmed.
  2. The target-label-shuffle null must reject a pure-noise signal.
  3. The confirm path fires only when the result carries the LOFO null stats the
     gate reads; absent those stats the gate must refute.

They complement the registration/routing tests, which never exercised the null.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propab.domain_modules.genomics import verifier as GV
from propab.domain_modules.genomics.adapter import (
    GenomicsAdapter,
    GenomicsExperimentSpec,
    compute_gene_features,
    dataset_is_synthetic,
    real_data_cached,
)
from propab.domain_modules.genomics.verifier import (
    classify_genomics_verdict,
    run_genomics_experiment,
)

TISSUES = ["Brain", "Heart", "Liver", "Lung", "Muscle", "Skin", "Blood"]

# Real-data tests SKIP (not error, not vacuously pass) when the real GTEx cache
# is absent or is the synthetic fallback, so CI/deploy goes green-with-skips
# rather than downloading GTEx or asserting nothing. Populate the real cache with
# ``scripts/build_real_domain_datasets.py``.
requires_real_data = pytest.mark.skipif(
    not real_data_cached(),
    reason="real GTEx data not cached; run scripts/build_real_domain_datasets.py",
)


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


def _noise_frame(seed: int = 0, n_genes: int = 300) -> pd.DataFrame:
    """Frame carrying a per-gene ``rand_target`` that is independent of every
    expression-derived feature, so the observed LOFO R2 cannot beat the null."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for gi in range(n_genes):
        gid = f"ENSG{gi:011d}"
        base = rng.normal(4.0, 1.0)
        rand_target = float(rng.normal())
        for tissue in TISSUES:
            expr = max(0.01, base + rng.normal(0, 0.8))
            rows.append(
                {"gene_id": gid, "tissue": tissue, "expression": float(expr), "rand_target": rand_target}
            )
    long = pd.DataFrame(rows)
    return long.merge(compute_gene_features(long), on="gene_id", how="left")


def _scripted_lofo(observed: float, nulls: list[float]):
    """Drop-in for ``_leave_one_tissue_out_r2`` returning the observed statistic
    on its first call and the scripted null values thereafter."""
    seq = [observed, *nulls]
    box = {"i": 0}

    def fn(X, y, groups):  # noqa: ANN001, ARG001
        i = box["i"]
        box["i"] += 1
        return seq[i] if i < len(seq) else 0.0

    return fn


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


def test_target_label_shuffle_null_rejects_pure_noise():
    """Random features vs a random target: observed R2 must not beat the null.

    The null permutes the target labels (``y``) across genes, so a signal that
    is really absent cannot beat it.
    """
    rng = np.random.default_rng(11)
    n = 400
    X = rng.normal(size=(n, 3))
    y = rng.normal(size=n)
    tissues = np.array([TISSUES[i % len(TISSUES)] for i in range(n)])
    observed, nulls = GV._target_label_shuffle_null(X, y, tissues, n_perm=40)
    p = float(np.mean(np.asarray(nulls) >= observed))
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


# --- the permutation p-value is the FRACTION of nulls >= observed ------------
# These exercise the real run_genomics_experiment p-value + verdict end-to-end.
# The pre-fix formula ``np.mean([1 for n in nulls if n >= observed])`` is the mean
# of an all-ones list (== 1.0 when any null ties/exceeds observed, nan when none
# do), so a genuine signal was reported "refuted" and could NEVER be confirmed.
# Both tests below FAIL against that old formula and PASS only with the fix.

def test_planted_signal_yields_small_p_and_confirms(monkeypatch):
    """Observed LOFO R2 beats the tissue-shuffle null on all but 2/50 perms ->
    p = 0.04 (small) -> *confirmed*. Old formula pins p to 1.0 and INVERTS the
    identical evidence to 'refuted'."""
    df = _long_frame(seed=3)
    monkeypatch.setattr(GenomicsAdapter, "load_frame", lambda self: df)
    # observed=0.30; only 2 of 50 tissue-shuffle perms tie/exceed it.
    monkeypatch.setattr(GV, "_leave_one_tissue_out_r2", _scripted_lofo(0.30, [0.34, 0.34] + [0.0] * 48))
    result = run_genomics_experiment(
        GenomicsExperimentSpec(
            feature_subset=["expression_variance", "mean_expression"],
            target_column="mean_expression",
        )
    )
    assert result["lofo_r2"] == pytest.approx(0.30)
    assert result["label_shuffle_null_p"] == pytest.approx(2 / 50)  # 0.04 — NOT 1.0
    assert result["verified_true_steps"] == 1
    verdict, _, conf = classify_genomics_verdict("planted cross-tissue signal", result)
    assert verdict == "confirmed", result
    assert conf > 0.8


def test_pure_noise_yields_large_p_and_is_not_confirmed(monkeypatch):
    """Real end-to-end run: target independent of every feature -> observed
    cannot beat the null -> p is a genuine (large) fraction and the verdict is
    never 'confirmed'. Guards against a p-value pinned regardless of the data."""
    df = _noise_frame(seed=1)
    monkeypatch.setattr(GenomicsAdapter, "load_frame", lambda self: df)
    result = run_genomics_experiment(
        GenomicsExperimentSpec(
            feature_subset=["expression_variance", "mean_expression"],
            target_column="rand_target",
        )
    )
    p = result["label_shuffle_null_p"]
    assert not np.isnan(p) and 0.0 <= p <= 1.0
    assert p >= 0.05, result
    assert result["verified_true_steps"] == 0
    verdict, _, _ = classify_genomics_verdict("pure noise", result)
    assert verdict != "confirmed"


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

@requires_real_data
def test_real_dataset_flag_is_honest():
    """When real GTEx data is cached, the synthetic flag must be False and the
    frame must carry real Ensembl gene ids across multiple named tissues.

    Skipped cleanly when only the synthetic fallback is on disk — a green run
    then honestly reports "real-data assertion not exercised" instead of a
    vacuous pass on the synthetic frame.
    """
    assert not dataset_is_synthetic()
    df = GenomicsAdapter().load_frame()
    assert {"gene_id", "tissue", "expression"}.issubset(df.columns)
    assert df["tissue"].nunique() >= 3
    assert df["gene_id"].nunique() >= 30
    # Real GTEx ids are Ensembl ENSG accessions; the synthetic fallback uses
    # a zero-padded counter (ENSG00000000000..) — assert real, non-degenerate ids.
    ids = df["gene_id"].astype(str)
    assert ids.str.startswith("ENSG").all()
    assert ids.nunique() == df["gene_id"].nunique()
    assert not ids.eq("ENSG00000000000").any()
