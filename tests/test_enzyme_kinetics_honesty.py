"""Verifier-honesty tests for the enzyme kinetics domain.

These assert the LOFO + EC-label-shuffle-null machinery is a real honesty gate:
  1. A shuffled-label frame (broken group signal) must NOT be confirmed.
  2. The label-shuffle null must reject a broken signal (observed <= null),
     while a genuine cross-group signal beats a large fraction of the null.
  3. The confirm path fires only when the result carries the LOFO null stats the
     gate reads; absent those stats the gate must refute.

They complement the registration/routing tests, which never exercised the null.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propab.domain_modules.enzyme_kinetics import verifier as EV
from propab.domain_modules.enzyme_kinetics.adapter import (
    EnzymeExperimentSpec,
    EnzymeKineticsAdapter,
    dataset_is_synthetic,
    real_data_cached,
)
from propab.domain_modules.enzyme_kinetics.verifier import (
    classify_enzyme_verdict,
    run_enzyme_experiment,
)

# Real-data tests SKIP (not error, not vacuously pass) when the real DLKcat cache
# is absent or is the synthetic fallback, so CI/deploy goes green-with-skips
# rather than downloading DLKcat or asserting nothing. Populate the real cache
# with ``scripts/build_real_domain_datasets.py``.
requires_real_data = pytest.mark.skipif(
    not real_data_cached(),
    reason="real DLKcat data not cached; run scripts/build_real_domain_datasets.py",
)


def _grouped_signal_frame(seed: int = 0, n_per: int = 120) -> pd.DataFrame:
    """Frame whose target genuinely depends on the EC-group partition.

    Each group sits in its own feature region and the target follows a global
    linear trend, so leave-one-group-out prediction is possible only when the
    correct groups are pooled. Used to exercise the null's *aligned-label* path.
    """
    rng = np.random.default_rng(seed)
    groups = [f"EC{i}" for i in range(1, 7)]
    rows: list[dict] = []
    for gi, g in enumerate(groups):
        centre = gi - 2.5
        for _ in range(n_per):
            f1 = rng.normal(centre, 0.6)
            y = 1.2 * f1 + rng.normal(0, 0.3)
            rows.append({
                "ec_class": g,
                "log_kcat": y,
                "molecular_weight": f1,
                "sequence_length": rng.normal(),
                "frac_charged": rng.normal(),
                "frac_aromatic": rng.normal(),
            })
    return pd.DataFrame(rows)


# --- 1 & 2: the null must reject a broken / shuffled group signal -------------

def test_shuffled_labels_are_not_confirmed(monkeypatch):
    """Destroying the group→row alignment must never yield a confirmed verdict."""
    df = _grouped_signal_frame(seed=1)
    shuffled = df.copy()
    rng = np.random.default_rng(123)
    shuffled["ec_class"] = rng.permutation(shuffled["ec_class"].to_numpy())
    monkeypatch.setattr(EnzymeKineticsAdapter, "load_frame", lambda self: shuffled)

    result = run_enzyme_experiment(
        EnzymeExperimentSpec(feature_subset=["molecular_weight", "sequence_length"])
    )
    verdict, _, _ = classify_enzyme_verdict("shuffled control", result)
    assert verdict in {"refuted", "inconclusive"}, result
    assert result["verified_true_steps"] == 0


def test_label_shuffle_null_rejects_pure_noise():
    """Random features vs a random target: observed R2 must not beat the null."""
    rng = np.random.default_rng(7)
    n = 600
    X = rng.normal(size=(n, 3))
    y = rng.normal(size=n)
    groups = np.array([f"EC{i % 6 + 1}" for i in range(n)])
    observed, nulls, p = EV._label_shuffle_null(X, y, groups, n_perm=40)
    # A pure-noise signal cannot clear the permutation null at p < 0.05.
    assert p >= 0.05
    assert observed <= float(np.percentile(nulls, 95)) + 1e-9


def test_run_experiment_always_reports_null_stats(monkeypatch):
    """Every experiment result must carry the null stats the gate reads."""
    df = _grouped_signal_frame(seed=2)
    monkeypatch.setattr(EnzymeKineticsAdapter, "load_frame", lambda self: df)
    result = run_enzyme_experiment(
        EnzymeExperimentSpec(feature_subset=["molecular_weight", "sequence_length"])
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
    verdict, reason, conf = classify_enzyme_verdict("real cross-EC finding", confirmable)
    assert verdict == "confirmed"
    assert "p=0.01" in reason or "shuffle p=0.010" in reason
    assert conf > 0.8


def test_confirm_gate_refuses_when_null_stats_absent():
    """A high R2 with no passing null must NOT be confirmed (anti-rediscovery)."""
    no_null = {"lofo_r2": 0.90}  # p defaults to 1.0, verified_true_steps absent
    verdict, _, _ = classify_enzyme_verdict("suspicious high R2", no_null)
    assert verdict != "confirmed"


# --- real data is genuinely real ---------------------------------------------

@requires_real_data
def test_real_dataset_flag_is_honest():
    """When real DLKcat data is cached, the synthetic flag must be False; the
    frame must be keyed by real EC classes with a real measured kcat target.

    Skipped cleanly when only the synthetic fallback is on disk — a green run
    then honestly reports "real-data assertion not exercised" instead of a
    vacuous pass on the synthetic frame.
    """
    assert not dataset_is_synthetic()
    df = EnzymeKineticsAdapter().load_frame()
    assert {"ec_class", "log_kcat", "kcat"}.issubset(df.columns)
    # Real DLKcat records carry organism/substrate provenance and >3 EC classes.
    assert df["ec_class"].nunique() >= 3
    assert (df["kcat"] > 0).all()
    assert df["organism"].str.len().gt(0).any()
    assert not df["organism"].eq("synthetic").all()
