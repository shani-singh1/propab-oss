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


def _noise_frame(seed: int = 0, n_per: int = 120) -> pd.DataFrame:
    """Frame whose target is independent of every feature and of the EC group,
    so the observed LOFO R2 cannot beat the EC-shuffle null."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(6 * n_per):
        rows.append({
            "ec_class": f"EC{i % 6 + 1}",
            "log_kcat": float(rng.normal()),
            "molecular_weight": float(rng.normal()),
            "sequence_length": float(rng.normal()),
            "frac_charged": float(rng.normal()),
            "frac_aromatic": float(rng.normal()),
        })
    return pd.DataFrame(rows)


def _scripted_lofo(observed: float, nulls: list[float]):
    """Drop-in for ``_lofo_r2`` returning observed on its first call and the
    scripted null values thereafter."""
    seq = [observed, *nulls]
    box = {"i": 0}

    def fn(X, y, groups):  # noqa: ANN001, ARG001
        i = box["i"]
        box["i"] += 1
        return seq[i] if i < len(seq) else 0.0

    return fn


# --- the permutation p-value is the FRACTION of nulls >= observed ------------
# The pre-fix formula ``np.mean([1 for n in nulls if n >= observed])`` is the mean
# of an all-ones list (== 1.0 when any null ties/exceeds observed, nan when none
# do), so a real cross-EC signal was reported "refuted" and could NEVER confirm.
# Both tests below FAIL against that old formula and PASS only with the fix.

def test_planted_signal_yields_small_p_and_confirms(monkeypatch):
    """Observed LOFO R2 beats the EC-shuffle null on all but 1/40 perms ->
    p = 0.025 (small) -> *confirmed*. Old formula pins p to 1.0 and INVERTS the
    identical evidence to 'refuted'."""
    df = _grouped_signal_frame(seed=3)
    monkeypatch.setattr(EnzymeKineticsAdapter, "load_frame", lambda self: df)
    # observed=0.30; only 1 of 40 EC-shuffle perms ties/exceeds it.
    monkeypatch.setattr(EV, "_lofo_r2", _scripted_lofo(0.30, [0.34] + [0.0] * 39))
    result = run_enzyme_experiment(
        EnzymeExperimentSpec(feature_subset=["molecular_weight", "sequence_length"])
    )
    assert result["lofo_r2"] == pytest.approx(0.30)
    assert result["label_shuffle_null_p"] == pytest.approx(1 / 40)  # 0.025 — NOT 1.0
    assert result["verified_true_steps"] == 1
    verdict, _, conf = classify_enzyme_verdict("planted cross-EC signal", result)
    assert verdict == "confirmed", result
    assert conf > 0.8


def test_pure_noise_yields_large_p_and_is_not_confirmed(monkeypatch):
    """Real end-to-end run: target independent of every feature -> observed
    cannot beat the null -> p is a genuine (large) fraction and the verdict is
    never 'confirmed'."""
    df = _noise_frame(seed=1)
    monkeypatch.setattr(EnzymeKineticsAdapter, "load_frame", lambda self: df)
    result = run_enzyme_experiment(
        EnzymeExperimentSpec(feature_subset=["molecular_weight", "sequence_length"])
    )
    p = result["label_shuffle_null_p"]
    assert not np.isnan(p) and 0.0 <= p <= 1.0
    assert p >= 0.05, result
    assert result["verified_true_steps"] == 0
    verdict, _, _ = classify_enzyme_verdict("pure noise", result)
    assert verdict != "confirmed"


# --- 1 & 2: the null must reject a broken / shuffled group signal -------------

def test_shuffled_target_is_not_confirmed(monkeypatch):
    """Breaking X→y (shuffling the TARGET) must never confirm — the honest negative
    control.

    NOTE: this previously shuffled the ``ec_class`` GROUP labels, but
    ``_grouped_signal_frame`` has a genuine *global* signal (log_kcat = 1.2·mw),
    which survives group-label shuffling. The old group-shuffle null (a bug — it
    permuted the split instead of the target) masked this by also losing power on
    shuffled groups. With the corrected target-permutation null, that data
    legitimately confirms a real X→y relationship. The real negative control is to
    destroy X→y itself, which we do here by permuting the target."""
    df = _grouped_signal_frame(seed=1)
    shuffled = df.copy()
    rng = np.random.default_rng(123)
    shuffled["log_kcat"] = rng.permutation(shuffled["log_kcat"].to_numpy())
    monkeypatch.setattr(EnzymeKineticsAdapter, "load_frame", lambda self: shuffled)

    result = run_enzyme_experiment(
        EnzymeExperimentSpec(feature_subset=["molecular_weight", "sequence_length"])
    )
    verdict, _, _ = classify_enzyme_verdict("shuffled target control", result)
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
