"""
Cross-topology-family verification for network diffusion.

Design (mirrors the ``graph_invariants`` correlation-replication shape and the
``enzyme_kinetics`` label-shuffle null, adapted to what real graph data can
honestly support):

1. Build a real dataset: sample induced subgraphs from each real SNAP network
   (topology family) and simulate a diffusion outcome on each (SIR or cascade).
2. Test a structural law: does a structural feature (e.g. degree heterogeneity
   ``<k^2>/<k>``) predict the diffusion outcome? Measure the per-family
   Spearman correlation between the feature and the simulated outcome.
3. Cross-topology-family holdout: hold out one real network family; require the
   correlation to REPLICATE on it with the same sign as (and comparable
   strength to) the training families.
4. Adversarial null: within-family label-shuffle. For each permutation, the
   outcome values are shuffled *within each family* (preserving both marginal
   distributions but destroying the structure->outcome link) and the held-out
   correlation recomputed. Emitted as the ``lofo_r2`` / ``label_shuffle_null_p95``
   / ``label_shuffle_permutation_p`` triple the core artifact gate reads.
5. Robustness: the same held-out correlation must also hold under an *alternate
   simulator* (SIR<->cascade), so a "confirmed" is not an artifact of one
   dynamics.

Why correlation-replication and not enzyme-style cross-family regression R²:
with two real networks the leave-one-family-out regression is degenerate (train
on one regime, predict a different absolute-outcome regime -> R²<<0 even when the
structural law genuinely holds). The rank-correlation replication + within-family
shuffle null is the honest test the real data supports, and it *rejects* a broken
signal (verified in tests: a pure-noise feature does not confirm).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from propab.domain_modules.network_diffusion.adapter import (
    REAL_NETWORKS,
    DiffusionSpec,
    Subgraph,
    sample_subgraphs,
    structural_features,
)
from propab.domain_modules.network_diffusion.simulator import simulate

# Sampling / simulation budget. Kept modest so preflight + a verification stay
# well under a minute, but large enough for the null to have power.
N_SUBGRAPHS_PER_FAMILY = 36
SUBGRAPH_TARGET_SIZE = 100
# beta chosen so both real families sit in the super-critical-but-not-saturated
# regime (the sparse collaboration net otherwise floors near zero outbreak).
SIR_BETA = 0.20
SIR_GAMMA = 0.5
CASCADE_P = 0.14
N_SIM_RUNS = 14
N_PERMUTATIONS = 200
RANDOM_SEED = 42

# Confirmation thresholds.
MIN_ABS_CORR = 0.20        # both train and held-out correlation must exceed this
MIN_ABS_CORR_ALT = 0.12    # alternate-simulator replication (weaker allowed)
NULL_ALPHA = 0.05


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no scipy dependency needed)."""
    if len(x) < 3:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    a_sorted = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def _build_dataset(spec: DiffusionSpec, *, simulator: str, rng: np.random.Generator) -> dict[str, Any]:
    """Sample real subgraphs per family and simulate outcomes. Returns per-family arrays."""
    beta = CASCADE_P if simulator == "cascade" else SIR_BETA
    data: dict[str, dict[str, np.ndarray]] = {}
    for fam in spec.families:
        subs = sample_subgraphs(
            fam,
            n_samples=N_SUBGRAPHS_PER_FAMILY,
            target_size=SUBGRAPH_TARGET_SIZE,
            rng=rng,
        )
        feats: list[float] = []
        outs: list[float] = []
        for sub in subs:
            f = structural_features(sub)
            feats.append(f[spec.structural_feature])
            outs.append(
                simulate(
                    sub,
                    simulator=simulator,
                    outcome=spec.outcome,
                    beta=beta,
                    gamma=SIR_GAMMA,
                    n_runs=N_SIM_RUNS,
                    rng=rng,
                )
            )
        data[fam] = {"x": np.asarray(feats, float), "y": np.asarray(outs, float)}
    return data


def _held_out_correlation(data: dict[str, dict[str, np.ndarray]], held: str) -> float:
    d = data.get(held)
    if d is None or len(d["x"]) < 5:
        return 0.0
    return _spearman(d["x"], d["y"])


def _train_correlation(data: dict[str, dict[str, np.ndarray]], held: str) -> float:
    corrs = [
        _spearman(d["x"], d["y"])
        for fam, d in data.items()
        if fam != held and len(d["x"]) >= 5
    ]
    return float(np.mean(corrs)) if corrs else 0.0


def _label_shuffle_null(
    data: dict[str, dict[str, np.ndarray]],
    held: str,
    *,
    observed_abs: float,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Within-family shuffle null on the held-out correlation.

    Shuffling y within each family preserves both marginals but breaks the
    structure->outcome pairing. Returns (null_p95, permutation_p) on |corr|.
    """
    nulls: list[float] = []
    d = data[held]
    x = d["x"]
    y = d["y"].copy()
    for _ in range(n_perm):
        rng.shuffle(y)
        nulls.append(abs(_spearman(x, y)))
    arr = np.asarray(nulls, float)
    p95 = float(np.percentile(arr, 95)) if arr.size else 1.0
    perm_p = float(np.mean(arr >= observed_abs)) if arr.size else 1.0
    return p95, perm_p


def run_diffusion_experiment(spec: DiffusionSpec) -> dict[str, Any]:
    """
    Run the full cross-topology-family diffusion verification and return an
    evidence dict the verdict pipeline / artifact gate can interpret.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    primary_sim = spec.simulator
    data = _build_dataset(spec, simulator=primary_sim, rng=rng)

    families = [f for f in spec.families if len(data.get(f, {}).get("x", [])) >= 5]
    if len(families) < 2:
        raise ValueError(f"Too few usable topology families: {families}")

    held = spec.held_out_family if spec.held_out_family in families else sorted(families)[0]
    train_corr = _train_correlation(data, held)
    held_corr = _held_out_correlation(data, held)
    by_family = {fam: _spearman(d["x"], d["y"]) for fam, d in data.items() if len(d["x"]) >= 5}

    observed_abs = abs(held_corr)
    p95, perm_p = _label_shuffle_null(
        data, held, observed_abs=observed_abs, n_perm=N_PERMUTATIONS, rng=rng
    )

    # Sign/strength consistency across families.
    same_sign = (
        math.copysign(1, held_corr) == math.copysign(1, train_corr)
        and abs(held_corr) > 1e-6
        and abs(train_corr) > 1e-6
    )
    claim_sign_ok = True
    if spec.claim_sign == "positive":
        claim_sign_ok = held_corr > 0 and train_corr > 0
    elif spec.claim_sign == "negative":
        claim_sign_ok = held_corr < 0 and train_corr < 0

    cross_family_replicates = (
        same_sign
        and abs(held_corr) >= MIN_ABS_CORR
        and abs(train_corr) >= MIN_ABS_CORR
        and claim_sign_ok
    )

    # Alternate-simulator robustness: recompute held-out correlation under the
    # other dynamics. A confirmed finding must not be simulator-specific.
    alt_sim = "cascade" if primary_sim != "cascade" else "sir"
    alt_data = _build_dataset(spec, simulator=alt_sim, rng=np.random.default_rng(RANDOM_SEED + 1))
    alt_held_corr = _held_out_correlation(alt_data, held)
    robust_to_simulator = (
        math.copysign(1, alt_held_corr) == math.copysign(1, held_corr)
        and abs(alt_held_corr) >= MIN_ABS_CORR_ALT
    )

    survives_null = observed_abs > p95 and perm_p < NULL_ALPHA
    verified = bool(cross_family_replicates and survives_null and robust_to_simulator)

    # lofo_r2 shape: the held-out correlation is the "leave-one-family-out"
    # generalization statistic the artifact gate compares against the null p95.
    return {
        # --- lofo/null triple the artifact gate reads -----------------------
        "lofo_r2": float(held_corr),
        "label_shuffle_null_p95": float(p95),
        "label_shuffle_permutation_p": float(perm_p),
        # --- primary metric --------------------------------------------------
        "metric_name": "cross_family_diffusion_correlation",
        "metric_value": float(held_corr),
        "verification_method": "cross_topology_family_lofo",
        # --- diagnostics -----------------------------------------------------
        "train_correlation": float(train_corr),
        "held_out_correlation": float(held_corr),
        "held_out_family": held,
        "by_family": {k: float(v) for k, v in by_family.items()},
        "structural_feature": spec.structural_feature,
        "outcome": spec.outcome,
        "simulator": primary_sim,
        "alternate_simulator": alt_sim,
        "alt_held_out_correlation": float(alt_held_corr),
        "robust_to_simulator": robust_to_simulator,
        "cross_family_replicates": cross_family_replicates,
        "n_families": len(families),
        "n_samples": int(sum(len(data[f]["x"]) for f in families)),
        "n_permutations": N_PERMUTATIONS,
        "uses_synthetic_data": False,
        "data_provenance": {fam: REAL_NETWORKS[fam] for fam in families},
        # --- verdict counters ------------------------------------------------
        "verified_true_steps": 1 if verified else 0,
        "verified_false_steps": 0 if verified else 1,
        "discovery_worthy": verified,
        "trivial_rediscovery": not verified,
    }


def classify_diffusion_verdict(hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
    _ = hypothesis_text
    # NB: use explicit None checks, not ``or`` defaults — a legitimate perm_p of
    # 0.0 (null never beat the observed correlation) is falsy and ``x or 1.0``
    # would silently turn a strong confirmation into a spurious refutation.
    held_raw = result.get("held_out_correlation")
    perm_raw = result.get("label_shuffle_permutation_p")
    held = float(held_raw) if held_raw is not None else 0.0
    perm_p = float(perm_raw) if perm_raw is not None else 1.0
    replicates = bool(result.get("cross_family_replicates"))
    robust = bool(result.get("robust_to_simulator"))
    sim = result.get("simulator", "sir")

    if int(result.get("verified_true_steps") or 0) >= 1 and replicates and robust and perm_p < NULL_ALPHA:
        return (
            "confirmed",
            f"structure->diffusion law replicates on held-out real family "
            f"(r={held:.3f}, shuffle p={perm_p:.3f}, robust across {sim}<->alt simulator)",
            0.88,
        )
    if abs(held) < 0.1 or perm_p > 0.5:
        return (
            "refuted",
            f"no structure->diffusion signal on held-out family (r={held:.3f}, shuffle p={perm_p:.3f})",
            0.82,
        )
    if replicates and not robust:
        return (
            "inconclusive",
            f"signal present but simulator-specific (r={held:.3f}; fails alternate-simulator robustness)",
            0.55,
        )
    return (
        "inconclusive",
        f"weak/partial cross-family diffusion signal (r={held:.3f}, shuffle p={perm_p:.3f})",
        0.55,
    )
