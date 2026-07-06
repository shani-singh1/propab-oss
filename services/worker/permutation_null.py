"""Self-contained label-permutation null for the generic worker path (D2).

A generic (think-act / heuristic) experiment that runs a significance comparison
on two real outcome arrays (``results_a`` vs ``results_b``, or treatment vs
baseline) has, until now, had no way to attach a *real* adversarial null to its
evidence. Without a null, ``artifact_verification._survives_permutation`` fails
closed and the verdict pipeline keeps such a result inconclusive forever.

This module computes a genuine **label-permutation null** from the experiment's
own outcome arrays — the SAME data the observed statistic was measured on:

    1. observed statistic  = |mean(a) - mean(b)|   (absolute mean difference)
    2. pool = concat(a, b); repeatedly (>= ``n_permutations``) shuffle the group
       labels, split back into two groups of the original sizes, and recompute
       the SAME statistic on the shuffled labels.
    3. permutation_p = (#{|perm_stat| >= |observed_stat|} + 1) / (n_permutations + 1)
       (the +1 is the standard, unbiased Monte-Carlo estimator that never yields
       a p of exactly 0).
    4. n_samples = len(a) + len(b).

Integrity properties (why this can never fabricate a passing null):
  * The p-value is computed ONLY from permutations of the two arrays actually
    passed in. Nothing is self-reported; the number is a deterministic function
    of the real data plus a fixed seed.
  * If both outcome arrays are not present (a single reported p-value, one array,
    or arrays too small), ``compute_label_permutation_null`` returns ``None`` —
    the caller then leaves ``permutation_p`` absent and the result correctly
    stays inconclusive. There is no code path that emits a ``permutation_p``
    without an actual permutation of two real arrays.
  * A degenerate observed statistic (both groups identical) yields the maximal
    p-value, so a no-difference experiment cannot spuriously "survive".

The permutation math is intentionally kept free of any propab imports so it can
be unit-tested in isolation and is deterministic under ``seed``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # numpy is a hard dependency of the significance tools; guard anyway.
    import numpy as _np
except Exception:  # pragma: no cover - numpy is always present in this repo
    _np = None  # type: ignore[assignment]

# The numeric-array param pairs a significance tool exposes, in preference order.
# Each entry is (name_of_group_a, name_of_group_b). ``bootstrap_confidence``
# (single ``values`` array) is intentionally excluded: a one-sample bootstrap is
# not a two-group comparison and cannot ground a label-permutation null.
_SIG_ARRAY_PAIRS: tuple[tuple[str, str], ...] = (
    ("results_a", "results_b"),
    ("our_results", "baseline_results"),
    ("treatment", "baseline"),
    ("treatment_results", "baseline_results"),
)

# Minimum observations *per group* for a permutation null to be meaningful.
_MIN_PER_GROUP = 2
# Default number of label permutations. >= 1000 as required by the D2 spec.
DEFAULT_N_PERMUTATIONS = 2000


@dataclass(frozen=True)
class PermutationNullResult:
    """Outcome of a genuine label-permutation null."""

    permutation_p: float
    n_samples: int
    n_permutations: int
    observed_stat: float
    n_a: int
    n_b: int

    def to_evidence_fields(self) -> dict[str, Any]:
        """The exact keys ``_survives_permutation`` (read-only) consumes, plus audit."""
        return {
            "permutation_p": self.permutation_p,
            "n_samples": self.n_samples,
            # Audit-only breadcrumbs; ignored by the gate but useful in the paper trace.
            "permutation_null_n_permutations": self.n_permutations,
            "permutation_null_observed_stat": self.observed_stat,
            "permutation_null_group_sizes": [self.n_a, self.n_b],
        }


def _coerce_numeric_array(value: Any) -> list[float] | None:
    """Return a homogeneous float list (len >= 2) or ``None``.

    Rejects bools (``bool`` is an ``int`` subclass) and any list containing a
    non-numeric element — we never silently drop values to force a null.
    """
    if not isinstance(value, (list, tuple)):
        return None
    out: list[float] = []
    for x in value:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        out.append(float(x))
    if len(out) < _MIN_PER_GROUP:
        return None
    return out


def extract_two_group_arrays(params: dict[str, Any] | None) -> tuple[list[float], list[float]] | None:
    """Pull two real outcome arrays from a significance-tool call's params.

    Fails closed (returns ``None``) unless BOTH arrays of a recognised pair are
    present, numeric, and have >= 2 observations each. A single array (e.g. a
    lone ``values`` for bootstrap, or only ``results_a``) is not enough to ground
    a label-permutation null and must not synthesize one.
    """
    if not isinstance(params, dict):
        return None
    for a_key, b_key in _SIG_ARRAY_PAIRS:
        a = _coerce_numeric_array(params.get(a_key))
        b = _coerce_numeric_array(params.get(b_key))
        if a is not None and b is not None:
            return a, b
    return None


def _abs_mean_diff(a: "Any", b: "Any") -> float:
    return float(abs(a.mean() - b.mean()))


def compute_label_permutation_null(
    results_a: Any,
    results_b: Any,
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 12345,
    statistic: str = "mean_diff",
) -> PermutationNullResult | None:
    """Compute a genuine label-permutation null from two real outcome arrays.

    Returns ``None`` (fail-closed) when the two arrays are not both present with
    at least 2 observations each, or numpy is unavailable — the caller then
    leaves ``permutation_p`` absent. Never fabricates a null.

    Deterministic under ``seed``. ``statistic`` currently supports the absolute
    mean difference (the statistic the significance tools compare on); the
    parameter exists so a caller that knows the exact tool statistic can extend
    this without changing the call site.
    """
    if _np is None:
        return None
    a = _coerce_numeric_array(list(results_a) if results_a is not None else None)
    b = _coerce_numeric_array(list(results_b) if results_b is not None else None)
    if a is None or b is None:
        return None
    if statistic != "mean_diff":
        # Only the mean-difference statistic is implemented; refuse rather than
        # silently computing a different null than the observed effect used.
        return None

    n_perm = max(1000, int(n_permutations))
    arr_a = _np.asarray(a, dtype=_np.float64)
    arr_b = _np.asarray(b, dtype=_np.float64)
    n_a = int(arr_a.size)
    n_b = int(arr_b.size)
    pooled = _np.concatenate([arr_a, arr_b])
    n_total = n_a + n_b
    total_sum = float(pooled.sum())

    observed = _abs_mean_diff(arr_a, arr_b)

    # Vectorized permutation: for each shuffle, only the sum of the first n_a
    # pooled elements is needed, since
    #   mean(a') - mean(b') = sum_a'/n_a - (total_sum - sum_a')/n_b.
    # Permutations are generated by argsort of uniform random keys (deterministic
    # under `seed`), batched to bound peak memory at large n_perm.
    rng = _np.random.default_rng(int(seed))
    ge = 0
    batch = max(1, min(n_perm, max(1, 4_000_000 // max(1, n_total))))
    done = 0
    while done < n_perm:
        this = min(batch, n_perm - done)
        keys = rng.random((this, n_total))
        # argsort gives a random permutation per row; take the first n_a indices.
        idx = _np.argsort(keys, axis=1)[:, :n_a]
        sum_a = pooled[idx].sum(axis=1)
        diff = sum_a / n_a - (total_sum - sum_a) / n_b
        stats = _np.abs(diff)
        ge += int(_np.count_nonzero(stats >= observed - 1e-12))
        done += this

    # Unbiased Monte-Carlo p-value: (#>= + 1) / (n_perm + 1).
    permutation_p = (ge + 1) / (n_perm + 1)

    return PermutationNullResult(
        permutation_p=float(permutation_p),
        n_samples=int(n_total),
        n_permutations=int(n_perm),
        observed_stat=float(observed),
        n_a=n_a,
        n_b=n_b,
    )
