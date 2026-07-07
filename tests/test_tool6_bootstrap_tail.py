"""TOOL6: the bootstrap significance tool's one-sided tails must not be inverted.

`cnt` counts the UPPER tail (resampled diff >= observed). Before the fix,
`alternative="less"` used `cnt/n_boot` (the upper tail), inverting the p-value:
a genuinely-lower group A got p≈1 and a higher one got p≈0. This guards the tails.
"""
from __future__ import annotations

from propab.tools.ml_research.statistical_significance import statistical_significance as ss


def _p(result) -> float:
    out = result.output
    return float(out.get("p_value", out.get("p")))


# Group A is clearly LOWER than group B.
_A = [1.0, 1.1, 0.9, 1.05, 0.95] * 6
_B = [2.0, 2.1, 1.9, 2.05, 1.95] * 6


def test_bootstrap_less_is_significant_when_a_below_b() -> None:
    # alternative="less" (A < B) is TRUE here -> p must be small.
    assert _p(ss(results_a=_A, results_b=_B, test="bootstrap", alternative="less")) < 0.05


def test_bootstrap_greater_not_significant_when_a_below_b() -> None:
    # alternative="greater" (A > B) is FALSE here -> p must be large.
    assert _p(ss(results_a=_A, results_b=_B, test="bootstrap", alternative="greater")) > 0.5


def test_bootstrap_greater_significant_when_a_above_b() -> None:
    # Swap: A now clearly ABOVE B -> "greater" true (small p), "less" false (large p).
    assert _p(ss(results_a=_B, results_b=_A, test="bootstrap", alternative="greater")) < 0.05
    assert _p(ss(results_a=_B, results_b=_A, test="bootstrap", alternative="less")) > 0.5


def test_bootstrap_two_sided_significant_either_direction() -> None:
    assert _p(ss(results_a=_A, results_b=_B, test="bootstrap", alternative="two_sided")) < 0.05
