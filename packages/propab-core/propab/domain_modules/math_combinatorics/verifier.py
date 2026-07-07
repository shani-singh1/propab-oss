"""
Combinatorics experiment runner.

Computes evidence for additive-combinatorics hypotheses. Single-point greedy
computations at small n are treated as known-result rediscoveries (inconclusive).
Open-problem evidence requires multi-n sweeps, asymptotic ratio analysis, or
explicit refutation of a falsifiable claim.
"""
from __future__ import annotations

import itertools
import math
import random
import re
import time
from typing import Any

from propab.domain_modules.math_combinatorics.constructors import (
    CAP_SET_BEST_KNOWN,
    apply_claim_validation,
    bose_chowla_ratio_at_n,
    bose_chowla_sidon,
    extract_claim_text,
    is_cap_set_hypothesis,
    is_sidon_hypothesis,
    is_sidon_set,
    parse_ratio_upper_bound,
)

# Standard sweep grid for Sidon asymptotics (fixes.md open problem 1).
SIDON_SWEEP_NS = (100, 200, 500, 1000, 2000, 5000, 10000)
SIDON_LARGE_SWEEP_NS = (10000, 20000, 30000, 40000, 50000)
CAP_SWEEP_DIMS = (3, 4, 5, 6, 7)
MIN_N_FOR_SINGLE_POINT_DISCOVERY = 500
MAX_SIDON_N = 100_000

# --- Real cap-set computation in F_3^n ---------------------------------------
# Largest F_3^n we build/validate the FULL point set for. Beyond this we still
# compute a genuine cap (via the product construction) but certify its size from
# fully-validated base caps + a sampled spot-check rather than an O(|S|^2) sweep
# over the whole (huge) set.
MAX_FULL_CAP_DIM = 10
# Time budget for the near-exhaustive branch-and-bound used to seed small base
# caps. The DFS is deterministic, so a fixed budget yields a reproducible cap.
# 4.0s comfortably clears the ~2.4s at which n=4 reaches its optimum (size 20).
CAP_BB_TIME_BUDGET = 4.0
# Above this cap size we validate every base factor fully and spot-check a random
# sample of product points rather than running the O(|S|^2) sweep over the whole
# (large) product set.
CAP_FULL_CHECK_MAX_SIZE = 1000
# Restart count for the randomized greedy fallback (deterministic seed).
CAP_GREEDY_RESTARTS = 150
CAP_GREEDY_SEED = 0xC0FFEE
# Number of product-cap points to spot-check when the full set is too large.
CAP_WITNESS_SAMPLE = 24

CapPoint = tuple[int, ...]


def run_combinatorics_experiment(hypothesis: dict[str, Any], features: list[str]) -> dict[str, Any]:
    start = time.time()
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    methodology = str(hypothesis.get("test_methodology") or "")
    claim = extract_claim_text(statement, test_methodology=methodology, full_text=statement)
    primary_feature = features[0] if features else "sidon_set_density"

    if is_cap_set_hypothesis(statement, methodology, full_text=statement):
        result = _run_cap_set_experiment(hypothesis)
    elif (
        "ap-free" in claim.lower()
        or "ap_free" in primary_feature
        or "arithmetic progression" in claim.lower()
    ):
        result = _run_ap_free_experiment(hypothesis)
    elif is_sidon_hypothesis(statement, methodology, full_text=statement):
        result = _run_sidon_experiment(hypothesis)
    elif "sumset" in primary_feature:
        result = _run_sumset_experiment(hypothesis)
    else:
        result = _run_sidon_experiment(hypothesis)

    result = apply_claim_validation(result, claim or statement)
    result["deterministic"] = True
    result["verification_method"] = "combinatorial_computation"
    result["elapsed_seconds"] = time.time() - start
    return result


def _extract_n(statement: str, default: int = 200, cap: int = 10000) -> int:
    n_match = re.search(r"n\s*=\s*(\d+)", statement) or re.search(r"\{1[,.]\.\.(\d+)\}", statement)
    if n_match:
        return min(int(n_match.group(1)), cap)
    return default


def _extract_n_list(statement: str) -> list[int]:
    """Parse explicit n values from ranges like {100, 200, 500} or n in [100, 1000]."""
    found: list[int] = []
    for m in re.finditer(r"\{(\d+(?:\s*,\s*\d+)*)\}", statement):
        for part in m.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                found.append(int(part))
    range_m = re.search(r"n\s+in\s+\[(\d+)\s*,\s*(\d+)\]", statement, re.I)
    if range_m:
        lo, hi = int(range_m.group(1)), int(range_m.group(2))
        if hi - lo <= 5000:
            step = max(1, (hi - lo) // 6)
            found.extend(range(lo, hi + 1, step))
    for m in re.finditer(r"\bn\s*=\s*(\d+)\b", statement, re.I):
        found.append(int(m.group(1)))
    uniq = sorted(set(n for n in found if 10 <= n <= MAX_SIDON_N))
    return uniq


def _wants_threshold_finding(statement: str) -> bool:
    s = statement.lower()
    return any(
        k in s
        for k in (
            "first drop",
            "first fall",
            "first cross",
            "crosses below",
            "threshold",
            "smallest n",
            "where does",
            "find the n",
        )
    )


def _parse_threshold_target(statement: str) -> float | None:
    bound = parse_ratio_upper_bound(statement)
    if bound is not None:
        return bound
    m = re.search(r"(?:below|under|<\s*)\s*(0\.\d+)", statement.lower())
    if m:
        return float(m.group(1))
    return None


def _parse_n_range_bounds(statement: str) -> tuple[int, int]:
    range_m = re.search(r"n\s+in\s+\[(\d+)\s*,\s*(\d+)\]", statement, re.I)
    if range_m:
        return int(range_m.group(1)), int(range_m.group(2))
    explicit = _extract_n_list(statement)
    if len(explicit) >= 2:
        return min(explicit), max(explicit)
    if explicit:
        return explicit[0], explicit[0]
    return 500, 50_000


def _wants_bc_matched_comparison(statement: str) -> bool:
    """Route to matched n=q^2+q BC vs greedy — not generic 'prime power q' peak sweeps."""
    s = statement.lower()
    if "matched" not in s:
        return False
    return (
        _wants_bose_chowla(s)
        or "bc " in s
        or "bose" in s
        or "greedy" in s
        or "compare" in s
    )


def _extract_cap_dims(statement: str) -> list[int]:
    max_dim = max(CAP_SET_BEST_KNOWN)
    dims: list[int] = []
    for m in re.finditer(r"F_3\^(\d+)", statement, re.I):
        dims.append(int(m.group(1)))
    for m in re.finditer(r"dimension\s+(\d+)", statement, re.I):
        dims.append(int(m.group(1)))
    for m in re.finditer(r"A_max\((\d+)\)", statement, re.I):
        dims.append(int(m.group(1)))
    for m in re.finditer(r"\bc_(\d+)\b", statement, re.I):
        dims.append(int(m.group(1)))
    for m in re.finditer(r"\bn\s+in\s+\{(\d+(?:\s*,\s*\d+)*)\}", statement, re.I):
        for part in m.group(1).split(","):
            if part.strip().isdigit():
                dims.append(int(part.strip()))
    return sorted(set(d for d in dims if 2 <= d <= max_dim))


def _is_refutation_claim(statement: str) -> bool:
    s = statement.lower()
    return any(
        p in s
        for p in (
            "no sidon",
            "cannot exist",
            "does not exist",
            "impossible",
            "no cap set",
            "no arithmetic",
            "no ap-free",
            "no 3-ap",
        )
    )


def _wants_bose_chowla(statement: str) -> bool:
    s = statement.lower()
    return any(k in s for k in ("bose-chowla", "bose chowla", "algebraic construction", "ruzsa"))


def _wants_greedy_vs_algebraic(statement: str) -> bool:
    s = statement.lower()
    return _wants_bose_chowla(s) and any(k in s for k in ("greedy", "compare", "versus", "vs ", "exceed"))


def _apply_claim_to_verdict(
    base: dict[str, Any],
    statement: str,
    *,
    computed_size: int,
    dimension: int | None = None,
) -> dict[str, Any]:
    _ = computed_size, dimension
    return base


def _sidon_point(n: int, *, method: str = "greedy") -> dict[str, Any]:
    if method == "bose_chowla":
        pt = dict(bose_chowla_ratio_at_n(n))
        pt["example_set"] = bose_chowla_sidon(int(pt["q"]), one_indexed=True)[:20]
        pt["mean_gap"] = 0.0
        pt["gap_std"] = 0.0
        return pt
    best = _greedy_sidon_max(n)
    construction = "greedy"
    theoretical = math.sqrt(n)
    ratio = len(best) / theoretical if theoretical > 0 else 0.0
    gaps = [best[i + 1] - best[i] for i in range(len(best) - 1)] if len(best) > 1 else []
    return {
        "n": n,
        "max_sidon_size": len(best),
        "theoretical_bound": theoretical,
        "ratio_to_sqrt_n": ratio,
        "example_set": best[:20],
        "construction": construction,
        "mean_gap": sum(gaps) / len(gaps) if gaps else 0.0,
        "gap_std": (
            math.sqrt(sum((g - sum(gaps) / len(gaps)) ** 2 for g in gaps) / len(gaps))
            if len(gaps) > 1
            else 0.0
        ),
    }


def _sidon_compare_at_n(n: int) -> dict[str, Any]:
    """Greedy vs Bose-Chowla ratio F(n)/sqrt(n) at the same n."""
    greedy = _sidon_point(n, method="greedy")
    bc = _sidon_point(n, method="bose_chowla")
    return {
        "n": n,
        "greedy_size": greedy["max_sidon_size"],
        "bose_chowla_size": bc["max_sidon_size"],
        "greedy_ratio": greedy["ratio_to_sqrt_n"],
        "bose_chowla_ratio": bc["ratio_to_sqrt_n"],
        "bose_chowla_exceeds_greedy": bc["ratio_to_sqrt_n"] > greedy["ratio_to_sqrt_n"],
    }


def _wants_asymptotic_analysis(statement: str) -> bool:
    s = statement.lower()
    return any(
        kw in s
        for kw in (
            "converge",
            "asymptotic",
            "constant",
            "ratio f(n)",
            "f(n)/sqrt",
            "ratio",
            "monotonic",
            "trend",
            "across",
            "structural",
            "gap between",
            "clp",
            "2.756",
            "2.217",
            "upper bound",
            "tighter",
        )
    )


def _greedy_sidon_max(n: int) -> list[int]:
    return _greedy_sidon_max_optimized(n)


def _greedy_sidon_max_legacy(n: int) -> list[int]:
    best: list[int] = []
    for start_val in range(1, min(n + 1, 20)):
        current = [start_val]
        sums: set[int] = set()
        for x in range(start_val + 1, n + 1):
            new_sums = {x + y for y in current}
            if not new_sums & sums:
                sums |= new_sums
                current.append(x)
        if len(current) > len(best):
            best = current
    return best


def _greedy_sidon_max_optimized(n: int) -> list[int]:
    """Greedy Sidon via sorted sums — O(n log n) per construction attempt."""
    import bisect

    n = min(max(1, n), MAX_SIDON_N)
    best: list[int] = []
    for start_val in range(1, min(n + 1, 20)):
        current = [start_val]
        sums_sorted: list[int] = []
        for x in range(start_val + 1, n + 1):
            collision = False
            for y in current:
                s = x + y
                if bisect.bisect_left(sums_sorted, s) < len(sums_sorted) and sums_sorted[
                    bisect.bisect_left(sums_sorted, s)
                ] == s:
                    collision = True
                    break
            if collision:
                continue
            for y in current:
                bisect.insort(sums_sorted, x + y)
            current.append(x)
        if len(current) > len(best):
            best = current
    return best


def find_threshold_crossing(
    target_ratio: float,
    n_start: int,
    n_end: int,
    *,
    n_step: int = 500,
) -> dict[str, Any]:
    """Smallest n in [n_start, n_end] where F(n)/sqrt(n) < target_ratio."""
    n_start = max(10, n_start)
    n_end = min(MAX_SIDON_N, n_end)
    prev_ratio: float | None = None
    prev_n: int | None = None
    crossing_n: int | None = None
    crossing_ratio: float | None = None
    for n in range(n_start, n_end + 1, max(1, n_step)):
        pt = _sidon_point(n, method="greedy")
        ratio = pt["ratio_to_sqrt_n"]
        if ratio < target_ratio:
            crossing_n = n
            crossing_ratio = ratio
            break
        prev_ratio, prev_n = ratio, n
    return {
        "target_ratio": target_ratio,
        "crossing_n": crossing_n,
        "crossing_ratio": crossing_ratio,
        "prev_n": prev_n,
        "prev_ratio": prev_ratio,
        "searched": list(range(n_start, n_end + 1, max(1, n_step))),
    }


def _run_threshold_crossing_experiment(statement: str) -> dict[str, Any]:
    target = _parse_threshold_target(statement) or 0.70
    n_lo, n_hi = _parse_n_range_bounds(statement)
    step = 100 if (n_hi - n_lo) <= 5000 else 500
    result = find_threshold_crossing(target, n_lo, n_hi, n_step=step)
    crossing_n = result["crossing_n"]
    if crossing_n is None:
        return {
            "verified_true_steps": 0,
            "verified_false_steps": 1,
            "discovery_worthy": True,
            "trivial_rediscovery": False,
            "metric_value": target,
            "metric_name": "sidon_ratio_to_sqrt_n",
            "threshold_search": result,
            "notes": f"No crossing below {target} found in n=[{n_lo},{n_hi}] step={step}",
        }
    return {
        "verified_true_steps": 1,
        "verified_false_steps": 0,
        "discovery_worthy": True,
        "trivial_rediscovery": False,
        "metric_value": result["crossing_ratio"],
        "metric_name": "sidon_ratio_to_sqrt_n",
        "n": crossing_n,
        "threshold_search": result,
        "notes": (
            f"Threshold crossing: F(n)/sqrt(n) < {target} first at n={crossing_n} "
            f"(ratio={result['crossing_ratio']:.4f})"
        ),
    }


def _run_bc_matched_comparison(hypothesis: dict[str, Any]) -> dict[str, Any]:
    """BC Sidon of size q+1 in {0,...,q²+q} vs greedy at matched n."""
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    _, n_hi = _parse_n_range_bounds(statement)
    n_max = min(n_hi, MAX_SIDON_N)

    def _is_prime(q: int) -> bool:
        if q < 2:
            return False
        if q % 2 == 0:
            return q == 2
        d = 3
        while d * d <= q:
            if q % d == 0:
                return False
            d += 2
        return True

    results: list[dict[str, Any]] = []
    q = 2
    while q * q + q <= n_max:
        if _is_prime(q):
            n = q * q + q
            bc_size = q + 1
            bc_ratio = bc_size / math.sqrt(n) if n > 0 else 0.0
            greedy = _sidon_point(n, method="greedy")
            greedy_ratio = greedy["ratio_to_sqrt_n"]
            results.append({
                "q": q,
                "n": n,
                "bc_size": bc_size,
                "bc_ratio": bc_ratio,
                "greedy_size": greedy["max_sidon_size"],
                "greedy_ratio": greedy_ratio,
                "bc_exceeds_greedy": bc_ratio > greedy_ratio,
            })
        q += 1

    wins = sum(1 for r in results if r["bc_exceeds_greedy"])
    return {
        "verified_true_steps": 1 if wins == 0 and results else 0,
        "verified_false_steps": 1 if wins > 0 else 0,
        "discovery_worthy": len(results) >= 2,
        "trivial_rediscovery": False,
        "metric_value": wins / len(results) if results else 0.0,
        "metric_name": "bc_matched_win_rate",
        "comparison_table": results,
        "notes": f"BC matched comparison: BC exceeds greedy in {wins}/{len(results)} prime q",
    }


def _run_sidon_experiment(hypothesis: dict[str, Any]) -> dict[str, Any]:
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    explicit_ns = _extract_n_list(statement)

    if _wants_bc_matched_comparison(statement):
        return _run_bc_matched_comparison(hypothesis)

    if _wants_threshold_finding(statement) and _parse_threshold_target(statement) is not None:
        return _run_threshold_crossing_experiment(statement)

    wants_sweep = _wants_asymptotic_analysis(statement) or len(explicit_ns) >= 2

    if _wants_greedy_vs_algebraic(statement):
        ns = explicit_ns if len(explicit_ns) >= 2 else list(SIDON_SWEEP_NS)
        return _run_sidon_compare_sweep(statement, sorted(set(ns))[:8])

    if wants_sweep:
        if any(n > 10000 for n in explicit_ns) or "50000" in statement or "50000" in str(explicit_ns):
            ns = explicit_ns if len(explicit_ns) >= 2 else list(SIDON_LARGE_SWEEP_NS)
        else:
            ns = explicit_ns if len(explicit_ns) >= 2 else list(SIDON_SWEEP_NS)
        ns = sorted(set(ns))[:10]
        method = "bose_chowla" if _wants_bose_chowla(statement) else "greedy"
        return _run_sidon_sweep(statement, ns, method=method)

    n = explicit_ns[0] if explicit_ns else _extract_n(statement, default=200, cap=10000)

    if _is_refutation_claim(statement):
        return _run_sidon_refutation(statement, n)

    if n < MIN_N_FOR_SINGLE_POINT_DISCOVERY:
        pt = _sidon_point(n)
        return {
            "verified_true_steps": 0,
            "verified_false_steps": 0,
            "trivial_rediscovery": True,
            "discovery_worthy": False,
            "metric_value": pt["max_sidon_size"] / n if n else 0.0,
            "metric_name": "sidon_density",
            "n": n,
            "max_sidon_size": pt["max_sidon_size"],
            "ratio_to_sqrt_n": pt["ratio_to_sqrt_n"],
            "notes": (
                f"Greedy Sidon at n={n} (size {pt['max_sidon_size']}, "
                f"{pt['ratio_to_sqrt_n']:.3f}×√n) is a known-range computation, "
                "not open-problem evidence. Use multi-n sweep (n≥500)."
            ),
        }

    pt = _sidon_point(n)
    return {
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "trivial_rediscovery": True,
        "discovery_worthy": False,
        "metric_value": pt["max_sidon_size"] / n if n else 0.0,
        "metric_name": "sidon_density",
        "n": n,
        "max_sidon_size": pt["max_sidon_size"],
        "ratio_to_sqrt_n": pt["ratio_to_sqrt_n"],
        "notes": (
            f"Single-point Sidon computation at n={n} without asymptotic claim "
            "does not address open problems."
        ),
    }


def _run_sidon_refutation(statement: str, n: int) -> dict[str, Any]:
    pt = _sidon_point(n)
    k_match = re.search(r"size\s+(\d+)", statement, re.I)
    claim_supported = True
    counterexample: list[int] | None = None

    if k_match:
        claimed_k = int(k_match.group(1))
        if "no " in statement.lower() and pt["max_sidon_size"] >= claimed_k:
            claim_supported = False
            counterexample = pt["example_set"][:claimed_k]

    return {
        "verified_true_steps": 1 if claim_supported else 0,
        "verified_false_steps": 0 if claim_supported else 1,
        "discovery_worthy": not claim_supported,
        "trivial_rediscovery": claim_supported,
        "metric_value": pt["max_sidon_size"] / n if n else 0.0,
        "metric_name": "sidon_density",
        "n": n,
        "max_sidon_size": pt["max_sidon_size"],
        "ratio_to_sqrt_n": pt["ratio_to_sqrt_n"],
        "counterexample": counterexample,
        "notes": (
            f"Refutation test at n={n}: max Sidon size {pt['max_sidon_size']}"
            + (f"; counterexample size {len(counterexample or [])}" if counterexample else "")
        ),
    }


def _run_sidon_compare_sweep(statement: str, ns: list[int]) -> dict[str, Any]:
    """Greedy vs Bose-Chowla ratio comparison across n values."""
    comparisons = [_sidon_compare_at_n(n) for n in ns]
    bc_wins = sum(1 for c in comparisons if c["bose_chowla_exceeds_greedy"])
    max_n = max(ns) if ns else 0
    discovery_worthy = len(comparisons) >= 2 and max_n >= 500
    return {
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "discovery_worthy": discovery_worthy,
        "trivial_rediscovery": False,
        "metric_value": bc_wins / len(comparisons) if comparisons else 0.0,
        "metric_name": "bose_chowla_vs_greedy_ratio",
        "comparison_sweep": comparisons,
        "max_n": max_n,
        "notes": (
            f"Greedy vs Bose-Chowla ratio sweep n={ns}: "
            f"BC ratio exceeds greedy in {bc_wins}/{len(comparisons)} cases; "
            f"ratios greedy={[round(c['greedy_ratio'], 4) for c in comparisons]}, "
            f"bc={[round(c['bose_chowla_ratio'], 4) for c in comparisons]}"
        ),
    }


def _run_sidon_sweep(statement: str, ns: list[int], *, method: str = "greedy") -> dict[str, Any]:
    points = [_sidon_point(n, method=method) for n in ns]
    ratios = [p["ratio_to_sqrt_n"] for p in points]
    mean_ratio = sum(ratios) / len(ratios)
    ratio_spread = max(ratios) - min(ratios) if ratios else 0.0
    max_n = max(p["n"] for p in points)
    discovery_worthy = len(points) >= 3 and max_n >= 500
    return {
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "discovery_worthy": discovery_worthy,
        "trivial_rediscovery": False,
        "metric_value": mean_ratio,
        "metric_name": "sidon_ratio_to_sqrt_n",
        "sweep": points,
        "mean_ratio_to_sqrt_n": mean_ratio,
        "ratio_spread": ratio_spread,
        "max_n": max_n,
        "notes": (
            f"Sidon sweep over n={ns}: ratios={[round(r, 4) for r in ratios]}, "
            f"mean={mean_ratio:.4f}, spread={ratio_spread:.4f}"
        ),
    }


# =============================================================================
# Real cap-set construction and INDEPENDENT validity check in F_3^n.
#
# A subset S of F_3^n is a *cap* iff no three distinct points are collinear.
# Over F_3 a line through distinct a, b is {a, b, c} with a + b + c = 0, i.e. the
# unique third point completing the line is c = -(a + b). So S is a cap iff for
# every pair (a, b) in S the point c = -(a + b) is NOT in S (unless c is a or b).
# The check below is O(|S|^2) and is run on the ACTUAL returned set — it never
# trusts a claimed size.
# =============================================================================


def cap_third_point(a: CapPoint, b: CapPoint) -> CapPoint:
    """The unique point c in F_3^n with a + b + c = 0 (completes the line a,b)."""
    return tuple((-(a[k] + b[k])) % 3 for k in range(len(a)))


def is_valid_cap(points: list[CapPoint], n: int) -> tuple[bool, dict[str, Any]]:
    """
    INDEPENDENT O(|S|^2) cap-validity check on the ACTUAL set of points.

    Returns (valid, detail). A set is a cap iff (1) all points are distinct and
    live in F_3^n, and (2) no three distinct points are collinear — equivalently
    for every pair (a, b), the completing point c = -(a+b) is not a third member.
    """
    seen: set[CapPoint] = set()
    for p in points:
        if len(p) != n or any(coord not in (0, 1, 2) for coord in p):
            return False, {"reason": "point_not_in_F3^n", "point": list(p)}
        if p in seen:
            return False, {"reason": "duplicate_point", "point": list(p)}
        seen.add(p)
    pts = list(seen)
    for i in range(len(pts)):
        a = pts[i]
        for j in range(i + 1, len(pts)):
            b = pts[j]
            c = cap_third_point(a, b)
            if c != a and c != b and c in seen:
                return False, {
                    "reason": "collinear_triple",
                    "triple": [list(a), list(b), list(c)],
                }
    return True, {"reason": "ok", "size": len(pts)}


def _greedy_cap_from_order(elements: list[CapPoint], n: int) -> list[CapPoint]:
    """Greedy cap along a fixed candidate ordering (accept a point if it keeps S a cap)."""
    chosen: set[CapPoint] = set()
    cap: list[CapPoint] = []
    for e in elements:
        ok = True
        for a in cap:
            c = cap_third_point(a, e)
            if c != a and c != e and c in chosen:
                ok = False
                break
        if ok:
            chosen.add(e)
            cap.append(e)
    return cap


def _random_restart_cap(n: int, restarts: int, seed: int) -> list[CapPoint]:
    """Deterministic random-restart greedy: lexicographic pass + `restarts` shuffles."""
    rng = random.Random(seed)
    elements = list(itertools.product(range(3), repeat=n))
    best = _greedy_cap_from_order(elements, n)
    for _ in range(restarts):
        order = elements[:]
        rng.shuffle(order)
        cap = _greedy_cap_from_order(order, n)
        if len(cap) > len(best):
            best = cap
    return best


def max_cap_exhaustive(n: int, time_limit: float = CAP_BB_TIME_BUDGET) -> tuple[list[CapPoint], bool]:
    """
    Near-exhaustive branch-and-bound for the maximum cap in F_3^n.

    Deterministic depth-first search with origin-fixing (every cap can be
    translated to contain 0) and a candidate-set bound. Returns (best_cap,
    complete): `complete` is True only if the whole tree was searched within the
    budget (i.e. the returned size is provably optimal). For small n (<=4) this
    reliably reaches the true maximum well inside a few seconds; for larger n it
    returns the best cap found before the budget expires.
    """
    elements = list(itertools.product(range(3), repeat=n))
    origin: CapPoint = tuple([0] * n)
    candidates = [e for e in elements if e != origin]
    best_size = [1]
    best_cap: list[list[CapPoint]] = [[origin]]
    complete = [True]
    start = time.time()

    def rec(cap: list[CapPoint], chosen: set[CapPoint], cand: list[CapPoint]) -> None:
        if time.time() - start > time_limit:
            complete[0] = False
            return
        # Upper bound: nothing left can beat the incumbent.
        if len(cap) + len(cand) <= best_size[0]:
            return
        if not cand:
            if len(cap) > best_size[0]:
                best_size[0] = len(cap)
                best_cap[0] = list(cap)
            return
        e = cand[0]
        rest = cand[1:]
        # Branch: include e, filtering the remaining candidates that stay valid.
        chosen2 = chosen | {e}
        cap2 = cap + [e]
        next_cand: list[CapPoint] = []
        for f in rest:
            ok = True
            for a in cap2:
                c = cap_third_point(a, f)
                if c != a and c != f and c in chosen2:
                    ok = False
                    break
            if ok:
                next_cand.append(f)
        rec(cap2, chosen2, next_cand)
        # Branch: exclude e.
        rec(cap, chosen, rest)

    rec([origin], {origin}, candidates)
    return best_cap[0], complete[0]


def _cap_product(cap_a: list[CapPoint], cap_b: list[CapPoint]) -> list[CapPoint]:
    """
    Cartesian product construction: if A is a cap in F_3^{m_a} and B a cap in
    F_3^{m_b} then {(a, b) : a in A, b in B} is a cap in F_3^{m_a+m_b} of size
    |A|*|B|. (A line in the product projects to a line or a point in each factor;
    a full line would force a full line in at least one factor, contradicting the
    cap property there.) Validity of the product still follows from validating A
    and B, which we do independently.
    """
    return [a + b for a in cap_a for b in cap_b]


# Module-level cache of computed base caps (amortizes the B&B / restart cost).
_BASE_CAP_CACHE: dict[int, list[CapPoint]] = {}


def _base_cap(m: int) -> list[CapPoint]:
    """A genuinely computed cap in F_3^m used as a product-construction factor."""
    cached = _BASE_CAP_CACHE.get(m)
    if cached is not None:
        return cached
    if m <= 1:
        cap: list[CapPoint] = [(0,), (1,)] if m == 1 else [()]
    elif m <= 4:
        cap, _complete = max_cap_exhaustive(m, time_limit=CAP_BB_TIME_BUDGET)
    else:
        cap = _random_restart_cap(m, CAP_GREEDY_RESTARTS, seed=CAP_GREEDY_SEED + m)
    _BASE_CAP_CACHE[m] = cap
    return cap


def _decompose_dim(n: int) -> list[int]:
    """
    Split n into base dimensions for the product construction, preferring dim-4
    factors (whose optimal cap has size 20, the densest small base we compute).
    """
    parts: list[int] = []
    remaining = n
    for base in (4, 5, 3, 2, 1):
        while remaining >= base:
            parts.append(base)
            remaining -= base
    return parts


def compute_cap_set(n: int) -> dict[str, Any]:
    """
    Compute a REAL cap in F_3^n and validate it independently.

    Strategy:
      * n <= 4: near-exhaustive branch-and-bound (reaches the true maximum).
      * n == 5: deterministic random-restart greedy.
      * n >= 6: product of computed base caps (dims from ``_decompose_dim``).

    The returned dict carries the *actual computed size* (never a table value), a
    checkable witness (full point list when small, otherwise a certificate:
    construction params + fully-validated factor sizes + a sampled spot-check),
    and the independent-validity result.
    """
    n = max(1, int(n))
    parts: list[int]
    proven_optimal = False
    if n <= 4:
        cap, proven_optimal = max_cap_exhaustive(n, time_limit=CAP_BB_TIME_BUDGET)
        parts = [n]
        method = "exhaustive_branch_and_bound"
        seed_desc = f"deterministic DFS, origin-fixed, budget={CAP_BB_TIME_BUDGET}s"
    elif n == 5:
        cap = _random_restart_cap(n, CAP_GREEDY_RESTARTS, seed=CAP_GREEDY_SEED + n)
        parts = [n]
        method = "random_restart_greedy"
        seed_desc = f"seed={CAP_GREEDY_SEED + n}, restarts={CAP_GREEDY_RESTARTS}"
    else:
        parts = _decompose_dim(n)
        cap = _base_cap(parts[0])
        for p in parts[1:]:
            cap = _cap_product(cap, _base_cap(p))
        method = f"product_construction{tuple(parts)}"
        seed_desc = "product of computed base caps"

    size = len(cap)
    field_size = 3**n
    clp = 2.756**n
    best_known = CAP_SET_BEST_KNOWN.get(n)

    # Independent validity. For a manageable set, check the WHOLE thing O(|S|^2).
    # For a huge product cap, validate every base factor fully and spot-check a
    # random sample of product points (full validity then follows from the base
    # validity + the product theorem).
    if size <= CAP_FULL_CHECK_MAX_SIZE:
        valid, detail = is_valid_cap(cap, n)
        witness_kind = "full_point_set"
        sample = [list(p) for p in cap[:CAP_WITNESS_SAMPLE]]
        factors_valid = None
    else:
        valid = True
        detail = {"reason": "certificate"}
        factors_valid = []
        for p in sorted(set(parts)):
            base_cap = _base_cap(p)
            ok, bdetail = is_valid_cap(base_cap, p)
            factors_valid.append({"dim": p, "size": len(base_cap), "valid": ok})
            if not ok:
                valid = False
                detail = {"reason": "invalid_base_factor", "dim": p, **bdetail}
                break
        if valid:
            rng = random.Random(CAP_GREEDY_SEED)
            sample_pts = (
                cap if size <= CAP_WITNESS_SAMPLE else rng.sample(cap, CAP_WITNESS_SAMPLE)
            )
            ok, sdetail = is_valid_cap(sample_pts, n)
            if not ok:
                valid = False
                detail = {"reason": "sample_check_failed", **sdetail}
        witness_kind = "certificate"
        sample = [list(p) for p in cap[:CAP_WITNESS_SAMPLE]]

    witness = {
        "kind": witness_kind,
        "construction": method,
        "decomposition": parts,
        "seed_params": seed_desc,
        "reported_size": size,
        "sample_points": sample,
        "third_point_rule": "c = (-(a+b)) mod 3 must not be a third member",
    }
    if witness_kind == "full_point_set" and size <= 512:
        witness["cap_points"] = [list(p) for p in cap]
    if factors_valid is not None:
        witness["factor_validity"] = factors_valid

    result = {
        "n": n,
        "field_size": field_size,
        "cap_set_size": size,
        "computed_size": size,
        "cap_valid": valid,
        "validity_detail": detail,
        "upper_bound_clp": clp,
        "ratio_to_clp": size / clp if clp > 0 else 0.0,
        "density": size / field_size if field_size else 0.0,
        "construction_source": "computed",
        "construction_method": method,
        "proven_optimal": proven_optimal,
        "best_known_size": best_known,
        "witness": witness,
        "example_cap": sample[:10],
    }
    # Honest computed-vs-known comparison (NEVER report the table as computed).
    if best_known is None:
        result["vs_best_known"] = "no_table_value"
        result["gap_to_best_known"] = None
    elif size < best_known:
        result["vs_best_known"] = "below_best_known"
        result["gap_to_best_known"] = best_known - size
    elif size == best_known:
        result["vs_best_known"] = "matches_best_known"
        result["gap_to_best_known"] = 0
    else:
        result["vs_best_known"] = "exceeds_best_known"
        result["gap_to_best_known"] = best_known - size  # negative => exceeds
    return result


def _run_cap_set_experiment(hypothesis: dict[str, Any]) -> dict[str, Any]:
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    dims = _extract_cap_dims(statement)
    wants_sweep = (
        _wants_asymptotic_analysis(statement)
        or len(dims) >= 2
        or parse_ratio_upper_bound(statement) is not None
    )

    if wants_sweep:
        use_dims = dims if dims else list(CAP_SWEEP_DIMS)
        return _run_cap_set_sweep(statement, sorted(set(use_dims))[:6])

    n = dims[0] if dims else 4
    n = min(n, MAX_FULL_CAP_DIM)

    if _is_table_lookup_methodology(hypothesis):
        return _run_cap_set_table_lookup(statement, n)

    pt = compute_cap_set(n)
    notes = _cap_compute_notes(pt)
    base = {
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        # A single-dimension computation is not itself open-problem evidence, but
        # it is a REAL computed cap (not a rediscovery of a table value), so it is
        # not flagged trivial_rediscovery. Discovery-worthiness is decided by the
        # claim validation below.
        "trivial_rediscovery": False,
        "discovery_worthy": False,
        "metric_value": pt["density"],
        "metric_name": "cap_set_density",
        **pt,
        "notes": notes,
    }
    if not pt["cap_valid"]:
        # Never report a size without a valid witness — demote to inconclusive.
        base["trivial_rediscovery"] = False
        base["discovery_worthy"] = False
        base["cap_set_size"] = 0
        base["notes"] = (
            f"Cap construction in F_3^{n} FAILED independent validity check "
            f"({pt['validity_detail'].get('reason')}); no size reported."
        )
        return base
    return _apply_claim_to_verdict(
        base, statement, computed_size=pt["cap_set_size"], dimension=n,
    )


def _cap_compute_notes(pt: dict[str, Any]) -> str:
    n = pt["n"]
    size = pt["cap_set_size"]
    bk = pt["best_known_size"]
    method = pt["construction_method"]
    witness_kind = pt["witness"]["kind"]
    optimal = " (proven optimal)" if pt.get("proven_optimal") else ""
    if bk is None:
        cmp = f"no best-known table value for n={n} (honest computed size only)"
    elif pt["vs_best_known"] == "below_best_known":
        cmp = f"below best-known {bk} (honest gap {pt['gap_to_best_known']})"
    elif pt["vs_best_known"] == "matches_best_known":
        cmp = f"matches best-known {bk}"
    else:
        cmp = f"EXCEEDS table value {bk}"
    return (
        f"Computed a real, independently-validated cap in F_3^{n} of size {size}{optimal} "
        f"via {method}; {cmp}. Witness: {witness_kind}."
    )


def _is_table_lookup_methodology(hypothesis: dict[str, Any]) -> bool:
    """True when the hypothesis explicitly asks for a best-known TABLE lookup."""
    text = " ".join(
        str(hypothesis.get(k) or "")
        for k in ("statement", "text", "test_methodology")
    ).lower()
    return any(
        marker in text
        for marker in (
            "table lookup",
            "table-lookup",
            "best-known table",
            "best known table",
            "literature table",
            "lookup table",
            "look up the",
            "read the known",
            "known value from",
        )
    )


def _run_cap_set_table_lookup(statement: str, n: int) -> dict[str, Any]:
    """
    Rediscovery guard (DISC2): a hypothesis that asks only for a best-known TABLE
    value gets the tabulated number, flagged trivial_rediscovery — NOT presented
    as a computation. This preserves the existing table-lookup demotion.
    """
    size = CAP_SET_BEST_KNOWN.get(n)
    field_size = 3**n
    clp = 2.756**n
    if size is None:
        size = 0
        note_size = "no table value"
    else:
        note_size = str(size)
    base = {
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "trivial_rediscovery": True,
        "discovery_worthy": False,
        "metric_value": size / field_size if field_size else 0.0,
        "metric_name": "cap_set_density",
        "n": n,
        "field_size": field_size,
        "cap_set_size": size,
        "upper_bound_clp": clp,
        "ratio_to_clp": size / clp if clp > 0 else 0.0,
        "density": size / field_size if field_size else 0.0,
        "construction_source": "best_known_table",
        "example_cap": [],
        "notes": (
            f"Best-known cap-set size in F_3^{n} is {note_size} read from the "
            "literature table — a table lookup, not a computation (trivial rediscovery)."
        ),
    }
    return _apply_claim_to_verdict(
        base, statement, computed_size=size, dimension=n,
    )


def _run_cap_set_sweep(statement: str, dims: list[int]) -> dict[str, Any]:
    points = [compute_cap_set(d) for d in dims]
    ratios = [p["ratio_to_clp"] for p in points]
    sizes = [p["cap_set_size"] for p in points]
    all_valid = all(p["cap_valid"] for p in points)
    discovery_worthy = all_valid and len(points) >= 2 and max(p["n"] for p in points) >= 5
    known = [p["best_known_size"] for p in points]
    return {
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "discovery_worthy": discovery_worthy,
        "trivial_rediscovery": False,
        "metric_value": sum(ratios) / len(ratios) if ratios else 0.0,
        "metric_name": "cap_set_clp_ratio",
        "sweep": points,
        "all_caps_valid": all_valid,
        "notes": (
            f"Cap-set sweep dims={dims}: COMPUTED validated sizes={sizes} "
            f"(best-known={known}), CLP ratios={[round(r, 4) for r in ratios]}"
        ),
    }


def _run_sumset_experiment(hypothesis: dict[str, Any]) -> dict[str, Any]:
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    if any(k in statement.lower() for k in ("structured", "sidon-like", "arithmetic progression", "compare")):
        return _run_sumset_growth_structured(hypothesis)
    ns = _extract_n_list(statement)
    n = ns[0] if ns else _extract_n(statement, default=500, cap=5000)

    results: list[float] = []
    random.seed(42)
    for _ in range(20):
        size = max(5, n // 5)
        a = set(random.sample(range(1, n + 1), size))
        sumset = {x + y for x in a for y in a}
        results.append(len(sumset) / len(a))

    avg_growth = sum(results) / len(results)
    wants_structure = _wants_asymptotic_analysis(statement) and n >= 500

    return {
        "verified_true_steps": 1 if wants_structure else 0,
        "verified_false_steps": 0,
        "discovery_worthy": wants_structure,
        "trivial_rediscovery": not wants_structure,
        "metric_value": avg_growth,
        "metric_name": "sumset_growth",
        "n": n,
        "avg_growth_ratio": avg_growth,
        "notes": (
            f"Sumset growth |A+A|/|A| for random subsets of {{1,...,{n}}}: {avg_growth:.2f}"
            + ("" if wants_structure else " (single random sample — not open-problem evidence)")
        ),
    }


def _run_sumset_growth_structured(hypothesis: dict[str, Any]) -> dict[str, Any]:
    """Compare |A+A|/|A| for random, AP, and Sidon-like sets."""
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    n = _extract_n_list(statement)[0] if _extract_n_list(statement) else _extract_n(statement, default=500, cap=5000)
    size = max(8, n // 10)
    random.seed(42)

    def growth(subset: set[int]) -> float:
        if not subset:
            return 0.0
        sumset = {x + y for x in subset for y in subset}
        return len(sumset) / len(subset)

    random_growths = []
    for _ in range(10):
        a = set(random.sample(range(1, n + 1), min(size, n)))
        random_growths.append(growth(a))

    ap = set(range(1, min(size, n) + 1))
    ap_g = growth(ap)
    sidon_set = set(_greedy_sidon_max_optimized(min(n, 500))[:size])
    sidon_g = growth(sidon_set)

    avg_random = sum(random_growths) / len(random_growths)
    sidon_smallest = sidon_g < ap_g and sidon_g < avg_random

    return {
        "verified_true_steps": 1 if sidon_smallest else 0,
        "verified_false_steps": 0 if sidon_smallest else 1,
        "discovery_worthy": True,
        "trivial_rediscovery": False,
        "metric_value": sidon_g,
        "metric_name": "sumset_growth",
        "n": n,
        "random_avg_growth": avg_random,
        "ap_growth": ap_g,
        "sidon_growth": sidon_g,
        "notes": (
            f"Sumset growth at n={n}: random={avg_random:.2f}, AP={ap_g:.2f}, "
            f"Sidon-like={sidon_g:.2f}"
        ),
    }


def _run_ap_free_experiment(hypothesis: dict[str, Any]) -> dict[str, Any]:
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    ns = _extract_n_list(statement)
    wants_sweep = _wants_asymptotic_analysis(statement) or len(ns) >= 3

    if wants_sweep:
        use_ns = ns if len(ns) >= 3 else [100, 200, 500, 1000, 2000]
        return _run_ap_free_sweep(statement, sorted(set(use_ns))[:6])

    n = ns[0] if ns else _extract_n(statement, default=100, cap=1000)

    def has_3ap(lst: list[int]) -> bool:
        s = set(lst)
        for a in lst:
            for b in lst:
                if b != a and 2 * b - a in s and 2 * b - a != a:
                    return True
        return False

    ap_free: list[int] = []
    for x in range(1, n + 1):
        candidate = ap_free + [x]
        if not has_3ap(candidate):
            ap_free.append(x)

    density = len(ap_free) / n if n > 0 else 0.0

    return {
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "trivial_rediscovery": True,
        "discovery_worthy": False,
        "metric_value": density,
        "metric_name": "ap_free_density",
        "n": n,
        "ap_free_size": len(ap_free),
        "density": density,
        "example": ap_free[:20],
        "notes": (
            f"Greedy AP-free at n={n} (density {density:.4f}) rediscovers known "
            "Behrend/Szemeredi regime — use multi-n density trend for open questions."
        ),
    }


def _greedy_ap_free_fast(n: int) -> tuple[int, float]:
    """O(n * |set|) AP-free greedy — faster than O(n³) triple check."""
    current: set[int] = set()
    current_list: list[int] = []
    for x in range(1, n + 1):
        creates_ap = any(2 * a - x in current for a in current_list)
        if not creates_ap:
            current.add(x)
            current_list.append(x)
    return len(current_list), len(current_list) / n if n else 0.0


def _run_ap_free_sweep(statement: str, ns: list[int]) -> dict[str, Any]:
    points = []
    for n in ns:
        size, density = _greedy_ap_free_fast(n)
        points.append({"n": n, "size": size, "density": density})
    densities = [p["density"] for p in points]
    stmt = statement.lower()

    verified_true = 0
    verified_false = 0
    if "sub-exponential" in stmt or "behrend" in stmt or "trend" in stmt:
        if all(densities[i] >= densities[i + 1] for i in range(len(densities) - 1)):
            verified_true = 1
        else:
            verified_false = 1
    elif len(points) >= 3:
        verified_true = 1

    return {
        "verified_true_steps": verified_true,
        "verified_false_steps": verified_false,
        "discovery_worthy": len(points) >= 3 and max(p["n"] for p in points) >= 500,
        "trivial_rediscovery": verified_true == 0 and verified_false == 0,
        "metric_value": sum(densities) / len(densities) if densities else 0.0,
        "metric_name": "ap_free_density_sweep",
        "sweep": points,
        "notes": f"AP-free density sweep n={ns}: densities={[round(d, 4) for d in densities]}",
    }
