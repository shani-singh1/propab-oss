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
    s = statement.lower()
    return any(k in s for k in ("matched", "prime power", "prime q", "q²+q", "q^2+q")) or (
        _wants_bose_chowla(s) and "prime" in s
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

    max_dim = max(CAP_SET_BEST_KNOWN)
    n = dims[0] if dims else 4
    n = min(n, max_dim)
    pt = _cap_set_best_known(n)
    base = {
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "trivial_rediscovery": True,
        "discovery_worthy": False,
        "metric_value": pt["density"],
        "metric_name": "cap_set_density",
        **pt,
        "notes": (
            f"Best-known cap-set size in F_3^{n} is {pt['cap_set_size']} "
            "(literature table, not greedy 2^n product)."
        ),
    }
    return _apply_claim_to_verdict(
        base, statement, computed_size=pt["cap_set_size"], dimension=n,
    )


def _cap_set_best_known(n: int) -> dict[str, Any]:
    """Return best-known cap-set size for F_3^n (not the trivial 2^n product)."""
    size = CAP_SET_BEST_KNOWN.get(n)
    if size is None:
        size = _cap_set_greedy_fallback(n)["cap_set_size"]
        source = "greedy_fallback"
    else:
        source = "best_known_table"
    field_size = 3**n
    clp = 2.756**n
    density = size / field_size if field_size else 0.0
    return {
        "n": n,
        "field_size": field_size,
        "cap_set_size": size,
        "upper_bound_clp": clp,
        "ratio_to_clp": size / clp if clp > 0 else 0.0,
        "density": density,
        "construction_source": source,
        "example_cap": [],
    }


def _cap_set_greedy_fallback(n: int) -> dict[str, Any]:
    """Greedy AP-free search in F_3^n — used only when n exceeds the table."""
    elements = list(itertools.product(range(3), repeat=n))

    def has_ap(a: tuple[int, ...], b: tuple[int, ...], c: tuple[int, ...]) -> bool:
        return all((a[i] + b[i] + c[i]) % 3 == 0 for i in range(n))

    cap: list[tuple[int, ...]] = []
    for elem in elements:
        valid = True
        for i in range(len(cap)):
            for j in range(i + 1, len(cap)):
                if has_ap(cap[i], cap[j], elem):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            cap.append(elem)

    clp = 2.756**n
    density = len(cap) / len(elements) if elements else 0.0
    return {
        "n": n,
        "field_size": len(elements),
        "cap_set_size": len(cap),
        "upper_bound_clp": clp,
        "ratio_to_clp": len(cap) / clp if clp > 0 else 0.0,
        "density": density,
        "example_cap": [list(c) for c in cap[:10]],
    }


def _run_cap_set_sweep(statement: str, dims: list[int]) -> dict[str, Any]:
    points = [_cap_set_best_known(d) for d in dims]
    ratios = [p["ratio_to_clp"] for p in points]
    sizes = [p["cap_set_size"] for p in points]
    discovery_worthy = len(points) >= 2 and max(p["n"] for p in points) >= 5
    return {
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "discovery_worthy": discovery_worthy,
        "trivial_rediscovery": False,
        "metric_value": sum(ratios) / len(ratios) if ratios else 0.0,
        "metric_name": "cap_set_clp_ratio",
        "sweep": points,
        "notes": (
            f"Cap-set sweep dims={dims}: best-known sizes={sizes}, "
            f"CLP ratios={[round(r, 4) for r in ratios]}"
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
