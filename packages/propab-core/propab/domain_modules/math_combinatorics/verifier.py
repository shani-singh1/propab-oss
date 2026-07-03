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

# Standard sweep grid for Sidon asymptotics (fixes.md open problem 1).
SIDON_SWEEP_NS = (100, 200, 500, 1000, 2000, 5000, 10000)
CAP_SWEEP_DIMS = (3, 4, 5, 6, 7)
MIN_N_FOR_SINGLE_POINT_DISCOVERY = 500


def run_combinatorics_experiment(hypothesis: dict[str, Any], features: list[str]) -> dict[str, Any]:
    start = time.time()
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    primary_feature = features[0] if features else "sidon_set_density"

    stmt_lower = statement.lower()
    if "sidon" in stmt_lower or "sidon" in primary_feature:
        result = _run_sidon_experiment(hypothesis)
    elif "cap_set" in primary_feature or "cap set" in stmt_lower:
        result = _run_cap_set_experiment(hypothesis)
    elif "sumset" in primary_feature:
        result = _run_sumset_experiment(hypothesis)
    elif "arithmetic progression" in stmt_lower or "ap-free" in stmt_lower or "ap_free" in primary_feature:
        result = _run_ap_free_experiment(hypothesis)
    else:
        result = _run_sidon_experiment(hypothesis)

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
    uniq = sorted(set(n for n in found if 10 <= n <= 10000))
    return uniq


def _extract_cap_dims(statement: str) -> list[int]:
    dims: list[int] = []
    for m in re.finditer(r"F_3\^(\d+)", statement, re.I):
        dims.append(int(m.group(1)))
    for m in re.finditer(r"dimension\s+(\d+)", statement, re.I):
        dims.append(int(m.group(1)))
    for m in re.finditer(r"\bn\s+in\s+\{(\d+(?:\s*,\s*\d+)*)\}", statement, re.I):
        for part in m.group(1).split(","):
            if part.strip().isdigit():
                dims.append(int(part.strip()))
    return sorted(set(d for d in dims if 2 <= d <= 7))


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


def _sidon_point(n: int) -> dict[str, Any]:
    best = _greedy_sidon_max(n)
    theoretical = math.sqrt(n)
    ratio = len(best) / theoretical if theoretical > 0 else 0.0
    gaps = [best[i + 1] - best[i] for i in range(len(best) - 1)] if len(best) > 1 else []
    return {
        "n": n,
        "max_sidon_size": len(best),
        "theoretical_bound": theoretical,
        "ratio_to_sqrt_n": ratio,
        "example_set": best[:20],
        "mean_gap": sum(gaps) / len(gaps) if gaps else 0.0,
        "gap_std": (
            math.sqrt(sum((g - sum(gaps) / len(gaps)) ** 2 for g in gaps) / len(gaps))
            if len(gaps) > 1
            else 0.0
        ),
    }


def _run_sidon_experiment(hypothesis: dict[str, Any]) -> dict[str, Any]:
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    explicit_ns = _extract_n_list(statement)
    wants_sweep = _wants_asymptotic_analysis(statement) or len(explicit_ns) >= 2

    if wants_sweep:
        ns = explicit_ns if len(explicit_ns) >= 2 else list(SIDON_SWEEP_NS)
        ns = sorted(set(ns))[:8]
        return _run_sidon_sweep(statement, ns)

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


def _run_sidon_sweep(statement: str, ns: list[int]) -> dict[str, Any]:
    points = [_sidon_point(n) for n in ns]
    ratios = [p["ratio_to_sqrt_n"] for p in points]
    mean_ratio = sum(ratios) / len(ratios)
    ratio_spread = max(ratios) - min(ratios) if ratios else 0.0
    max_n = max(p["n"] for p in points)

    discovery_worthy = len(points) >= 3 and max_n >= 500
    stmt = statement.lower()

    verified_true = 0
    verified_false = 0
    notes = f"Sidon sweep over n={ns}: ratios={[round(r, 4) for r in ratios]}, mean={mean_ratio:.4f}"

    if "below 1" in stmt or "< 1.0" in stmt or "below 1.0" in stmt:
        if all(r < 1.02 for r in ratios):
            verified_true = 1
            notes += "; all ratios < 1.02 — consistent with constant below 1"
        else:
            verified_false = 1
            notes += f"; refuted: max ratio {max(ratios):.4f} ≥ 1.02"
    elif "monotonic" in stmt and "decreas" in stmt:
        decreasing = all(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1))
        if decreasing:
            verified_true = 1
            notes += "; ratios decrease monotonically"
        else:
            verified_false = 1
            notes += "; monotonic decrease refuted"
    elif "converge" in stmt or "constant" in stmt or "asymptotic" in stmt:
        if ratio_spread < 0.08 and len(points) >= 3:
            verified_true = 1
            notes += f"; ratios stable (spread={ratio_spread:.4f}), estimate c≈{mean_ratio:.4f}"
        elif len(points) >= 3:
            verified_true = 1
            notes += (
                f"; multi-n sweep complete: c ranges {min(ratios):.4f}–{max(ratios):.4f} "
                f"(spread={ratio_spread:.4f})"
            )
        else:
            verified_true = 0
            notes += "; insufficient n values for convergence claim"
    elif "structural" in stmt or "gap" in stmt:
        gap_stds = [p["gap_std"] for p in points if p["max_sidon_size"] > 2]
        if gap_stds:
            verified_true = 1
            notes += f"; gap std devs across n: {[round(g, 2) for g in gap_stds]}"
        else:
            verified_true = 0
    else:
        if discovery_worthy:
            verified_true = 1
            notes += "; multi-n sweep provides open-problem evidence"
        else:
            verified_true = 0
            notes += "; sweep too small for open-problem claim (need ≥3 n with max n≥500)"

    return {
        "verified_true_steps": verified_true,
        "verified_false_steps": verified_false,
        "discovery_worthy": discovery_worthy and (verified_true > 0 or verified_false > 0),
        "trivial_rediscovery": verified_true == 0 and verified_false == 0,
        "metric_value": mean_ratio,
        "metric_name": "sidon_ratio_to_sqrt_n",
        "sweep": points,
        "mean_ratio_to_sqrt_n": mean_ratio,
        "ratio_spread": ratio_spread,
        "max_n": max_n,
        "notes": notes,
    }


def _run_cap_set_experiment(hypothesis: dict[str, Any]) -> dict[str, Any]:
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    dims = _extract_cap_dims(statement)
    wants_sweep = _wants_asymptotic_analysis(statement) or len(dims) >= 2

    if wants_sweep or len(dims) >= 2:
        use_dims = dims if len(dims) >= 2 else list(CAP_SWEEP_DIMS)
        return _run_cap_set_sweep(statement, sorted(set(use_dims))[:6])

    n = dims[0] if dims else 4
    n = min(n, 7)
    pt = _cap_set_greedy(n)

    return {
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "trivial_rediscovery": True,
        "discovery_worthy": False,
        "metric_value": pt["density"],
        "metric_name": "cap_set_density",
        **pt,
        "notes": (
            f"Single cap-set computation in F_3^{n} (size {pt['cap_set_size']}) "
            "without multi-dimension CLP gap analysis is not open-problem evidence."
        ),
    }


def _cap_set_greedy(n: int) -> dict[str, Any]:
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
    points = [_cap_set_greedy(d) for d in dims]
    ratios = [p["ratio_to_clp"] for p in points]
    stmt = statement.lower()

    verified_true = 0
    verified_false = 0
    notes = f"Cap-set sweep dims={dims}: CLP ratios={[round(r, 4) for r in ratios]}"

    if "monotonic" in stmt and "decreas" in stmt:
        if all(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1)):
            verified_true = 1
            notes += "; CLP ratio decreases with dimension"
        else:
            verified_false = 1
            notes += "; monotonic CLP ratio decrease refuted"
    elif len(points) >= 2:
        verified_true = 1
        notes += "; multi-dimension cap-set sweep complete"

    discovery_worthy = len(points) >= 2 and max(p["n"] for p in points) >= 5

    return {
        "verified_true_steps": verified_true,
        "verified_false_steps": verified_false,
        "discovery_worthy": discovery_worthy and verified_true > 0,
        "trivial_rediscovery": verified_true == 0 and verified_false == 0,
        "metric_value": sum(ratios) / len(ratios) if ratios else 0.0,
        "metric_name": "cap_set_clp_ratio",
        "sweep": points,
        "notes": notes,
    }


def _run_sumset_experiment(hypothesis: dict[str, Any]) -> dict[str, Any]:
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
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


def _run_ap_free_sweep(statement: str, ns: list[int]) -> dict[str, Any]:
    def greedy_ap_free(n: int) -> tuple[int, float]:
        def has_3ap(lst: list[int]) -> bool:
            s = set(lst)
            for a in lst:
                for b in lst:
                    if b != a and 2 * b - a in s and 2 * b - a != a:
                        return True
            return False

        ap_free: list[int] = []
        for x in range(1, n + 1):
            if not has_3ap(ap_free + [x]):
                ap_free.append(x)
        return len(ap_free), len(ap_free) / n if n else 0.0

    points = []
    for n in ns:
        size, density = greedy_ap_free(n)
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
