"""Sidon/cap-set constructors and claim parsing for the combinatorics verifier."""
from __future__ import annotations

import math
import re
from typing import Any

# Best-known cap-set sizes in F_3^n (OEIS / literature; n <= 8).
CAP_SET_BEST_KNOWN: dict[int, int] = {
    1: 1,
    2: 2,
    3: 9,
    4: 20,
    5: 45,
    6: 112,
    7: 236,
    8: 512,
}

# Hypothesis keywords requiring statistics the verifier does not compute.
_UNIMPLEMENTED_STATS_RE = re.compile(
    r"|".join(
        (
            r"poisson",
            r"fourier",
            r"chi[\s-]?squared",
            r"decile",
            r"\bmcmc\b",
            r"kolmogorov[\s-]?smirnov",
            r"\bks[\s-]?test\b",
            r"spectral peak",
            r"modular uniformity",
            r"tabu search",
            r"evolutionary algorithm",
            r"stochastic hill",
            r"markov chain",
            r"geometric distribution",
            r"hill[\s-]?climbing",
            r"simulated annealing",
        )
    ),
    re.I,
)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def largest_prime_le(n: int) -> int:
    for p in range(max(2, n), 1, -1):
        if is_prime(p):
            return p
    return 2


def is_sidon_set(values: list[int]) -> bool:
    lst = sorted(set(values))
    sums: set[int] = set()
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            s = lst[i] + lst[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def bose_chowla_sidon(q: int, *, one_indexed: bool = True) -> list[int]:
    """
    Bose-Chowla Sidon set of size q for prime q.

    Elements live in {0, ..., q^2 - 1} (or {1, ...} when one_indexed).
    Construction: S = { (i^2 + i)/2 + i*q : i = 0..q-1 }.
    """
    if not is_prime(q) or q < 2:
        return []
    base = sorted({((i * i + i) // 2 + i * q) for i in range(q)})
    if len(base) != q or not is_sidon_set(base):
        return []
    if one_indexed:
        return [x + 1 for x in base]
    return base


def bose_chowla_ratio_at_n(n: int) -> dict[str, Any]:
    """
    Theoretical Bose-Chowla density at scale n: use largest valid prime q and
    ratio q/sqrt(n). Does not embed into {1,...,n} (which artificially shrinks BC).
    """
    best_q = 2
    for candidate in range(2, int(math.isqrt(n)) + 1):
        if is_prime(candidate) and bose_chowla_sidon(candidate):
            best_q = candidate
    size = best_q
    theoretical = math.sqrt(n) if n > 0 else 1.0
    ratio = size / theoretical if theoretical > 0 else 0.0
    return {
        "n": n,
        "q": best_q,
        "max_sidon_size": size,
        "theoretical_bound": theoretical,
        "ratio_to_sqrt_n": ratio,
        "construction": f"bose_chowla_q={best_q}",
    }


def bose_chowla_for_n(n: int) -> tuple[list[int], int]:
    """Largest prime q with a valid Bose-Chowla set embeddable in {1,...,n}."""
    pt = bose_chowla_ratio_at_n(n)
    q = int(pt["q"])
    raw = bose_chowla_sidon(q, one_indexed=True)
    embedded = [x for x in raw if 1 <= x <= n]
    return embedded, q


def requires_unimplemented_statistics(statement: str) -> bool:
    return bool(_UNIMPLEMENTED_STATS_RE.search(statement or ""))


_SCOPE_SPLIT_RE = re.compile(
    r"(?im)\n(?:Population|Distribution|Claimed generalization|"
    r"Expected failure modes|OOD test)\s*:",
)


def extract_claim_text(
    statement: str = "",
    *,
    test_methodology: str = "",
    full_text: str = "",
) -> str:
    """
    Scientific claim only — exclude scope boilerplate appended to hypothesis text.

    Routing and claim parsing must use this, not enriched scope/population lines.
    """
    raw = (statement or "").strip() or (full_text or "").strip()
    claim = _SCOPE_SPLIT_RE.split(raw, maxsplit=1)[0].strip()
    meth_raw = (test_methodology or "").strip()
    meth_str = ""
    if meth_raw.startswith("{"):
        try:
            import json as _json

            parsed = _json.loads(meth_raw)
            if isinstance(parsed, dict):
                meth_str = str(parsed.get("methodology") or "").strip()
        except (TypeError, ValueError):
            meth_str = ""
    elif meth_raw:
        meth_str = meth_raw
    if meth_str and meth_str not in claim:
        return f"{claim}\n{meth_str}".strip()
    return claim


def is_cap_set_hypothesis(
    statement: str,
    test_methodology: str = "",
    *,
    full_text: str = "",
) -> bool:
    """True when the scientific claim is about cap sets (not scope boilerplate)."""
    claim = extract_claim_text(
        statement,
        test_methodology=test_methodology,
        full_text=full_text or statement,
    )
    s = claim.lower()
    return bool(
        re.search(r"\bcap[\s-]?set\b", s)
        or re.search(r"f_3\^", s)
        or "a_3(" in s
        or "|a_max(" in s
        or "c_n =" in s
        or "c_n=" in s
        or (
            "clp" in s
            and re.search(r"ratio|bound|dim|f_3|cap|a_max|a_3", s)
        )
    )


def is_sidon_hypothesis(
    statement: str,
    test_methodology: str = "",
    *,
    full_text: str = "",
) -> bool:
    claim = extract_claim_text(
        statement,
        test_methodology=test_methodology,
        full_text=full_text or statement,
    )
    s = claim.lower()
    if "ap-free" in s or "ap free" in s or "arithmetic progression-free" in s:
        return False
    if is_cap_set_hypothesis(statement, test_methodology, full_text=full_text):
        return False
    sidon_markers = (
        "sidon",
        "greedy",
        "bose-chowla",
        "bose chowla",
        "f(n)/sqrt",
        "f(n)/√",
        "f_greedy",
        "f_bc",
        "mian-chowla",
        "mian chowla",
    )
    return any(m in s for m in sidon_markers)


def evidence_metric_matches_hypothesis(statement: str, result: dict[str, Any]) -> bool:
    """Type-check: cap-set claims cannot confirm on Sidon-ratio evidence, etc."""
    if requires_unimplemented_statistics(statement):
        return True
    metric = str(result.get("metric_name") or "")
    if is_cap_set_hypothesis(statement):
        return metric.startswith("cap_set")
    return True


def parse_minimum_int_claim(statement: str) -> int | None:
    """Parse 'at least K', '>= K', 'minimum size K' from hypothesis text."""
    s = statement.lower()
    patterns = (
        r"(?:at least|>=|≥|minimum(?:\s+size)?(?:\s+of)?(?:\s+is)?)\s*(\d{2,5})",
        r"size\s+(?:at least|>=)\s*(\d{2,5})",
        r"is\s+at\s+least\s*(\d{2,5})",
        r"\|\s*A_max\(\d+\)\|\s*>=\s*(\d{2,5})",
    )
    for pat in patterns:
        m = re.search(pat, s, re.I)
        if m:
            return int(m.group(1))
    return None


def parse_interval_claim(statement: str) -> tuple[float, float] | None:
    """Parse numeric interval [lo, hi] from ratio/band claims."""
    s = statement.replace("√", "sqrt")
    patterns = (
        r"(?:within|in|stays in|remains in|interval)\s*(?:the\s+)?\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]",
        r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]",
    )
    for pat in patterns:
        m = re.search(pat, s, re.I)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo <= hi:
                # F(n)/sqrt(n) ratios live in ~[0, 2]; larger values are n-ranges.
                if lo > 10 or hi > 10:
                    continue
                return lo, hi
    return None


def parse_ratio_upper_bound(statement: str) -> float | None:
    s = statement.lower()
    if "below 1.0" in s or "below 1" in s or "< 1.0" in s:
        return 1.0
    m = re.search(r"(?:below|under|<\s*)\s*(\d+\.?\d*)", s)
    if m:
        return float(m.group(1))
    return None


def parse_cap_dimension_claim(statement: str) -> tuple[int | None, int | None]:
    """Parse a_3(d) ... at least K → (dimension, min_size)."""
    m = re.search(
        r"a_3\s*\(\s*(\d+)\s*\).*?(?:at least|>=|≥|minimum)\s*(\d+)",
        statement,
        re.I | re.DOTALL,
    )
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = re.search(r"f_3\^(\d+).*?(?:at least|>=)\s*(\d+)", statement, re.I | re.DOTALL)
    if m2:
        return int(m2.group(1)), int(m2.group(2))
    return None, None


def _claim_result(
    *,
    supported: bool,
    notes_suffix: str,
    claim_type: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "claim_checked": True,
        "claim_supported": supported,
        "claim_type": claim_type,
        "notes_suffix": notes_suffix,
        **extra,
    }


def evaluate_numeric_claim(
    statement: str,
    *,
    computed_size: int,
    dimension: int | None = None,
) -> dict[str, Any]:
    """Compare computed size against parsed numeric minimum claim."""
    dim, min_for_dim = parse_cap_dimension_claim(statement)
    if dim is not None and min_for_dim is not None and dimension is not None and dim == dimension:
        threshold = min_for_dim
    else:
        threshold = parse_minimum_int_claim(statement)

    if threshold is None:
        return {"claim_checked": False}

    if computed_size >= threshold:
        return _claim_result(
            supported=True,
            notes_suffix=f" claim >= {threshold} supported (computed {computed_size})",
            claim_type="minimum_size",
            claimed_minimum=threshold,
            computed_size=computed_size,
        )
    return _claim_result(
        supported=False,
        notes_suffix=f" REFUTED: claim >= {threshold} but computed {computed_size}",
        claim_type="minimum_size",
        claimed_minimum=threshold,
        computed_size=computed_size,
    )


def evaluate_ratio_interval_claim(
    statement: str,
    ratios: list[float],
    *,
    tolerance: float = 0.015,
) -> dict[str, Any]:
    """Check all ratios lie in a parsed [lo, hi] band."""
    band = parse_interval_claim(statement)
    if not band or not ratios:
        return {"claim_checked": False}
    lo, hi = band
    if all(lo - tolerance <= r <= hi + tolerance for r in ratios):
        return _claim_result(
            supported=True,
            notes_suffix=f" all ratios in [{lo}, {hi}] (actual {min(ratios):.4f}–{max(ratios):.4f})",
            claim_type="ratio_interval",
            claimed_interval=[lo, hi],
            observed_ratios=ratios,
        )
    return _claim_result(
        supported=False,
        notes_suffix=(
            f" REFUTED: claimed ratios in [{lo}, {hi}] but observed "
            f"{min(ratios):.4f}–{max(ratios):.4f}"
        ),
        claim_type="ratio_interval",
        claimed_interval=[lo, hi],
        observed_ratios=ratios,
    )


def evaluate_ratio_upper_bound_claim(
    statement: str,
    ratios: list[float],
    *,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    bound = parse_ratio_upper_bound(statement)
    if bound is None or not ratios:
        return {"claim_checked": False}
    if all(r < bound + tolerance for r in ratios):
        return _claim_result(
            supported=True,
            notes_suffix=f" all ratios < {bound} (max {max(ratios):.4f})",
            claim_type="ratio_upper_bound",
            claimed_upper=bound,
        )
    return _claim_result(
        supported=False,
        notes_suffix=f" REFUTED: claimed ratios < {bound} but max {max(ratios):.4f}",
        claim_type="ratio_upper_bound",
        claimed_upper=bound,
    )


def evaluate_monotonic_decrease_claim(statement: str, ratios: list[float]) -> dict[str, Any]:
    s = statement.lower()
    if "monotonic" not in s or "decreas" not in s:
        return {"claim_checked": False}
    ok = all(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1))
    if ok:
        return _claim_result(
            supported=True,
            notes_suffix=" ratios decrease monotonically",
            claim_type="monotonic_decrease",
        )
    return _claim_result(
        supported=False,
        notes_suffix=" REFUTED: monotonic decrease fails",
        claim_type="monotonic_decrease",
    )


def evaluate_cap_growth_monotone_claim(
    statement: str,
    sizes_by_dim: list[tuple[int, int]],
) -> dict[str, Any]:
    s = statement.lower()
    if "strictly increasing" not in s and ("monotonic" not in s or "increas" not in s):
        return {"claim_checked": False}
    if len(sizes_by_dim) < 2:
        return {"claim_checked": False}
    rates = [size ** (1.0 / dim) for dim, size in sizes_by_dim if dim > 0 and size > 0]
    ok = all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))
    if ok:
        return _claim_result(
            supported=True,
            notes_suffix=f" c_n strictly increasing: {[round(r, 4) for r in rates]}",
            claim_type="cap_growth_monotone",
        )
    return _claim_result(
        supported=False,
        notes_suffix=f" REFUTED: c_n not strictly increasing: {[round(r, 4) for r in rates]}",
        claim_type="cap_growth_monotone",
    )


def evaluate_bc_vs_greedy_claim(
    statement: str,
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check Bose-Chowla vs greedy ratio claims on comparison sweeps."""
    s = statement.lower()
    if not comparisons:
        return {"claim_checked": False}
    wants_exceed = any(k in s for k in ("exceed", "versus", "vs ", "compare", "larger than", "greater than"))
    if not wants_exceed:
        return {"claim_checked": False}
    bc_wins = sum(1 for c in comparisons if c.get("bose_chowla_exceeds_greedy"))
    if bc_wins >= len(comparisons) // 2 + 1:
        return _claim_result(
            supported=True,
            notes_suffix=f"; BC ratio exceeds greedy in {bc_wins}/{len(comparisons)} cases",
            claim_type="bc_vs_greedy",
        )
    if bc_wins == 0:
        return _claim_result(
            supported=False,
            notes_suffix=f" REFUTED: BC ratio never exceeds greedy ({bc_wins}/{len(comparisons)})",
            claim_type="bc_vs_greedy",
        )
    return _claim_result(
        supported=False,
        notes_suffix=f" REFUTED: mixed BC vs greedy results ({bc_wins}/{len(comparisons)} BC wins)",
        claim_type="bc_vs_greedy",
    )


def evaluate_experiment_claim(
    statement: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate computed evidence against all parseable claims in hypothesis text.
    Returns claim_checked/claim_supported and suggested verified_true/false steps.
    """
    if requires_unimplemented_statistics(statement):
        return _claim_result(
            supported=False,
            notes_suffix=(
                " REFUTED: hypothesis requires spectral/structural statistics "
                "not implemented in verifier"
            ),
            claim_type="unimplemented_statistics",
        )

    if result.get("comparison_sweep"):
        checks: list[dict[str, Any]] = [
            evaluate_bc_vs_greedy_claim(statement, result["comparison_sweep"]),
        ]
    else:
        checks = []

    ratios: list[float] = []
    if result.get("sweep"):
        ratios = [float(p.get("ratio_to_sqrt_n") or 0) for p in result["sweep"]]
    elif result.get("comparison_sweep"):
        ratios = [float(p.get("greedy_ratio") or 0) for p in result["comparison_sweep"]]
    elif result.get("ratio_to_sqrt_n") is not None:
        ratios = [float(result["ratio_to_sqrt_n"])]

    if ratios:
        checks.extend([
            evaluate_ratio_interval_claim(statement, ratios),
            evaluate_ratio_upper_bound_claim(statement, ratios),
            evaluate_monotonic_decrease_claim(statement, ratios),
        ])
    if result.get("sweep") and is_cap_set_hypothesis(statement):
        sizes = [(int(p["n"]), int(p["cap_set_size"])) for p in result["sweep"]]
        checks.append(evaluate_cap_growth_monotone_claim(statement, sizes))
    for pt in result.get("sweep") or []:
        if "cap_set_size" in pt:
            checks.append(
                evaluate_numeric_claim(
                    statement,
                    computed_size=int(pt["cap_set_size"]),
                    dimension=int(pt.get("n") or 0),
                ),
            )
    if result.get("cap_set_size") is not None:
        checks.append(
            evaluate_numeric_claim(
                statement,
                computed_size=int(result["cap_set_size"]),
                dimension=int(result.get("n") or 0) or None,
            ),
        )

    checked = [c for c in checks if c.get("claim_checked")]
    if not checked:
        return {"claim_checked": False}

    supported = [c for c in checked if c.get("claim_supported")]
    refuted = [c for c in checked if not c.get("claim_supported")]
    suffix = "".join(c.get("notes_suffix") or "" for c in checked)
    if refuted:
        return {
            "claim_checked": True,
            "claim_supported": False,
            "notes_suffix": suffix,
            "verified_true_steps": 0,
            "verified_false_steps": 1,
        }
    if supported:
        return {
            "claim_checked": True,
            "claim_supported": True,
            "notes_suffix": suffix,
            "verified_true_steps": 1,
            "verified_false_steps": 0,
        }
    return {"claim_checked": True, "claim_supported": False, "notes_suffix": suffix}


def _is_table_lookup_evidence(result: dict[str, Any]) -> bool:
    """True when the result's evidence came from a best-known lookup table.

    A best-known-table lookup is a *known value*, not an independent
    computation. Such evidence may confirm arithmetic but must never be
    called a discovery. We inspect (a) the top-level construction_source,
    (b) every entry of a ``sweep``/``comparison_sweep`` list, and
    (c) any nested ``threshold_search`` sub-result.
    """
    if result.get("construction_source") == "best_known_table":
        return True
    for key in ("sweep", "comparison_sweep"):
        entries = result.get(key)
        if isinstance(entries, list):
            for entry in entries:
                if (
                    isinstance(entry, dict)
                    and entry.get("construction_source") == "best_known_table"
                ):
                    return True
    threshold = result.get("threshold_search")
    if isinstance(threshold, dict) and threshold.get("construction_source") == "best_known_table":
        return True
    return False


def apply_claim_validation(
    result: dict[str, Any],
    statement: str,
) -> dict[str, Any]:
    """Apply claim validation; override verified_* when claims are checked."""
    if not evidence_metric_matches_hypothesis(statement, result):
        out = dict(result)
        out["metric_mismatch"] = True
        out["verified_true_steps"] = 0
        out["verified_false_steps"] = 0
        out["discovery_worthy"] = False
        out["trivial_rediscovery"] = True
        out["notes"] = (
            str(out.get("notes") or "")
            + f" INCONCLUSIVE: metric {out.get('metric_name')} mismatches hypothesis type"
        )
        return out

    claim = evaluate_experiment_claim(statement, result)
    if not claim.get("claim_checked"):
        # No parseable claim — do not confirm on sweep completion alone.
        out = dict(result)
        if int(out.get("verified_true_steps") or 0) > 0 and not claim.get("claim_supported"):
            out["verified_true_steps"] = 0
            out["verified_false_steps"] = 0
            out["discovery_worthy"] = False
            out["trivial_rediscovery"] = True
            out["notes"] = (
                str(out.get("notes") or "")
                + "; no falsifiable numeric claim parsed — sweep data only (inconclusive)"
            )
        return out

    out = dict(result)
    out["claim_checked"] = True
    out["claim_supported"] = claim.get("claim_supported")
    out["notes"] = str(out.get("notes") or "") + str(claim.get("notes_suffix") or "")
    if claim.get("claim_supported"):
        out["verified_true_steps"] = 1
        out["verified_false_steps"] = 0
        if _is_table_lookup_evidence(out):
            # The claim is merely "supported" by a best-known lookup value.
            # The arithmetic verdict stands, but this is a REDISCOVERY of a
            # known value, not an independent discovery.
            out["discovery_worthy"] = False
            out["trivial_rediscovery"] = True
            out["notes"] = str(out.get("notes") or "") + (
                "; rediscovery: verified against best-known table "
                "(a known value, not an independent computation)"
            )
        else:
            out["discovery_worthy"] = True
            out["trivial_rediscovery"] = False
    else:
        out["verified_true_steps"] = 0
        out["verified_false_steps"] = 1
        out["discovery_worthy"] = True
        out["trivial_rediscovery"] = False
    return out
