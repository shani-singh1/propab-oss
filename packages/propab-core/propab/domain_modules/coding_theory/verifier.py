"""
Coding-theory experiment runner.

Given a hypothesis about a binary linear [n, k, d] code, this builds an ACTUAL
generator matrix over GF(2), computes its TRUE minimum distance by exhaustive
enumeration of the 2^k - 1 nonzero codewords, independently re-checks the witness
codeword, and reports an honest computed-vs-known verdict:

  * a code whose computed d meets-or-below the best-known table value is a
    trivial rediscovery (discovery_worthy=False, trivial_rediscovery=True);
  * a code whose computed d strictly BEATS the best-known lower bound at an entry
    is discovery-worthy (verified_true_steps=1);
  * a table-lookup path (distance without a real witness) is always demoted to
    rediscovery — never a discovery.

Evidence carries deterministic=True only alongside a real proof method
("exhaustive_enumeration") and a re-checkable witness, so it routes as
"deterministic" through the core verdict pipeline without gaming the gate.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from propab.domain_modules.coding_theory.constructors import (
    MAX_EXHAUSTIVE_K,
    best_known_distance,
    build_named_code,
    compute_min_distance,
    is_table_lookup_evidence,
    parse_code_params,
    parse_construction_name,
    random_generator,
    recompute_distance_of_witness,
    trivial_rediscovery,
)


def _explicit_generator(hypothesis: dict[str, Any]) -> np.ndarray | None:
    """A caller may pass an explicit generator matrix to verify directly."""
    for key in ("generator_matrix", "generator", "G"):
        g = hypothesis.get(key)
        if g is not None:
            try:
                return np.asarray(g, dtype=np.int64) % 2
            except Exception:  # noqa: BLE001
                return None
    ev = hypothesis.get("evidence")
    if isinstance(ev, dict):
        for key in ("generator_matrix", "generator", "G"):
            g = ev.get(key)
            if g is not None:
                try:
                    return np.asarray(g, dtype=np.int64) % 2
                except Exception:  # noqa: BLE001
                    return None
    return None


def _select_generator(hypothesis: dict[str, Any]) -> tuple[np.ndarray | None, str, str]:
    """
    Decide which real generator matrix to build/verify.

    Returns (generator, construction_name, source_note). Priority:
    1. explicit generator matrix on the hypothesis,
    2. a requested named construction (hamming, simplex, ...),
    3. a systematic random [n, k] code when n, k are given.
    """
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    methodology = str(hypothesis.get("test_methodology") or "")

    g = _explicit_generator(hypothesis)
    if g is not None:
        return g, "explicit", "explicit generator matrix supplied on hypothesis"

    name = parse_construction_name(statement, methodology)
    params = parse_code_params(statement)
    if name is not None:
        # Choose the parameter for the named family.
        if name in ("hamming", "extended_hamming", "simplex", "reed_muller"):
            param = _infer_family_param(name, params)
        elif name == "repetition":
            param = params.get("n") or 3
        elif name == "parity_check":
            param = (params.get("k") or ((params.get("n") or 4) - 1))
        else:
            param = params.get("n") or 3
        built = build_named_code(name, int(param))
        if built is not None:
            return built, name, f"named construction {name}(param={param})"

    n, k = params.get("n"), params.get("k")
    if n is not None and k is not None and 0 < k <= n and k <= MAX_EXHAUSTIVE_K:
        return (
            random_generator(int(n), int(k), seed=int(hypothesis.get("seed") or 0)),
            "systematic_random",
            f"systematic random [{n},{k}] generator (seed={hypothesis.get('seed') or 0})",
        )
    return None, "none", "no generator, named construction, or (n,k) found in hypothesis"


def _infer_family_param(name: str, params: dict[str, int | None]) -> int:
    """Infer the family parameter r/m from requested n (or a small default)."""
    n = params.get("n")
    k = params.get("k")
    if name == "hamming":
        # n = 2^r - 1
        if n is not None:
            r = max(2, int(round(np.log2(n + 1))))
            return r
        if k is not None:
            # k = 2^r - 1 - r; small search
            for r in range(2, 8):
                if (2 ** r - 1 - r) == k:
                    return r
        return 3
    if name == "extended_hamming":
        if n is not None:
            return max(2, int(round(np.log2(n))))
        return 3
    if name == "simplex":
        # n = 2^r - 1, k = r
        if k is not None:
            return max(2, int(k))
        if n is not None:
            return max(2, int(round(np.log2(n + 1))))
        return 3
    if name == "reed_muller":
        # n = 2^m
        if n is not None:
            return max(1, int(round(np.log2(n))))
        if k is not None:
            return max(1, int(k) - 1)
        return 3
    return 3


def run_coding_experiment(
    hypothesis: dict[str, Any],
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Build a real code, compute its true minimum distance, verdict novel-vs-known."""
    start = time.time()
    _ = features
    statement = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    params = parse_code_params(statement)
    claimed_d = params.get("d")

    generator, construction, source_note = _select_generator(hypothesis)

    if generator is None:
        return {
            "verified_true_steps": 0,
            "verified_false_steps": 0,
            "discovery_worthy": False,
            "trivial_rediscovery": False,
            "deterministic": True,
            "verification_method": "exhaustive_enumeration",
            "metric_name": "code_minimum_distance",
            "metric_value": None,
            "construction": construction,
            "notes": f"No verifiable code could be built: {source_note}",
            "elapsed_seconds": time.time() - start,
        }

    dist = compute_min_distance(generator)
    n, k = int(dist["n"]), int(dist["k"])
    computed_d = dist.get("min_distance")

    if not dist.get("generator_valid") or computed_d is None:
        return {
            "verified_true_steps": 0,
            "verified_false_steps": 0,
            "discovery_worthy": False,
            "trivial_rediscovery": False,
            "deterministic": True,
            "verification_method": "exhaustive_enumeration",
            "metric_name": "code_minimum_distance",
            "metric_value": None,
            "n": n,
            "k": k,
            "construction": construction,
            "generator_matrix": dist.get("generator_matrix"),
            "witness_codeword": None,
            "notes": f"{source_note}. {dist.get('notes')}",
            "elapsed_seconds": time.time() - start,
        }

    # --- Independent re-check of the witness (guards silent table-value bugs) ---
    recheck = recompute_distance_of_witness(
        dist["generator_matrix"], dist["witness_message"],
    )
    witness_ok = bool(
        recheck.get("ok")
        and recheck.get("weight") == computed_d
        and recheck.get("recomputed_codeword") == dist.get("witness_codeword")
    )

    known = best_known_distance(n, k)
    base: dict[str, Any] = {
        "deterministic": True,
        "verification_method": "exhaustive_enumeration",
        "metric_name": "code_minimum_distance",
        "metric_value": computed_d,
        "n": n,
        "k": k,
        "computed_min_distance": computed_d,
        "best_known_distance": known,
        "claimed_distance": claimed_d,
        "construction": construction,
        "construction_source": construction,
        "generator_matrix": dist["generator_matrix"],
        "witness_message": dist["witness_message"],
        "witness_codeword": dist["witness_codeword"],
        "witness_weight": (recheck.get("weight") if recheck.get("ok") else None),
        "witness_recheck_ok": witness_ok,
        "codewords_enumerated": dist.get("codewords_enumerated"),
        "enumeration_complete": dist.get("enumeration_complete"),
    }

    # If the witness fails independent recomputation, refuse to certify anything.
    if not witness_ok:
        base.update(
            verified_true_steps=0,
            verified_false_steps=0,
            discovery_worthy=False,
            trivial_rediscovery=False,
            notes=(
                f"{source_note}. WITNESS FAILED independent recomputation "
                f"(recheck={recheck}); distance {computed_d} not certifiable."
            ),
            elapsed_seconds=time.time() - start,
        )
        return base

    return _classify_distance_result(base, statement, claimed_d, computed_d, known, source_note, start)


def _classify_distance_result(
    base: dict[str, Any],
    statement: str,
    claimed_d: int | None,
    computed_d: int,
    known: int | None,
    source_note: str,
    start: float,
) -> dict[str, Any]:
    """Assign verdict fields honestly from computed-vs-known and any claimed d."""
    n, k = base["n"], base["k"]
    is_table = is_table_lookup_evidence(base)
    rediscovery = trivial_rediscovery(base, n, k, computed_d)

    # 1. Explicit falsifiable claim "[n,k] has distance >= claimed_d": check it.
    if claimed_d is not None:
        if computed_d >= claimed_d:
            # Claim met. Novel only if it strictly beats the best-known bound.
            beats_known = known is not None and computed_d > known
            base.update(
                verified_true_steps=1,
                verified_false_steps=0,
                discovery_worthy=bool(beats_known) and not is_table,
                trivial_rediscovery=(not beats_known) or is_table,
                notes=(
                    f"{source_note}. Built real [{n},{k}] code; computed d={computed_d} "
                    f"(claim d>={claimed_d} SUPPORTED). "
                    + (
                        f"BEATS best-known lower bound {known} — discovery-worthy."
                        if beats_known
                        else f"best-known={known}; meets/reproduces known bound "
                        "(rediscovery, not novel)."
                    )
                ),
                elapsed_seconds=time.time() - start,
            )
            return base
        # Claim refuted by real computation.
        base.update(
            verified_true_steps=0,
            verified_false_steps=1,
            discovery_worthy=False,
            trivial_rediscovery=False,
            notes=(
                f"{source_note}. Built real [{n},{k}] code; computed d={computed_d} "
                f"REFUTES claim d>={claimed_d} (witness weight {computed_d})."
            ),
            elapsed_seconds=time.time() - start,
        )
        return base

    # 2. No explicit claim: honest computed-vs-known report.
    if is_table:
        base.update(
            verified_true_steps=0,
            verified_false_steps=0,
            discovery_worthy=False,
            trivial_rediscovery=True,
            notes=f"{source_note}. Distance from table lookup, not a real witness — rediscovery.",
            elapsed_seconds=time.time() - start,
        )
        return base

    if known is not None and computed_d > known:
        base.update(
            verified_true_steps=1,
            verified_false_steps=0,
            discovery_worthy=True,
            trivial_rediscovery=False,
            notes=(
                f"{source_note}. Real [{n},{k}] code with computed d={computed_d} "
                f"BEATS best-known lower bound {known}. Witness re-checked."
            ),
            elapsed_seconds=time.time() - start,
        )
        return base

    # Meets-or-below known, or no table entry: honest, not oversold.
    base.update(
        verified_true_steps=0,
        verified_false_steps=0,
        discovery_worthy=False,
        trivial_rediscovery=bool(rediscovery),
        notes=(
            f"{source_note}. Real [{n},{k}] code, computed d={computed_d}, "
            f"best-known={known}. "
            + (
                "Reproduces/meets known bound (rediscovery, not novel)."
                if rediscovery
                else "No table entry for this [n,k]; reported honestly as a "
                "computed value, not a claimed improvement."
            )
        ),
        elapsed_seconds=time.time() - start,
    )
    return base
