"""
Discovery kernel for math_combinatorics.

This subpackage adds the machinery to set a FIRST, machine-verifiable
computational record on a specific open problem: the maximum size of a B_3
(threefold-sum-distinct) set in the binary cube {0,1}^n (OEIS A396704).

Nothing here edits the existing verifier/plugin/constructors. The intended
plugin hook is documented in ``PLUGIN_WIRING.md``.

Public API
----------
- ``is_B3`` / ``certify_b3_record``      -- deterministic, paranoid witness checks
- ``RECORDS`` / ``get_record`` / ``best_known`` -- sourced best-known registry
- ``find_max_b3`` / ``find_b3_set``      -- the smart finder (ILS + repair + B&B)
- ``canonical_form`` / ``translate_to_origin`` -- hyperoctahedral symmetry helpers
"""
from __future__ import annotations

from propab.domain_modules.math_combinatorics.discovery.verifier import (
    certify_b3_record,
    is_B3,
    threefold_sums,
)
from propab.domain_modules.math_combinatorics.discovery.record_registry import (
    RECORDS,
    best_known,
    get_record,
    record_status,
)
from propab.domain_modules.math_combinatorics.discovery.symmetry import (
    canonical_form,
    translate_to_origin,
)
from propab.domain_modules.math_combinatorics.discovery.finder import (
    B3Index,
    CollisionCountIndex,
    find_b3_set,
    find_max_b3,
    max_b3_branch_and_bound,
)
from propab.domain_modules.math_combinatorics.discovery.cp_sat_finder import (
    attempt_a7_size17,
    decide_b3_cpsat,
    max_b3_cpsat,
    ortools_available,
)
from propab.domain_modules.math_combinatorics.discovery.modular_golomb import (
    A004135_TERMS,
    A004136_TERMS,
    attempt_open_term,
    certify_modular_ruler,
    decide_modular_ruler,
    is_modular_sidon,
    min_modular_ruler,
)

__all__ = [
    "is_B3",
    "threefold_sums",
    "certify_b3_record",
    "RECORDS",
    "get_record",
    "best_known",
    "record_status",
    "canonical_form",
    "translate_to_origin",
    "B3Index",
    "CollisionCountIndex",
    "find_b3_set",
    "find_max_b3",
    "max_b3_branch_and_bound",
    # Exact CP-SAT backend (OR-tools): sound sat/unsat decisions + optimization.
    "decide_b3_cpsat",
    "max_b3_cpsat",
    "attempt_a7_size17",
    "ortools_available",
    # Modular / cyclic Golomb rulers (OEIS A004135 / A004136): CP-SAT-friendly frontier.
    "decide_modular_ruler",
    "min_modular_ruler",
    "is_modular_sidon",
    "certify_modular_ruler",
    "attempt_open_term",
    "A004135_TERMS",
    "A004136_TERMS",
]
