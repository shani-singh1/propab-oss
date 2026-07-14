"""Target A — best-known binary linear codes [n, k, d].

The base-rate target: a cheap EXACT verifier (exhaustive enumeration of the 2^k - 1 nonzero
codewords), a legible public record (Grassl/Brouwer, codetables.de), and hundreds of genuinely
open cells. A candidate is a k x n generator matrix over GF(2); the objective is to maximise the
minimum distance d.

This module is a THIN adapter. The verifier already exists and is battle-tested — see
``domain_modules/coding_theory/constructors.py`` (``compute_min_distance``, ``is_valid_generator``,
``trivial_rediscovery``, ``is_table_lookup_evidence``). Nothing here recomputes a distance.


WHY THIS FILE CARRIES ITS OWN RECORD TABLE
------------------------------------------
``coding_theory.BEST_KNOWN_TABLE`` is NOT usable as the baseline for a discovery claim:

  1. It is WRONG in at least two cells.

     [16,12] -> it lists d=4. The true value is d=2, and that is provable in one line without
     consulting any table: a [16,12] code has r = n - k = 4 parity checks, so its parity-check
     matrix H has 4 rows and 16 columns; d >= 3 requires all columns to be nonzero and pairwise
     distinct, but GF(2)^4 contains only 15 nonzero vectors < 16. Hence d <= 2. (Equivalently,
     sphere-packing: 2^12 * (1 + 16) = 69632 > 65536 = 2^16.) codetables.de independently reports
     lower = upper = 2 for [16,12].

     [31,16] -> it lists d=7, commented "BCH [31,16,7]; optimal". The BCH code at [31,16] does have
     d=7, but it is NOT optimal: codetables.de reports lower = upper = 8, achieved by shortening the
     [32,17,8] Cheng-Sloane code at one coordinate. The comment conflated "the distance of the BCH
     code" with "the best distance at this cell". This is the DANGEROUS direction of error: a
     baseline that is too LOW would let the engine bank a [31,16,8] code -- which is merely the
     known optimum -- as a novel discovery. This file previously copied that same 7 into its own
     _CLOSED_ANCHORS; it is corrected to 8 below. ``ECCProblem._baseline`` takes the MAX over both
     sources, so the corrected value here now governs even though constructors.py still says 7.

  2. Every one of its cells is CLOSED (lower bound == upper bound == proven optimum). It contains no
     open cell, so a campaign run against it can only ever produce a rediscovery.

So the table below is sourced independently, records the UPPER bound as well as the lower bound (a
result above a *proven* upper bound is our bug, not a discovery), and carries a citation per bound.
An [n,k] cell with no sourced record is NOT ready to run: ``ECCProblem`` refuses to construct for it
rather than silently defaulting to a baseline of zero.

HOW THE OPEN CELLS BELOW WERE SOURCED
-------------------------------------
Source: M. Grassl, "Bounds on the minimum distance of linear codes and quantum codes",
https://codetables.de -- the per-cell BKLC pages, accessed 2026-07-15.

The procedure is spelled out because "an LLM read the numbers off a grid" is exactly how a fake
record gets shipped:

  1. The bounds GRID (n <= 64, k <= 16) was fetched once as raw HTML and parsed by regex. It was
     used ONLY to shortlist which cells are open. No bound in this file is taken from it.
  2. Each shortlisted cell's OWN BKLC page was then fetched individually (122 requests) and its
     ``lower bound:`` / ``upper bound:`` fields read by regex out of the raw HTML. Both numbers, and
     the derivation chain and literature keys recorded with them, come from that per-cell page.
     Nothing here is recalled from memory, summarised, or inferred.
  3. Grid and per-cell page were cross-checked against each other; all 122 agree.
  4. Every cell was then put through an elementary-bound battery that consults no table at all:
     Singleton, Griesmer (against the ACHIEVED distance), sphere-packing/Hamming, and the GF(2)^r
     distinct-column argument that kills [16,12,4]. All 122 pass. The battery is re-run as a test --
     see tests/evolve/targets/test_ecc_records.py, which is what keeps this table honest.

The registry therefore holds ONLY cells with lower < upper (a real gap) and k <= MAX_EXHAUSTIVE_K
(so the verifier can actually certify a witness by full 2^k enumeration). A cell that could not be
sourced to this standard was left out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...domain_modules.coding_theory.constructors import (
    MAX_EXHAUSTIVE_K,
    best_known_distance,
    compute_min_distance,
    is_table_lookup_evidence,
    is_valid_generator,
    recompute_distance_of_witness,
    trivial_rediscovery,
)
from ..problem import NEG_INF, Candidate, Verdict

CODETABLES = "M. Grassl, codetables.de (BKLC), accessed 2026-07-14"


# --------------------------------------------------------------------------- #
# The sourced record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BestKnown:
    """The real, citable record for one [n, k] cell.

    ``lower`` is the best minimum distance anyone has achieved (the number to beat).
    ``upper`` is the proven upper bound (no [n, k, upper + 1] code exists).
    The cell is OPEN — worth attacking — exactly when lower < upper.
    """

    n: int
    k: int
    lower: int
    upper: int
    lower_source: str
    upper_source: str
    accessed: str = CODETABLES

    @property
    def is_open(self) -> bool:
        return self.lower < self.upper

    @property
    def gap(self) -> int:
        return self.upper - self.lower


# Exactly-determined cells (lower == upper == optimum) for n <= 16, k = 1..n.
# Row n, position i  ->  k = i + 1. Cross-validated against two independent sources: the
# codetables.de bounds grid and the repo's BEST_KNOWN_TABLE. They agree on every entry EXCEPT
# [16,12], where the repo says 4 and the truth is 2 (see module docstring for the proof).
_OPTIMAL_SMALL_N: dict[int, list[int]] = {
    2: [2, 1],
    3: [3, 2, 1],
    4: [4, 2, 2, 1],
    5: [5, 3, 2, 2, 1],
    6: [6, 4, 3, 2, 2, 1],
    7: [7, 4, 4, 3, 2, 2, 1],
    8: [8, 5, 4, 4, 2, 2, 2, 1],
    9: [9, 6, 4, 4, 3, 2, 2, 2, 1],
    10: [10, 6, 5, 4, 4, 3, 2, 2, 2, 1],
    11: [11, 7, 6, 5, 4, 4, 3, 2, 2, 2, 1],
    12: [12, 8, 6, 6, 4, 4, 4, 3, 2, 2, 2, 1],
    13: [13, 8, 7, 6, 5, 4, 4, 4, 3, 2, 2, 2, 1],
    14: [14, 9, 8, 7, 6, 5, 4, 4, 4, 3, 2, 2, 2, 1],
    15: [15, 10, 8, 8, 7, 6, 5, 4, 4, 4, 3, 2, 2, 2, 1],
    16: [16, 10, 8, 8, 8, 6, 6, 5, 4, 4, 4, 2, 2, 2, 2, 1],
}

_CLOSED_ANCHORS: dict[tuple[int, int], tuple[int, str]] = {
    (23, 12): (7, "binary Golay code [23,12,7]; optimal"),
    (24, 12): (8, "extended binary Golay code [24,12,8]; optimal"),
    # NOT 7. The BCH code at [31,16] has d=7, but the OPTIMUM is 8: shorten the [32,17,8]
    # Cheng-Sloane code at one coordinate. codetables.de reports lower = upper = 8, and the upper
    # bound follows by a one-step Griesmer bound from Ub(22,15)=4. constructors.py still says 7 --
    # see the module docstring; _baseline() takes the max, so this value is the one that governs.
    (31, 16): (8, "shortening of the [32,17,8] Cheng-Sloane code; optimal (lower = upper = 8)"),
    (31, 21): (5, "BCH [31,21,5]; optimal"),
    (31, 26): (3, "Hamming [31,26,3]; optimal"),
}


_REFERENCES: dict[str, str] = {
    "B2x": (
        "See A.E. Brouwer & T. Verhoeff, An updated table of minimum-distance bounds for binary linear "
        "codes, IEEE Trans. Inform. Th. 39 (1993) 662-677."
    ),
    "BCH": (
        "T. Kasami & N. Tokura, Some remarks on BCH bounds and minimum weights of binary primitive BCH "
        "codes, IEEE Trans. Inform. Theory IT-15 (May 1969) 408-413. Or: a BCH code."
    ),
    "BE": (
        "J. Bierbrauer & Y. Edel, New code parameters from Reed-Solomon subfield subcodes, IEEE Trans. "
        "Inf. Th. 43 (1997) 953-968."
    ),
    "BZ": (
        "E. L. Blokh & V. V. Zyablov, Coding of generalized concatenated codes, Probl. Inform. Transm. "
        "10 (1974) 218-222."
    ),
    "Bo0": "I Boukliev, private comm. 1995-1997.",
    "Bou": (
        "I. Boukliev, Some new bounds on minimum length for quaternary codes of dimension five, "
        "preprint, July 1994."
    ),
    "Bro": (
        "A.E. Brouwer, The linear programming bound for binary linear codes, IEEE Trans. Inform. Th. 39 "
        "(1993) 677-680."
    ),
    "CDJ": (
        "Huy T. Cao, Randall L. Dougherty & Heeralal Janwa, A [55,16,19] binary Goppa code and related "
        "codes, having large minimum distance, IEEE Trans. Inform. Theory 37 (Sep. 1991) 1432."
    ),
    "CS": "Y. Cheng & N.J.A. Sloane, Codes from symmetry groups, SIAM J. Discrete Math. 2 (1989) 28-37.",
    "Ch": (
        "Y. Cheng, New linear codes constructed by concatenating, extending, and shortening methods, "
        "IEEE Trans. Inform. Theory IT-33 (Sept. 1987) 719-721."
    ),
    "DJ": (
        "R. Dougherty & H. Janwa, Covering radius computations for binary cyclic codes, Math. Comput. "
        "57 (July 1991) 415-434."
    ),
    "DM": (
        "S.M. Dodunekov & N.L. Manev, An improvement of the Griesmer bound for some small minimum "
        "distances, Discr. Appl. Math. 12 (Oct. 1985) 103-114."
    ),
    "FB": (
        "P. Farkavs & K. Brühl, Three best binary linear block codes of minimum distance fifteen, IEEE "
        "Trans. Inf. Th. 40 (1994) 949-951."
    ),
    "GB5": (
        "T. A. Gulliver & V. K. Bhargava, New optimal binary linear codes of dimensions 9 and 10, IEEE "
        "Trans. Inform. Theory 43 (1997) 314-316."
    ),
    "GG": "B. Groneick & S. Grosse, New binary codes, IEEE Trans. Inform. Theory 40 (1994) 510-512.",
    "Gu9": "T. A. Gulliver, personal communications 1993-1998.",
    "He": "P.W. Heijnen, Er bestaat geen binaire [33,9,13] code, Afstudeerverslag, T.U. Delft, Oct. 1993.",
    "Ja": "D.B. Jaffe, Binary linear codes: new results on nonexistence, 1996, code.ps.gz.",
    "LC": (
        "M. Loeloeian & J. Conan, A [55,16,19] binary Goppa code, IEEE Trans. Inform. Theory IT-30 "
        "(Sep. 1984) 773."
    ),
    "Mo": "M. Morii, email comm., Sept. 1993.",
    "Pu": (
        "C.L.M. van Pul, On bounds on codes, Master's Thesis, Dept. of Math. and Comp. Sc., Eindhoven "
        "Univ. of Techn., The Netherlands, Aug. 1982."
    ),
    "Pu2": "C.L.M. van Pul, [26,13,8] does not exist, priv. comm. 1985.",
    "QR": "A quadratic residue code.",
    "SRC": (
        "N.J.A. Sloane, S.M. Reddy & C.L. Chen, New binary codes, IEEE Trans. Inform. Theory IT-18 "
        "(July 1972) 503-510."
    ),
    # UNEXPANDED UPSTREAM REFERENCE. codetables.de names this key for the lower bound of 5 cells,
    # but its own reference list renders "Sh1:" with no text on every one of those pages. We refuse
    # to guess what it stands for. The BOUND is unaffected: the value is stated on the cell's own
    # page and carries a full derivation chain -- only the bibliographic expansion is missing
    # upstream. See _UNEXPANDED_UPSTREAM_KEYS and the tests.
    "Sh1": (
        "UNEXPANDED: codetables.de cites key 'Sh1' for this bound but gives no bibliographic entry "
        "for it on any cell page. The bound value and its derivation chain are still taken from the "
        "cell's own codetables.de page; only this reference is missing upstream."
    ),
    "Si": (
        "J. Simonis, Binary even [25,15,6] codes do not exist, IEEE Trans. Inform. Theory IT-33 (Jan. "
        "1987) 151-153."
    ),
    "XBC": "Extended BCH code.",
    "YH1": (
        "Oyvind Ytrehus & Tor Helleseth, There is no binary [25,8,10] code, IEEE Trans. Inform. Theory "
        "36 (May 1990) 695-696."
    ),
    # UNEXPANDED UPSTREAM REFERENCE (11 cells). Same situation as "Sh1". Neighbouring construction
    # keys in this list ("XBC": Extended BCH code, "QR": A quadratic residue code) suggest "cy"
    # denotes a cyclic code, but codetables.de gives no text for it and we do not guess.
    "cy": (
        "UNEXPANDED: codetables.de cites key 'cy' for this bound but gives no bibliographic entry "
        "for it on any cell page. The bound value and its derivation chain are still taken from the "
        "cell's own codetables.de page; only this reference is missing upstream."
    ),
    "vT3": (
        "H.C.A. van Tilborg, The smallest length of binary 7-dimensional linear codes with prescribed "
        "minimum distance, Discr. Math. 33 (1981) 197-207."
    ),
}


# Every genuinely OPEN cell (lower < upper) with k <= MAX_EXHAUSTIVE_K and n <= 64.
# (n, k, lower, upper, lower derivation, lower refs, upper derivation, upper refs)
# Derivations are quoted verbatim from that cell's own codetables.de page; the ref tuples
# are the literature keys that page cites for each bound, resolved via _REFERENCES.
_OPEN_CELLS: tuple[tuple[int, int, int, int, str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    (35, 10, 12, 13,
     (
         "Lb(35,10) = 12 is found by taking a subcode of: | Lb(35,12) = 12 is found by "
         "shortening of: | Lb(37,14) = 12 is found by adding a parity check bit to: | Lb(36,14) "
         "= 11 Mo"
     ),
     ("Mo",),
     (
         "Ub(35,10) = 13 is found by considering shortening to: | Ub(34,9) = 13 is found by "
         "considering truncation to: | Ub(33,9) = 12 He"
     ),
     ("He",),
    ),
    (36, 10, 13, 14,
     (
         "Lb(36,10) = 13 is found by shortening of: | Lb(38,12) = 13 is found by truncation of: "
         "| Lb(39,12) = 14 BE"
     ),
     ("BE",),
     (
         "Ub(36,10) = 14 is found by considering shortening to: | Ub(33,7) = 14 otherwise adding "
         "a parity check bit would contradict: | Ub(34,7) = 15 vT3"
     ),
     ("vT3",),
    ),
    (43, 10, 16, 17,
     (
         "Lb(43,10) = 16 is found by taking a subcode of: | Lb(43,12) = 16 is found by "
         "shortening of: | Lb(45,14) = 16 Bo0"
     ),
     ("Bo0",),
     (
         "Ub(43,10) = 17 follows by a one-step Griesmer bound from: | Ub(25,9) = 8 is found by "
         "considering shortening to: | Ub(24,8) = 8 otherwise adding a parity check bit would "
         "contradict: | Ub(25,8) = 9 YH1"
     ),
     ("YH1",),
    ),
    (44, 10, 17, 18,
     "Lb(44,10) = 17 is found by truncation of: | Lb(45,10) = 18 Gu9",
     ("Gu9",),
     (
         "Ub(44,10) = 18 follows by a one-step Griesmer bound from: | Ub(25,9) = 8 is found by "
         "considering shortening to: | Ub(24,8) = 8 otherwise adding a parity check bit would "
         "contradict: | Ub(25,8) = 9 YH1"
     ),
     ("YH1",),
    ),
    (46, 10, 18, 19,
     (
         "Lb(46,10) = 18 is found by shortening of: | Lb(47,11) = 18 is found by adding a parity "
         "check bit to: | Lb(46,11) = 17 GG"
     ),
     ("GG",),
     (
         "Ub(46,10) = 19 follows by a one-step Griesmer bound from: | Ub(26,9) = 9 is found by "
         "considering shortening to: | Ub(25,8) = 9 YH1"
     ),
     ("YH1",),
    ),
    (47, 10, 19, 20,
     "Lb(47,10) = 19 is found by truncation of: | Lb(48,10) = 20 GB5",
     ("GB5",),
     "Ub(47,10) = 20 follows by the Griesmer bound.",
     (),
    ),
    (51, 10, 20, 21,
     (
         "Lb(51,10) = 20 is found by taking a subcode of: | Lb(51,11) = 20 is found by "
         "shortening of: | Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) "
         "= 19 LC"
     ),
     ("LC",),
     (
         "Ub(51,10) = 21 is found by considering shortening to: | Ub(50,9) = 21 is found by "
         "considering truncation to: | Ub(49,9) = 20 Ja"
     ),
     ("Ja",),
    ),
    (52, 10, 21, 22,
     "Lb(52,10) = 21 Pu",
     ("Pu",),
     (
         "Ub(52,10) = 22 follows by a one-step Griesmer bound from: | Ub(29,9) = 11 is found by "
         "considering shortening to: | Ub(28,8) = 11 DM"
     ),
     ("DM",),
    ),
    (54, 10, 22, 23,
     (
         "Lb(54,10) = 22 is found by shortening of: | Lb(55,11) = 22 is found by adding a parity "
         "check bit to: | Lb(54,11) = 21 GG"
     ),
     ("GG",),
     "Ub(54,10) = 23 is found by considering shortening to: | Ub(53,9) = 23 Bro",
     ("Bro",),
    ),
    (55, 10, 23, 24,
     "Lb(55,10) = 23 is found by truncation of: | Lb(56,10) = 24 BZ",
     ("BZ",),
     "Ub(55,10) = 24 follows by the Griesmer bound.",
     (),
    ),
    (59, 10, 24, 25,
     (
         "Lb(59,10) = 24 is found by taking a subcode of: | Lb(59,11) = 24 is found by "
         "shortening of: | Lb(64,16) = 24 XBC"
     ),
     ("XBC",),
     "Ub(59,10) = 25 follows by a one-step Griesmer bound from: | Ub(33,9) = 12 He",
     ("He",),
    ),
    (60, 10, 25, 26,
     "Lb(60,10) = 25 Ch",
     ("Ch",),
     "Ub(60,10) = 26 follows by a one-step Griesmer bound from: | Ub(33,9) = 12 He",
     ("He",),
    ),
    (62, 10, 26, 27,
     "Lb(62,10) = 26 is found by shortening of: | Lb(63,11) = 26 BCH",
     ("BCH",),
     (
         "Ub(62,10) = 27 follows by a one-step Griesmer bound from: | Ub(34,9) = 13 is found by "
         "considering truncation to: | Ub(33,9) = 12 He"
     ),
     ("He",),
    ),
    (63, 10, 27, 28,
     "Lb(63,10) = 27 is found by truncation of: | Lb(64,10) = 28 XBC",
     ("XBC",),
     (
         "Ub(63,10) = 28 follows by a one-step Griesmer bound from: | Ub(34,9) = 13 is found by "
         "considering truncation to: | Ub(33,9) = 12 He"
     ),
     ("He",),
    ),
    (36, 11, 12, 13,
     (
         "Lb(36,11) = 12 is found by taking a subcode of: | Lb(36,13) = 12 is found by "
         "shortening of: | Lb(37,14) = 12 is found by adding a parity check bit to: | Lb(36,14) "
         "= 11 Mo"
     ),
     ("Mo",),
     (
         "Ub(36,11) = 13 is found by considering shortening to: | Ub(34,9) = 13 is found by "
         "considering truncation to: | Ub(33,9) = 12 He"
     ),
     ("He",),
    ),
    (37, 11, 13, 14,
     (
         "Lb(37,11) = 13 is found by shortening of: | Lb(38,12) = 13 is found by truncation of: "
         "| Lb(39,12) = 14 BE"
     ),
     ("BE",),
     (
         "Ub(37,11) = 14 is found by considering shortening to: | Ub(33,7) = 14 otherwise adding "
         "a parity check bit would contradict: | Ub(34,7) = 15 vT3"
     ),
     ("vT3",),
    ),
    (44, 11, 16, 17,
     (
         "Lb(44,11) = 16 is found by taking a subcode of: | Lb(44,13) = 16 is found by "
         "shortening of: | Lb(45,14) = 16 Bo0"
     ),
     ("Bo0",),
     (
         "Ub(44,11) = 17 follows by a one-step Griesmer bound from: | Ub(26,10) = 8 is found by "
         "considering shortening to: | Ub(24,8) = 8 otherwise adding a parity check bit would "
         "contradict: | Ub(25,8) = 9 YH1"
     ),
     ("YH1",),
    ),
    (45, 11, 16, 18,
     "Lb(45,11) = 16 is found by taking a subcode of: | Lb(45,14) = 16 Bo0",
     ("Bo0",),
     (
         "Ub(45,11) = 18 follows by a one-step Griesmer bound from: | Ub(26,10) = 8 is found by "
         "considering shortening to: | Ub(24,8) = 8 otherwise adding a parity check bit would "
         "contradict: | Ub(25,8) = 9 YH1"
     ),
     ("YH1",),
    ),
    (46, 11, 17, 18,
     "Lb(46,11) = 17 GG",
     ("GG",),
     (
         "Ub(46,11) = 18 follows by a one-step Griesmer bound from: | Ub(27,10) = 9 is found by "
         "considering shortening to: | Ub(25,8) = 9 YH1"
     ),
     ("YH1",),
    ),
    (48, 11, 18, 19,
     (
         "Lb(48,11) = 18 is found by shortening of: | Lb(50,13) = 18 is found by adding a parity "
         "check bit to: | Lb(49,13) = 17 B2x"
     ),
     ("B2x",),
     "Ub(48,11) = 19 Ja",
     ("Ja",),
    ),
    (49, 11, 19, 20,
     "Lb(49,11) = 19 B2x",
     ("B2x",),
     (
         "Ub(49,11) = 20 follows by a one-step Griesmer bound from: | Ub(28,10) = 10 follows by "
         "a one-step Griesmer bound from: | Ub(17,9) = 5 follows by a one-step Griesmer bound "
         "from: | Ub(11,8) = 2 is found by considering shortening to: | Ub(8,5) = 2 is found by "
         "construction B:"
     ),
     (),
    ),
    (52, 11, 20, 21,
     (
         "Lb(52,11) = 20 is found by taking a subcode of: | Lb(52,12) = 20 is found by "
         "shortening of: | Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) "
         "= 19 LC"
     ),
     ("LC",),
     (
         "Ub(52,11) = 21 is found by considering shortening to: | Ub(50,9) = 21 is found by "
         "considering truncation to: | Ub(49,9) = 20 Ja"
     ),
     ("Ja",),
    ),
    (53, 11, 20, 22,
     (
         "Lb(53,11) = 20 is found by taking a subcode of: | Lb(53,13) = 20 is found by "
         "shortening of: | Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) "
         "= 19 LC"
     ),
     ("LC",),
     (
         "Ub(53,11) = 22 follows by a one-step Griesmer bound from: | Ub(30,10) = 11 is found by "
         "considering shortening to: | Ub(28,8) = 11 DM"
     ),
     ("DM",),
    ),
    (54, 11, 21, 22,
     "Lb(54,11) = 21 GG",
     ("GG",),
     (
         "Ub(54,11) = 22 is found by considering shortening to: | Ub(52,9) = 22 otherwise adding "
         "a parity check bit would contradict: | Ub(53,9) = 23 Bro"
     ),
     ("Bro",),
    ),
    (55, 11, 22, 23,
     "Lb(55,11) = 22 is found by adding a parity check bit to: | Lb(54,11) = 21 GG",
     ("GG",),
     "Ub(55,11) = 23 is found by considering shortening to: | Ub(53,9) = 23 Bro",
     ("Bro",),
    ),
    (56, 11, 22, 24,
     "Lb(56,11) = 22 is found by shortening of: | Lb(58,13) = 22 GG",
     ("GG",),
     "Ub(56,11) = 24 follows by the Griesmer bound.",
     (),
    ),
    (57, 11, 23, 24,
     "Lb(57,11) = 23 SRC",
     ("SRC",),
     (
         "Ub(57,11) = 24 follows by a one-step Griesmer bound from: | Ub(32,10) = 12 follows by "
         "a one-step Griesmer bound from: | Ub(19,9) = 6 follows by a one-step Griesmer bound "
         "from: | Ub(12,8) = 3 is found by considering shortening to: | Ub(9,5) = 3 is found by "
         "considering truncation to: | Ub(8,5) = 2 is found by construction B:"
     ),
     (),
    ),
    (61, 11, 24, 25,
     (
         "Lb(61,11) = 24 is found by taking a subcode of: | Lb(61,13) = 24 is found by "
         "shortening of: | Lb(64,16) = 24 XBC"
     ),
     ("XBC",),
     "Ub(61,11) = 25 is found by construction B:",
     (),
    ),
    (62, 11, 25, 26,
     "Lb(62,11) = 25 is found by truncation of: | Lb(63,11) = 26 BCH",
     ("BCH",),
     (
         "Ub(62,11) = 26 follows by a one-step Griesmer bound from: | Ub(35,10) = 13 is found by "
         "considering shortening to: | Ub(34,9) = 13 is found by considering truncation to: | "
         "Ub(33,9) = 12 He"
     ),
     ("He",),
    ),
    (64, 11, 26, 27,
     (
         "Lb(64,11) = 26 is found by shortening of: | Lb(66,13) = 26 is found by adding a parity "
         "check bit to: | Lb(65,13) = 25 cy"
     ),
     ("cy",),
     "Ub(64,11) = 27 is found by considering truncation to: | Ub(63,11) = 26 Ja",
     ("Ja",),
    ),
    (37, 12, 12, 13,
     (
         "Lb(37,12) = 12 is found by taking a subcode of: | Lb(37,14) = 12 is found by adding a "
         "parity check bit to: | Lb(36,14) = 11 Mo"
     ),
     ("Mo",),
     (
         "Ub(37,12) = 13 is found by considering shortening to: | Ub(34,9) = 13 is found by "
         "considering truncation to: | Ub(33,9) = 12 He"
     ),
     ("He",),
    ),
    (38, 12, 13, 14,
     "Lb(38,12) = 13 is found by truncation of: | Lb(39,12) = 14 BE",
     ("BE",),
     (
         "Ub(38,12) = 14 is found by considering shortening to: | Ub(33,7) = 14 otherwise adding "
         "a parity check bit would contradict: | Ub(34,7) = 15 vT3"
     ),
     ("vT3",),
    ),
    (41, 12, 14, 15,
     (
         "Lb(41,12) = 14 is found by shortening of: | Lb(44,15) = 14 is found by adding a parity "
         "check bit to: | Lb(43,15) = 13 cy"
     ),
     ("cy",),
     "Ub(41,12) = 15 is found by considering shortening to: | Ub(39,10) = 15 Bou",
     ("Bou",),
    ),
    (42, 12, 15, 16,
     (
         "Lb(42,12) = 15 is found by shortening of: | Lb(44,14) = 15 is found by truncation of: "
         "| Lb(45,14) = 16 Bo0"
     ),
     ("Bo0",),
     "Ub(42,12) = 16 follows by the Griesmer bound.",
     (),
    ),
    (46, 12, 16, 17,
     (
         "Lb(46,12) = 16 is found by taking a subcode of: | Lb(46,14) = 16 is found by "
         "lengthening of: | Lb(45,14) = 16 Bo0"
     ),
     ("Bo0",),
     (
         "Ub(46,12) = 17 follows by a one-step Griesmer bound from: | Ub(28,11) = 8 otherwise "
         "adding a parity check bit would contradict: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (47, 12, 16, 18,
     (
         "Lb(47,12) = 16 is found by taking a subcode of: | Lb(47,14) = 16 is found by "
         "shortening of: | Lb(49,16) = 16 is found by adding a parity check bit to: | Lb(48,16) "
         "= 15 FB"
     ),
     ("FB",),
     (
         "Ub(47,12) = 18 follows by a one-step Griesmer bound from: | Ub(28,11) = 8 otherwise "
         "adding a parity check bit would contradict: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (48, 12, 17, 18,
     "Lb(48,12) = 17 is found by shortening of: | Lb(49,13) = 17 B2x",
     ("B2x",),
     "Ub(48,12) = 18 follows by a one-step Griesmer bound from: | Ub(29,11) = 9 Ja",
     ("Ja",),
    ),
    (49, 12, 18, 19,
     (
         "Lb(49,12) = 18 is found by shortening of: | Lb(50,13) = 18 is found by adding a parity "
         "check bit to: | Lb(49,13) = 17 B2x"
     ),
     ("B2x",),
     "Ub(49,12) = 19 follows by a one-step Griesmer bound from: | Ub(29,11) = 9 Ja",
     ("Ja",),
    ),
    (50, 12, 18, 20,
     (
         "Lb(50,12) = 18 is found by taking a subcode of: | Lb(50,13) = 18 is found by adding a "
         "parity check bit to: | Lb(49,13) = 17 B2x"
     ),
     ("B2x",),
     "Ub(50,12) = 20 follows by a one-step Griesmer bound from: | Ub(29,11) = 9 Ja",
     ("Ja",),
    ),
    (51, 12, 19, 20,
     "Lb(51,12) = 19 is found by shortening of: | Lb(55,16) = 19 LC",
     ("LC",),
     (
         "Ub(51,12) = 20 follows by a one-step Griesmer bound from: | Ub(30,11) = 10 follows by "
         "a one-step Griesmer bound from: | Ub(19,10) = 5 is found by construction B:"
     ),
     (),
    ),
    (53, 12, 20, 21,
     (
         "Lb(53,12) = 20 is found by taking a subcode of: | Lb(53,13) = 20 is found by "
         "shortening of: | Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) "
         "= 19 LC"
     ),
     ("LC",),
     (
         "Ub(53,12) = 21 is found by considering shortening to: | Ub(50,9) = 21 is found by "
         "considering truncation to: | Ub(49,9) = 20 Ja"
     ),
     ("Ja",),
    ),
    (54, 12, 20, 22,
     (
         "Lb(54,12) = 20 is found by taking a subcode of: | Lb(54,14) = 20 is found by "
         "shortening of: | Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) "
         "= 19 LC"
     ),
     ("LC",),
     (
         "Ub(54,12) = 22 follows by a one-step Griesmer bound from: | Ub(31,11) = 11 follows by "
         "a one-step Griesmer bound from: | Ub(19,10) = 5 is found by construction B:"
     ),
     (),
    ),
    (55, 12, 21, 22,
     (
         "Lb(55,12) = 20 is found by taking a subcode of: | Lb(55,15) = 20 is found by "
         "shortening of: | Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) "
         "= 19 LC"
     ),
     ("LC",),
     (
         "Ub(55,12) = 22 is found by considering shortening to: | Ub(52,9) = 22 otherwise adding "
         "a parity check bit would contradict: | Ub(53,9) = 23 Bro"
     ),
     ("Bro",),
    ),
    (57, 12, 22, 23,
     "Lb(57,12) = 22 is found by shortening of: | Lb(58,13) = 22 GG",
     ("GG",),
     "Ub(57,12) = 23 Ja",
     ("Ja",),
    ),
    (58, 12, 23, 24,
     "Lb(58,12) = 22 is found by taking a subcode of: | Lb(58,13) = 22 GG",
     ("GG",),
     (
         "Ub(58,12) = 24 follows by a one-step Griesmer bound from: | Ub(33,11) = 12 follows by "
         "a one-step Griesmer bound from: | Ub(20,10) = 6 follows by a one-step Griesmer bound "
         "from: | Ub(13,9) = 3 is found by considering shortening to: | Ub(9,5) = 3 is found by "
         "considering truncation to: | Ub(8,5) = 2 is found by construction B:"
     ),
     (),
    ),
    (62, 12, 24, 25,
     (
         "Lb(62,12) = 24 is found by taking a subcode of: | Lb(62,14) = 24 is found by "
         "shortening of: | Lb(64,16) = 24 XBC"
     ),
     ("XBC",),
     (
         "Ub(62,12) = 25 is found by considering shortening to: | Ub(61,11) = 25 is found by "
         "construction B:"
     ),
     (),
    ),
    (63, 12, 24, 26,
     (
         "Lb(63,12) = 24 is found by taking a subcode of: | Lb(63,15) = 24 is found by "
         "shortening of: | Lb(64,16) = 24 XBC"
     ),
     ("XBC",),
     (
         "Ub(63,12) = 26 follows by a one-step Griesmer bound from: | Ub(36,11) = 13 is found by "
         "considering shortening to: | Ub(34,9) = 13 is found by considering truncation to: | "
         "Ub(33,9) = 12 He"
     ),
     ("He",),
    ),
    (64, 12, 25, 26,
     "Lb(64,12) = 25 is found by shortening of: | Lb(65,13) = 25 cy",
     ("cy",),
     "Ub(64,12) = 26 is found by considering shortening to: | Ub(63,11) = 26 Ja",
     ("Ja",),
    ),
    (39, 13, 12, 13,
     (
         "Lb(39,13) = 12 is found by taking a subcode of: | Lb(39,15) = 12 is found by "
         "shortening of: | Lb(48,24) = 12 QR"
     ),
     ("QR",),
     "Ub(39,13) = 13 Ja",
     ("Ja",),
    ),
    (40, 13, 13, 14,
     (
         "Lb(40,13) = 12 is found by taking a subcode of: | Lb(40,16) = 12 is found by "
         "shortening of: | Lb(48,24) = 12 QR"
     ),
     ("QR",),
     "Ub(40,13) = 14 is found by considering shortening to: | Ub(36,9) = 14 Ja",
     ("Ja",),
    ),
    (47, 13, 16, 17,
     (
         "Lb(47,13) = 16 is found by taking a subcode of: | Lb(47,14) = 16 is found by "
         "shortening of: | Lb(49,16) = 16 is found by adding a parity check bit to: | Lb(48,16) "
         "= 15 FB"
     ),
     ("FB",),
     (
         "Ub(47,13) = 17 follows by a one-step Griesmer bound from: | Ub(29,12) = 8 is found by "
         "considering shortening to: | Ub(28,11) = 8 otherwise adding a parity check bit would "
         "contradict: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (48, 13, 16, 18,
     (
         "Lb(48,13) = 16 is found by taking a subcode of: | Lb(48,15) = 16 is found by "
         "shortening of: | Lb(49,16) = 16 is found by adding a parity check bit to: | Lb(48,16) "
         "= 15 FB"
     ),
     ("FB",),
     (
         "Ub(48,13) = 18 follows by a one-step Griesmer bound from: | Ub(29,12) = 8 is found by "
         "considering shortening to: | Ub(28,11) = 8 otherwise adding a parity check bit would "
         "contradict: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (49, 13, 17, 18,
     "Lb(49,13) = 17 B2x",
     ("B2x",),
     (
         "Ub(49,13) = 18 follows by a one-step Griesmer bound from: | Ub(30,12) = 9 is found by "
         "considering shortening to: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (50, 13, 18, 19,
     "Lb(50,13) = 18 is found by adding a parity check bit to: | Lb(49,13) = 17 B2x",
     ("B2x",),
     (
         "Ub(50,13) = 19 follows by a one-step Griesmer bound from: | Ub(30,12) = 9 is found by "
         "considering shortening to: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (51, 13, 18, 20,
     (
         "Lb(51,13) = 18 is found by shortening of: | Lb(54,16) = 18 is found by truncation of: "
         "| Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) = 19 LC"
     ),
     ("LC",),
     (
         "Ub(51,13) = 20 follows by a one-step Griesmer bound from: | Ub(30,12) = 9 is found by "
         "considering shortening to: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (52, 13, 19, 20,
     "Lb(52,13) = 19 is found by shortening of: | Lb(55,16) = 19 LC",
     ("LC",),
     (
         "Ub(52,13) = 20 follows by a one-step Griesmer bound from: | Ub(31,12) = 10 follows by "
         "a one-step Griesmer bound from: | Ub(20,11) = 5 is found by considering shortening to: "
         "| Ub(19,10) = 5 is found by construction B:"
     ),
     (),
    ),
    (55, 13, 20, 21,
     (
         "Lb(55,13) = 20 is found by taking a subcode of: | Lb(55,15) = 20 is found by "
         "shortening of: | Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) "
         "= 19 LC"
     ),
     ("LC",),
     "Ub(55,13) = 21 Ja",
     ("Ja",),
    ),
    (56, 13, 20, 22,
     (
         "Lb(56,13) = 20 is found by taking a subcode of: | Lb(56,16) = 20 is found by adding a "
         "parity check bit to: | Lb(55,16) = 19 LC"
     ),
     ("LC",),
     "Ub(56,13) = 22 follows by a one-step Griesmer bound from: | Ub(33,12) = 11 Ja",
     ("Ja",),
    ),
    (57, 13, 21, 22,
     "Lb(57,13) = 21 is found by truncation of: | Lb(58,13) = 22 GG",
     ("GG",),
     (
         "Ub(57,13) = 22 is found by considering shortening to: | Ub(56,12) = 22 otherwise "
         "adding a parity check bit would contradict: | Ub(57,12) = 23 Ja"
     ),
     ("Ja",),
    ),
    (58, 13, 22, 23,
     "Lb(58,13) = 22 GG",
     ("GG",),
     "Ub(58,13) = 23 is found by considering shortening to: | Ub(57,12) = 23 Ja",
     ("Ja",),
    ),
    (59, 13, 23, 24,
     "Lb(59,13) = 22 is found by shortening of: | Lb(64,18) = 22 XBC",
     ("XBC",),
     (
         "Ub(59,13) = 24 follows by a one-step Griesmer bound from: | Ub(34,12) = 12 follows by "
         "a one-step Griesmer bound from: | Ub(21,11) = 6 follows by a one-step Griesmer bound "
         "from: | Ub(14,10) = 3 is found by considering shortening to: | Ub(9,5) = 3 is found by "
         "considering truncation to: | Ub(8,5) = 2 is found by construction B:"
     ),
     (),
    ),
    (63, 13, 24, 25,
     (
         "Lb(63,13) = 24 is found by taking a subcode of: | Lb(63,15) = 24 is found by "
         "shortening of: | Lb(64,16) = 24 XBC"
     ),
     ("XBC",),
     (
         "Ub(63,13) = 25 is found by considering shortening to: | Ub(61,11) = 25 is found by "
         "construction B:"
     ),
     (),
    ),
    (64, 13, 24, 26,
     "Lb(64,13) = 24 is found by taking a subcode of: | Lb(64,16) = 24 XBC",
     ("XBC",),
     (
         "Ub(64,13) = 26 follows by a one-step Griesmer bound from: | Ub(37,12) = 13 is found by "
         "considering shortening to: | Ub(34,9) = 13 is found by considering truncation to: | "
         "Ub(33,9) = 12 He"
     ),
     ("He",),
    ),
    (32, 14, 8, 9,
     "Lb(32,14) = 8 is found by taking a subcode of: | Lb(32,17) = 8 CS",
     ("CS",),
     "Ub(32,14) = 9 is found by considering shortening to: | Ub(29,11) = 9 Ja",
     ("Ja",),
    ),
    (33, 14, 9, 10,
     (
         "Lb(33,14) = 9 is found by shortening of: | Lb(35,16) = 9 is found by truncation of: | "
         "Lb(36,16) = 10 Sh1"
     ),
     ("Sh1",),
     (
         "Ub(33,14) = 10 follows by a one-step Griesmer bound from: | Ub(22,13) = 5 follows by a "
         "one-step Griesmer bound from: | Ub(16,12) = 2 is found by construction B:"
     ),
     (),
    ),
    (40, 14, 12, 13,
     (
         "Lb(40,14) = 12 is found by taking a subcode of: | Lb(40,16) = 12 is found by "
         "shortening of: | Lb(48,24) = 12 QR"
     ),
     ("QR",),
     "Ub(40,14) = 13 is found by considering shortening to: | Ub(39,13) = 13 Ja",
     ("Ja",),
    ),
    (41, 14, 12, 14,
     (
         "Lb(41,14) = 12 is found by taking a subcode of: | Lb(41,17) = 12 is found by "
         "shortening of: | Lb(48,24) = 12 QR"
     ),
     ("QR",),
     "Ub(41,14) = 14 follows by a one-step Griesmer bound from: | Ub(26,13) = 7 Pu2",
     ("Pu2",),
    ),
    (42, 14, 13, 14,
     "Lb(42,14) = 13 is found by shortening of: | Lb(43,15) = 13 cy",
     ("cy",),
     (
         "Ub(42,14) = 14 is found by considering shortening to: | Ub(38,10) = 14 otherwise "
         "adding a parity check bit would contradict: | Ub(39,10) = 15 Bou"
     ),
     ("Bou",),
    ),
    (48, 14, 16, 17,
     (
         "Lb(48,14) = 16 is found by taking a subcode of: | Lb(48,15) = 16 is found by "
         "shortening of: | Lb(49,16) = 16 is found by adding a parity check bit to: | Lb(48,16) "
         "= 15 FB"
     ),
     ("FB",),
     (
         "Ub(48,14) = 17 follows by a one-step Griesmer bound from: | Ub(30,13) = 8 is found by "
         "considering shortening to: | Ub(28,11) = 8 otherwise adding a parity check bit would "
         "contradict: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (49, 14, 16, 18,
     (
         "Lb(49,14) = 16 is found by taking a subcode of: | Lb(49,16) = 16 is found by adding a "
         "parity check bit to: | Lb(48,16) = 15 FB"
     ),
     ("FB",),
     (
         "Ub(49,14) = 18 follows by a one-step Griesmer bound from: | Ub(30,13) = 8 is found by "
         "considering shortening to: | Ub(28,11) = 8 otherwise adding a parity check bit would "
         "contradict: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (50, 14, 17, 18,
     (
         "Lb(50,14) = 16 is found by taking a subcode of: | Lb(50,16) = 16 is found by "
         "shortening of: | Lb(51,17) = 16 cy"
     ),
     ("cy",),
     (
         "Ub(50,14) = 18 follows by a one-step Griesmer bound from: | Ub(31,13) = 9 is found by "
         "considering shortening to: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (52, 14, 18, 19,
     (
         "Lb(52,14) = 18 is found by shortening of: | Lb(54,16) = 18 is found by truncation of: "
         "| Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) = 19 LC"
     ),
     ("LC",),
     "Ub(52,14) = 19 is found by considering truncation to: | Ub(51,14) = 18 Ja",
     ("Ja",),
    ),
    (53, 14, 19, 20,
     "Lb(53,14) = 19 is found by shortening of: | Lb(55,16) = 19 LC",
     ("LC",),
     (
         "Ub(53,14) = 20 follows by a one-step Griesmer bound from: | Ub(32,13) = 10 follows by "
         "a one-step Griesmer bound from: | Ub(21,12) = 5 is found by considering shortening to: "
         "| Ub(19,10) = 5 is found by construction B:"
     ),
     (),
    ),
    (56, 14, 20, 21,
     (
         "Lb(56,14) = 20 is found by taking a subcode of: | Lb(56,16) = 20 is found by adding a "
         "parity check bit to: | Lb(55,16) = 19 LC"
     ),
     ("LC",),
     (
         "Ub(56,14) = 21 follows by a one-step Griesmer bound from: | Ub(34,13) = 10 otherwise "
         "adding a parity check bit would contradict: | Ub(35,13) = 11 Ja"
     ),
     ("Ja",),
    ),
    (57, 14, 20, 22,
     (
         "Lb(57,14) = 20 is found by taking a subcode of: | Lb(57,16) = 20 is found by "
         "lengthening of: | Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) "
         "= 19 LC"
     ),
     ("LC",),
     (
         "Ub(57,14) = 22 follows by a one-step Griesmer bound from: | Ub(34,13) = 10 otherwise "
         "adding a parity check bit would contradict: | Ub(35,13) = 11 Ja"
     ),
     ("Ja",),
    ),
    (58, 14, 20, 22,
     (
         "Lb(58,14) = 20 is found by taking a subcode of: | Lb(58,16) = 20 is found by "
         "lengthening of: | Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) "
         "= 19 LC"
     ),
     ("LC",),
     "Ub(58,14) = 22 follows by a one-step Griesmer bound from: | Ub(35,13) = 11 Ja",
     ("Ja",),
    ),
    (59, 14, 21, 22,
     (
         "Lb(59,14) = 21 is found by shortening of: | Lb(63,18) = 21 is found by truncation of: "
         "| Lb(64,18) = 22 XBC"
     ),
     ("XBC",),
     "Ub(59,14) = 22 Ja",
     ("Ja",),
    ),
    (60, 14, 22, 23,
     "Lb(60,14) = 22 is found by shortening of: | Lb(64,18) = 22 XBC",
     ("XBC",),
     "Ub(60,14) = 23 is found by considering truncation to: | Ub(59,14) = 22 Ja",
     ("Ja",),
    ),
    (61, 14, 23, 24,
     (
         "Lb(61,14) = 23 is found by shortening of: | Lb(63,16) = 23 is found by truncation of: "
         "| Lb(64,16) = 24 XBC"
     ),
     ("XBC",),
     (
         "Ub(61,14) = 24 follows by a one-step Griesmer bound from: | Ub(36,13) = 12 is found by "
         "considering shortening to: | Ub(30,7) = 12 otherwise adding a parity check bit would "
         "contradict: | Ub(31,7) = 13 vT3"
     ),
     ("vT3",),
    ),
    (64, 14, 24, 25,
     "Lb(64,14) = 24 is found by taking a subcode of: | Lb(64,16) = 24 XBC",
     ("XBC",),
     (
         "Ub(64,14) = 25 follows by a one-step Griesmer bound from: | Ub(38,13) = 12 otherwise "
         "adding a parity check bit would contradict: | Ub(39,13) = 13 Ja"
     ),
     ("Ja",),
    ),
    (33, 15, 8, 9,
     (
         "Lb(33,15) = 8 is found by taking a subcode of: | Lb(33,17) = 8 is found by shortening "
         "of: | Lb(38,22) = 8 Sh1"
     ),
     ("Sh1",),
     "Ub(33,15) = 9 is found by considering shortening to: | Ub(29,11) = 9 Ja",
     ("Ja",),
    ),
    (34, 15, 9, 10,
     (
         "Lb(34,15) = 9 is found by shortening of: | Lb(35,16) = 9 is found by truncation of: | "
         "Lb(36,16) = 10 Sh1"
     ),
     ("Sh1",),
     (
         "Ub(34,15) = 10 follows by a one-step Griesmer bound from: | Ub(23,14) = 5 follows by a "
         "one-step Griesmer bound from: | Ub(17,13) = 2 is found by considering shortening to: | "
         "Ub(16,12) = 2 is found by construction B:"
     ),
     (),
    ),
    (37, 15, 10, 11,
     (
         "Lb(37,15) = 10 is found by taking a subcode of: | Lb(37,16) = 10 is found by "
         "shortening of: | Lb(42,21) = 10 QR"
     ),
     ("QR",),
     "Ub(37,15) = 11 is found by considering shortening to: | Ub(35,13) = 11 Ja",
     ("Ja",),
    ),
    (38, 15, 11, 12,
     (
         "Lb(38,15) = 11 is found by shortening of: | Lb(47,24) = 11 is found by truncation of: "
         "| Lb(48,24) = 12 QR"
     ),
     ("QR",),
     (
         "Ub(38,15) = 12 follows by a one-step Griesmer bound from: | Ub(25,14) = 6 follows by a "
         "one-step Griesmer bound from: | Ub(18,13) = 3 is found by considering shortening to: | "
         "Ub(17,12) = 3 is found by considering truncation to: | Ub(16,12) = 2 is found by "
         "construction B:"
     ),
     (),
    ),
    (41, 15, 12, 13,
     (
         "Lb(41,15) = 12 is found by taking a subcode of: | Lb(41,17) = 12 is found by "
         "shortening of: | Lb(48,24) = 12 QR"
     ),
     ("QR",),
     "Ub(41,15) = 13 is found by considering shortening to: | Ub(39,13) = 13 Ja",
     ("Ja",),
    ),
    (42, 15, 12, 14,
     (
         "Lb(42,15) = 12 is found by taking a subcode of: | Lb(42,18) = 12 is found by "
         "shortening of: | Lb(48,24) = 12 QR"
     ),
     ("QR",),
     (
         "Ub(42,15) = 14 follows by a one-step Griesmer bound from: | Ub(27,14) = 7 is found by "
         "considering shortening to: | Ub(26,13) = 7 Pu2"
     ),
     ("Pu2",),
    ),
    (43, 15, 13, 14,
     "Lb(43,15) = 13 cy",
     ("cy",),
     (
         "Ub(43,15) = 14 is found by considering shortening to: | Ub(38,10) = 14 otherwise "
         "adding a parity check bit would contradict: | Ub(39,10) = 15 Bou"
     ),
     ("Bou",),
    ),
    (45, 15, 14, 15,
     (
         "Lb(45,15) = 14 is found by shortening of: | Lb(46,16) = 14 is found by adding a parity "
         "check bit to: | Lb(45,16) = 13 DJ"
     ),
     ("DJ",),
     "Ub(45,15) = 15 is found by considering shortening to: | Ub(43,13) = 15 Ja",
     ("Ja",),
    ),
    (46, 15, 15, 16,
     (
         "Lb(46,15) = 14 is found by taking a subcode of: | Lb(46,16) = 14 is found by adding a "
         "parity check bit to: | Lb(45,16) = 13 DJ"
     ),
     ("DJ",),
     (
         "Ub(46,15) = 16 follows by a one-step Griesmer bound from: | Ub(29,14) = 8 follows by a "
         "one-step Griesmer bound from: | Ub(20,13) = 4 follows by a one-step Griesmer bound "
         "from: | Ub(15,12) = 2 is found by considering shortening to: | Ub(8,5) = 2 is found by "
         "construction B:"
     ),
     (),
    ),
    (49, 15, 16, 17,
     (
         "Lb(49,15) = 16 is found by taking a subcode of: | Lb(49,16) = 16 is found by adding a "
         "parity check bit to: | Lb(48,16) = 15 FB"
     ),
     ("FB",),
     (
         "Ub(49,15) = 17 follows by a one-step Griesmer bound from: | Ub(31,14) = 8 is found by "
         "considering shortening to: | Ub(28,11) = 8 otherwise adding a parity check bit would "
         "contradict: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (50, 15, 16, 18,
     (
         "Lb(50,15) = 16 is found by taking a subcode of: | Lb(50,16) = 16 is found by "
         "shortening of: | Lb(51,17) = 16 cy"
     ),
     ("cy",),
     (
         "Ub(50,15) = 18 follows by a one-step Griesmer bound from: | Ub(31,14) = 8 is found by "
         "considering shortening to: | Ub(28,11) = 8 otherwise adding a parity check bit would "
         "contradict: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (51, 15, 16, 18,
     "Lb(51,15) = 16 is found by taking a subcode of: | Lb(51,17) = 16 cy",
     ("cy",),
     (
         "Ub(51,15) = 18 follows by a one-step Griesmer bound from: | Ub(32,14) = 9 is found by "
         "considering shortening to: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (52, 15, 17, 18,
     (
         "Lb(52,15) = 17 is found by shortening of: | Lb(53,16) = 17 is found by truncation of: "
         "| Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) = 19 LC"
     ),
     ("LC",),
     "Ub(52,15) = 18 is found by considering shortening to: | Ub(51,14) = 18 Ja",
     ("Ja",),
    ),
    (53, 15, 18, 19,
     (
         "Lb(53,15) = 18 is found by shortening of: | Lb(54,16) = 18 is found by truncation of: "
         "| Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) = 19 LC"
     ),
     ("LC",),
     (
         "Ub(53,15) = 19 is found by considering shortening to: | Ub(52,14) = 19 is found by "
         "considering truncation to: | Ub(51,14) = 18 Ja"
     ),
     ("Ja",),
    ),
    (54, 15, 19, 20,
     "Lb(54,15) = 19 is found by shortening of: | Lb(55,16) = 19 LC",
     ("LC",),
     (
         "Ub(54,15) = 20 follows by a one-step Griesmer bound from: | Ub(33,14) = 10 follows by "
         "a one-step Griesmer bound from: | Ub(22,13) = 5 follows by a one-step Griesmer bound "
         "from: | Ub(16,12) = 2 is found by construction B:"
     ),
     (),
    ),
    (57, 15, 20, 21,
     (
         "Lb(57,15) = 20 is found by taking a subcode of: | Lb(57,16) = 20 is found by "
         "lengthening of: | Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) "
         "= 19 LC"
     ),
     ("LC",),
     (
         "Ub(57,15) = 21 follows by a one-step Griesmer bound from: | Ub(35,14) = 10 is found by "
         "considering shortening to: | Ub(34,13) = 10 otherwise adding a parity check bit would "
         "contradict: | Ub(35,13) = 11 Ja"
     ),
     ("Ja",),
    ),
    (58, 15, 20, 22,
     (
         "Lb(58,15) = 20 is found by taking a subcode of: | Lb(58,16) = 20 is found by "
         "lengthening of: | Lb(56,16) = 20 is found by adding a parity check bit to: | Lb(55,16) "
         "= 19 LC"
     ),
     ("LC",),
     (
         "Ub(58,15) = 22 follows by a one-step Griesmer bound from: | Ub(35,14) = 10 is found by "
         "considering shortening to: | Ub(34,13) = 10 otherwise adding a parity check bit would "
         "contradict: | Ub(35,13) = 11 Ja"
     ),
     ("Ja",),
    ),
    (59, 15, 20, 22,
     (
         "Lb(59,15) = 20 is found by taking a subcode of: | Lb(59,16) = 20 is found by "
         "shortening of: | Lb(60,17) = 20 CDJ"
     ),
     ("CDJ",),
     (
         "Ub(59,15) = 22 follows by a one-step Griesmer bound from: | Ub(36,14) = 11 is found by "
         "considering shortening to: | Ub(35,13) = 11 Ja"
     ),
     ("Ja",),
    ),
    (60, 15, 21, 22,
     (
         "Lb(60,15) = 21 is found by shortening of: | Lb(63,18) = 21 is found by truncation of: "
         "| Lb(64,18) = 22 XBC"
     ),
     ("XBC",),
     "Ub(60,15) = 22 is found by considering shortening to: | Ub(59,14) = 22 Ja",
     ("Ja",),
    ),
    (61, 15, 22, 23,
     "Lb(61,15) = 22 is found by shortening of: | Lb(64,18) = 22 XBC",
     ("XBC",),
     (
         "Ub(61,15) = 23 is found by considering shortening to: | Ub(60,14) = 23 is found by "
         "considering truncation to: | Ub(59,14) = 22 Ja"
     ),
     ("Ja",),
    ),
    (62, 15, 23, 24,
     (
         "Lb(62,15) = 23 is found by shortening of: | Lb(63,16) = 23 is found by truncation of: "
         "| Lb(64,16) = 24 XBC"
     ),
     ("XBC",),
     (
         "Ub(62,15) = 24 follows by a one-step Griesmer bound from: | Ub(37,14) = 12 follows by "
         "a one-step Griesmer bound from: | Ub(24,13) = 6 follows by a one-step Griesmer bound "
         "from: | Ub(17,12) = 3 is found by considering truncation to: | Ub(16,12) = 2 is found "
         "by construction B:"
     ),
     (),
    ),
    (34, 16, 8, 9,
     (
         "Lb(34,16) = 8 is found by taking a subcode of: | Lb(34,18) = 8 is found by shortening "
         "of: | Lb(38,22) = 8 Sh1"
     ),
     ("Sh1",),
     (
         "Ub(34,16) = 9 follows by a one-step Griesmer bound from: | Ub(24,15) = 4 otherwise "
         "adding a parity check bit would contradict: | Ub(25,15) = 5 Si"
     ),
     ("Si",),
    ),
    (35, 16, 9, 10,
     "Lb(35,16) = 9 is found by truncation of: | Lb(36,16) = 10 Sh1",
     ("Sh1",),
     (
         "Ub(35,16) = 10 follows by a one-step Griesmer bound from: | Ub(24,15) = 4 otherwise "
         "adding a parity check bit would contradict: | Ub(25,15) = 5 Si"
     ),
     ("Si",),
    ),
    (38, 16, 10, 11,
     (
         "Lb(38,16) = 10 is found by taking a subcode of: | Lb(38,17) = 10 is found by "
         "shortening of: | Lb(42,21) = 10 QR"
     ),
     ("QR",),
     "Ub(38,16) = 11 is found by considering shortening to: | Ub(35,13) = 11 Ja",
     ("Ja",),
    ),
    (39, 16, 11, 12,
     (
         "Lb(39,16) = 11 is found by shortening of: | Lb(47,24) = 11 is found by truncation of: "
         "| Lb(48,24) = 12 QR"
     ),
     ("QR",),
     (
         "Ub(39,16) = 12 follows by a one-step Griesmer bound from: | Ub(26,15) = 6 follows by a "
         "one-step Griesmer bound from: | Ub(19,14) = 3 is found by considering shortening to: | "
         "Ub(17,12) = 3 is found by considering truncation to: | Ub(16,12) = 2 is found by "
         "construction B:"
     ),
     (),
    ),
    (42, 16, 12, 13,
     (
         "Lb(42,16) = 12 is found by taking a subcode of: | Lb(42,18) = 12 is found by "
         "shortening of: | Lb(48,24) = 12 QR"
     ),
     ("QR",),
     (
         "Ub(42,16) = 13 follows by a one-step Griesmer bound from: | Ub(28,15) = 6 otherwise "
         "adding a parity check bit would contradict: | Ub(29,15) = 7 Ja"
     ),
     ("Ja",),
    ),
    (43, 16, 12, 14,
     (
         "Lb(43,16) = 12 is found by taking a subcode of: | Lb(43,19) = 12 is found by "
         "shortening of: | Lb(48,24) = 12 QR"
     ),
     ("QR",),
     (
         "Ub(43,16) = 14 follows by a one-step Griesmer bound from: | Ub(28,15) = 6 otherwise "
         "adding a parity check bit would contradict: | Ub(29,15) = 7 Ja"
     ),
     ("Ja",),
    ),
    (44, 16, 13, 14,
     (
         "Lb(44,16) = 12 is found by taking a subcode of: | Lb(44,20) = 12 is found by "
         "shortening of: | Lb(48,24) = 12 QR"
     ),
     ("QR",),
     "Ub(44,16) = 14 follows by a one-step Griesmer bound from: | Ub(29,15) = 7 Ja",
     ("Ja",),
    ),
    (46, 16, 14, 15,
     "Lb(46,16) = 14 is found by adding a parity check bit to: | Lb(45,16) = 13 DJ",
     ("DJ",),
     "Ub(46,16) = 15 is found by considering shortening to: | Ub(43,13) = 15 Ja",
     ("Ja",),
    ),
    (47, 16, 15, 16,
     "Lb(47,16) = 14 is found by shortening of: | Lb(48,17) = 14 BZ",
     ("BZ",),
     (
         "Ub(47,16) = 16 follows by a one-step Griesmer bound from: | Ub(30,15) = 8 follows by a "
         "one-step Griesmer bound from: | Ub(21,14) = 4 follows by a one-step Griesmer bound "
         "from: | Ub(16,13) = 2 is found by considering shortening to: | Ub(8,5) = 2 is found by "
         "construction B:"
     ),
     (),
    ),
    (50, 16, 16, 17,
     "Lb(50,16) = 16 is found by shortening of: | Lb(51,17) = 16 cy",
     ("cy",),
     (
         "Ub(50,16) = 17 follows by a one-step Griesmer bound from: | Ub(32,15) = 8 is found by "
         "considering shortening to: | Ub(28,11) = 8 otherwise adding a parity check bit would "
         "contradict: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (51, 16, 16, 18,
     "Lb(51,16) = 16 is found by taking a subcode of: | Lb(51,17) = 16 cy",
     ("cy",),
     (
         "Ub(51,16) = 18 follows by a one-step Griesmer bound from: | Ub(32,15) = 8 is found by "
         "considering shortening to: | Ub(28,11) = 8 otherwise adding a parity check bit would "
         "contradict: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (52, 16, 16, 18,
     (
         "Lb(52,16) = 16 is found by taking a subcode of: | Lb(52,17) = 16 is found by "
         "shortening of: | Lb(56,21) = 16 is found by adding a parity check bit to: | Lb(55,21) "
         "= 15 cy"
     ),
     ("cy",),
     (
         "Ub(52,16) = 18 follows by a one-step Griesmer bound from: | Ub(33,15) = 9 is found by "
         "considering shortening to: | Ub(29,11) = 9 Ja"
     ),
     ("Ja",),
    ),
    (53, 16, 17, 18,
     (
         "Lb(53,16) = 17 is found by truncation of: | Lb(56,16) = 20 is found by adding a parity "
         "check bit to: | Lb(55,16) = 19 LC"
     ),
     ("LC",),
     "Ub(53,16) = 18 is found by considering shortening to: | Ub(51,14) = 18 Ja",
     ("Ja",),
    ),
    (54, 16, 18, 19,
     (
         "Lb(54,16) = 18 is found by truncation of: | Lb(56,16) = 20 is found by adding a parity "
         "check bit to: | Lb(55,16) = 19 LC"
     ),
     ("LC",),
     (
         "Ub(54,16) = 19 is found by considering shortening to: | Ub(52,14) = 19 is found by "
         "considering truncation to: | Ub(51,14) = 18 Ja"
     ),
     ("Ja",),
    ),
    (55, 16, 19, 20,
     "Lb(55,16) = 19 LC",
     ("LC",),
     (
         "Ub(55,16) = 20 follows by a one-step Griesmer bound from: | Ub(34,15) = 10 follows by "
         "a one-step Griesmer bound from: | Ub(23,14) = 5 follows by a one-step Griesmer bound "
         "from: | Ub(17,13) = 2 is found by considering shortening to: | Ub(16,12) = 2 is found "
         "by construction B:"
     ),
     (),
    ),
    (58, 16, 20, 21,
     (
         "Lb(58,16) = 20 is found by lengthening of: | Lb(56,16) = 20 is found by adding a "
         "parity check bit to: | Lb(55,16) = 19 LC"
     ),
     ("LC",),
     (
         "Ub(58,16) = 21 follows by a one-step Griesmer bound from: | Ub(36,15) = 10 is found by "
         "considering shortening to: | Ub(34,13) = 10 otherwise adding a parity check bit would "
         "contradict: | Ub(35,13) = 11 Ja"
     ),
     ("Ja",),
    ),
    (59, 16, 20, 22,
     "Lb(59,16) = 20 is found by shortening of: | Lb(60,17) = 20 CDJ",
     ("CDJ",),
     (
         "Ub(59,16) = 22 follows by a one-step Griesmer bound from: | Ub(36,15) = 10 is found by "
         "considering shortening to: | Ub(34,13) = 10 otherwise adding a parity check bit would "
         "contradict: | Ub(35,13) = 11 Ja"
     ),
     ("Ja",),
    ),
    (60, 16, 20, 22,
     "Lb(60,16) = 20 is found by taking a subcode of: | Lb(60,17) = 20 CDJ",
     ("CDJ",),
     (
         "Ub(60,16) = 22 follows by a one-step Griesmer bound from: | Ub(37,15) = 11 is found by "
         "considering shortening to: | Ub(35,13) = 11 Ja"
     ),
     ("Ja",),
    ),
    (61, 16, 21, 22,
     (
         "Lb(61,16) = 21 is found by shortening of: | Lb(63,18) = 21 is found by truncation of: "
         "| Lb(64,18) = 22 XBC"
     ),
     ("XBC",),
     "Ub(61,16) = 22 is found by considering shortening to: | Ub(59,14) = 22 Ja",
     ("Ja",),
    ),
    (62, 16, 22, 23,
     "Lb(62,16) = 22 is found by shortening of: | Lb(64,18) = 22 XBC",
     ("XBC",),
     (
         "Ub(62,16) = 23 is found by considering shortening to: | Ub(60,14) = 23 is found by "
         "considering truncation to: | Ub(59,14) = 22 Ja"
     ),
     ("Ja",),
    ),
    (63, 16, 23, 24,
     "Lb(63,16) = 23 is found by truncation of: | Lb(64,16) = 24 XBC",
     ("XBC",),
     (
         "Ub(63,16) = 24 follows by a one-step Griesmer bound from: | Ub(38,15) = 12 follows by "
         "a one-step Griesmer bound from: | Ub(25,14) = 6 follows by a one-step Griesmer bound "
         "from: | Ub(18,13) = 3 is found by considering shortening to: | Ub(17,12) = 3 is found "
         "by considering truncation to: | Ub(16,12) = 2 is found by construction B:"
     ),
     (),
    ),
)


# Reference keys that codetables.de CITES but never EXPANDS: its own reference list renders them
# with an empty body on every page that uses them. 16 cells (all on the lower bound) are affected.
# They are kept, not dropped, because the thing that matters for an honest baseline -- the bound
# VALUE -- is stated on the cell's own authoritative page and carries a full derivation chain; it is
# only the bibliography that is missing, and it is missing UPSTREAM. Anything that wants to hold the
# stricter "every bound traces to a named paper" line can filter on this set rather than trusting a
# citation string that we would otherwise have had to invent.
_UNEXPANDED_UPSTREAM_KEYS: frozenset[str] = frozenset({"Sh1", "cy"})


def cells_with_unexpanded_reference() -> list[BestKnown]:
    """The registry cells whose bound cites a key codetables.de never expands (see above)."""
    return [
        BEST_KNOWN[(n, k)]
        for n, k, _lo, _up, _ld, lk, _ud, uk in _OPEN_CELLS
        if _UNEXPANDED_UPSTREAM_KEYS & set(lk + uk)
    ]


def _cite(derivation: str, keys: tuple[str, ...]) -> str:
    """A bound's full provenance: how codetables.de derives it, plus the papers it cites.

    ``keys`` are the literature keys that the cell's own page lists for this bound, so a missing
    key is a parse bug, not something to paper over -- resolve it strictly.
    """
    if not keys:
        return derivation
    cites = "; ".join(f"[{key}] {_REFERENCES[key]}" for key in keys)
    return f"{derivation} -- {cites}"


def _build_registry() -> dict[tuple[int, int], BestKnown]:
    reg: dict[tuple[int, int], BestKnown] = {}
    for n, row in _OPTIMAL_SMALL_N.items():
        for i, d in enumerate(row):
            k = i + 1
            note = f"exactly determined: optimal binary [{n},{k},{d}] code"
            reg[(n, k)] = BestKnown(n, k, d, d, note, note)
    for (n, k), (d, note) in _CLOSED_ANCHORS.items():
        reg[(n, k)] = BestKnown(n, k, d, d, note, note)

    # ---- GENUINELY OPEN CELLS (lower < upper), each read off its own codetables.de page. ----
    for n, k, lower, upper, lo_deriv, lo_keys, up_deriv, up_keys in _OPEN_CELLS:
        reg[(n, k)] = BestKnown(
            n, k, lower, upper, _cite(lo_deriv, lo_keys), _cite(up_deriv, up_keys)
        )
    return reg


BEST_KNOWN: dict[tuple[int, int], BestKnown] = _build_registry()


class UnsourcedCellError(ValueError):
    """Raised for an [n, k] cell with no sourced best-known record.

    A cell whose record we cannot cite is not ready to run: without a real baseline, any
    ``is_improvement`` verdict is meaningless.
    """


def lookup(n: int, k: int) -> BestKnown | None:
    """The sourced record for [n, k], or None if we cannot cite one."""
    return BEST_KNOWN.get((int(n), int(k)))


def open_cells(max_k: int = MAX_EXHAUSTIVE_K) -> list[BestKnown]:
    """Sourced cells with a real gap (lower < upper) that this verifier can actually check.

    Sorted cheapest-to-verify first: verification cost is 2^k, independent of n.
    """
    cells = [r for r in BEST_KNOWN.values() if r.is_open and r.k <= max_k]
    return sorted(cells, key=lambda r: (r.k, r.n))


# --------------------------------------------------------------------------- #
# The Problem
# --------------------------------------------------------------------------- #
class ECCProblem:
    """One [n, k] cell of the binary-linear-code table. Implements `Problem`."""

    def __init__(self, n: int, k: int) -> None:
        n, k = int(n), int(k)
        if not 0 < k <= n:
            raise ValueError(f"require 0 < k <= n, got n={n}, k={k}")
        if k > MAX_EXHAUSTIVE_K:
            raise ValueError(
                f"k={k} exceeds the exact verifier's exhaustive limit "
                f"({MAX_EXHAUSTIVE_K}). Without a full 2^k enumeration there is no honest "
                "witness, so this cell cannot be run."
            )
        record = lookup(n, k)
        if record is None:
            raise UnsourcedCellError(
                f"no sourced best-known record for [{n},{k}] — this cell is NOT ready to run. "
                f"Look it up on codetables.de and add it to {__name__}.BEST_KNOWN with a "
                "citation for both the lower and the upper bound."
            )
        self.n = n
        self.k = k
        self.record = record
        self.name = f"ecc[{n},{k}]"

    # -- prompt surface ----------------------------------------------------- #
    def describe(self) -> str:
        r = self.record
        status = (
            f"OPEN: nobody has achieved d={r.upper}, and no code with d>{r.upper} exists "
            f"(proven). Closing this gap is a publishable result."
            if r.is_open
            else f"CLOSED: d={r.lower} is proven optimal. No improvement is possible."
        )
        return f"""\
TARGET: a binary linear code with block length n={self.n} and dimension k={self.k}.

OBJECTIVE: maximise the minimum distance d — the smallest Hamming weight among the 2^{self.k}-1
nonzero codewords. Higher d is strictly better.

CURRENT RECORD
  best known (lower bound):  d = {r.lower}   [{r.lower_source}]
  proven upper bound:        d = {r.upper}   [{r.upper_source}]
  status: {status}
  To count as a discovery your code must achieve d >= {r.lower + 1}.

CANDIDATE FORMAT
  Return a generator matrix G as a {self.k} x {self.n} numpy array of 0/1 (dtype uint8), or a list
  of such matrices. The code is the row space of G over GF(2).

HARD CONSTRAINTS
  - G must be exactly {self.k} rows by {self.n} columns.
  - G must have FULL RANK {self.k} over GF(2) (rows linearly independent mod 2). A rank-deficient G
    is not a [{self.n},{self.k}] code at all and scores nothing.
  - Deterministic: seed any RNG explicitly.

THE VIEW THAT MAKES THIS TRACTABLE
  Think of G by its COLUMNS, not its rows: G is {self.n} column vectors c_1..c_{self.n} drawn from
  GF(2)^{self.k}. For a nonzero message u, the codeword uG has weight |{{j : <u, c_j> = 1}}|, so

        d = min over the 2^{self.k}-1 nonzero u of  #{{ j : <u, c_j> = 1 }}

  i.e. you are choosing {self.n} points in GF(2)^{self.k} so that NO nonzero hyperplane through the
  origin contains too many of them. Full rank == the columns span GF(2)^{self.k}. Good codes in this
  regime come from structure, not from randomness: quasi-cyclic / circulant blocks, cyclic codes
  from a generator polynomial, shortened or punctured BCH codes, and column multisets with a
  prescribed symmetry group (that is how the current record here was set).
"""

    # -- verifier ----------------------------------------------------------- #
    def _invalid(self, reason: str) -> Verdict:
        return Verdict(
            valid=False,
            score=NEG_INF,
            detail={"reason": reason, "target": [self.n, self.k], "n": self.n, "k": self.k},
        )

    def verify(self, candidate: Candidate) -> Verdict:
        """Exact, deterministic, and total: a mutated program can emit literally anything, so
        every failure path returns Verdict(valid=False, score=-inf) instead of raising."""
        try:
            return self._verify(candidate)
        except Exception as exc:  # noqa: BLE001 — garbage in, verdict out. Never raise.
            return self._invalid(f"candidate rejected: {type(exc).__name__}: {exc}")

    def _verify(self, candidate: Candidate) -> Verdict:
        g = self._coerce(candidate)
        if isinstance(g, str):
            return self._invalid(g)

        valid, reason = is_valid_generator(g)
        if not valid:
            return self._invalid(f"invalid generator: {reason}")

        # The exact verifier: exhaustive enumeration of all 2^k - 1 nonzero codewords.
        result = compute_min_distance(g)
        d = result.get("min_distance")
        witness = result.get("witness_codeword")
        message = result.get("witness_message")
        if d is None or witness is None or message is None:
            return self._invalid(
                str(result.get("notes") or "verifier could not certify a distance")
            )

        # Independent re-check on a second code path. A distance whose achieving codeword does not
        # reproduce is a bug, not a result — and a result without a witness is not a result.
        recheck = recompute_distance_of_witness(g, message)
        if (
            not recheck.get("ok")
            or recheck.get("weight") != int(d)
            or list(recheck.get("recomputed_codeword") or []) != list(witness)
        ):
            return self._invalid(
                "witness re-check failed: the claimed distance has no valid achieving codeword"
            )

        return Verdict(
            valid=True,
            score=float(d),
            detail={
                "target": [self.n, self.k],
                "n": self.n,
                "k": self.k,
                "min_distance": int(d),
                "witness_codeword": list(witness),
                "witness_message": list(message),
                "witness_weight": int(recheck["weight"]),
                "generator_matrix": g.astype(int).tolist(),
                "method": "exhaustive_enumeration",
                "verification_method": "exhaustive_enumeration",
                "construction_source": "program_candidate",
                "codewords_enumerated": result.get("codewords_enumerated"),
                "best_known_lower": self._baseline(),
                "best_known_upper": self.record.upper,
                "best_known_source": self.record.lower_source,
                "beats_best_known": int(d) > self._baseline(),
            },
        )

    def _coerce(self, candidate: Candidate) -> np.ndarray | str:
        """Coerce a candidate to a k x n GF(2) matrix, or return a reason string."""
        if candidate is None:
            return "candidate is None"
        if isinstance(candidate, (str, bytes, dict, set)):
            return f"candidate is a {type(candidate).__name__}, not a matrix"
        arr = np.asarray(candidate)
        if arr.dtype == object or arr.dtype.kind not in "biuf":
            return f"candidate is not a numeric matrix (dtype {arr.dtype})"
        if arr.ndim != 2:
            return f"candidate must be a 2-D k x n matrix, got ndim={arr.ndim}"
        if arr.shape != (self.k, self.n):
            return f"candidate shape {tuple(arr.shape)} != target ({self.k}, {self.n})"
        if arr.dtype.kind == "f":
            if not np.all(np.isfinite(arr)):
                return "candidate contains NaN/inf"
            if not np.all(arr == np.floor(arr)):
                return "candidate has non-integer entries; a GF(2) matrix is required"
        if np.any(np.abs(arr) > 2**31):
            return "candidate has out-of-range entries"
        return np.asarray(arr, dtype=np.int64) % 2

    # -- the record --------------------------------------------------------- #
    def _baseline(self) -> int:
        """The number to beat. Conservative by construction: the HIGHEST claim from any source
        wins, so we can never bank a 'record' against a baseline that is too low."""
        claims = [self.record.lower]
        repo = best_known_distance(self.n, self.k)
        if repo is not None:
            claims.append(int(repo))
        return max(claims)

    def best_known(self) -> float:
        return float(self._baseline())

    def is_improvement(self, verdict: Verdict) -> bool:
        try:
            return self._is_improvement(verdict)
        except Exception:  # noqa: BLE001 — an unjudgeable verdict is not an improvement.
            return False

    def _is_improvement(self, verdict: Verdict) -> bool:
        if verdict is None or not getattr(verdict, "valid", False):
            return False
        score = getattr(verdict, "score", NEG_INF)
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            return False
        detail: dict[str, Any] = getattr(verdict, "detail", None) or {}

        # Judge only candidates verified for THIS cell.
        if list(detail.get("target") or []) != [self.n, self.k]:
            return False
        d = detail.get("min_distance")
        if not isinstance(d, int) or isinstance(d, bool):
            return False
        if int(float(score)) != d:  # score and evidence must agree
            return False

        # 1. The existing anti-self-deception guards. A distance read off a table, or asserted
        #    without an achieving codeword, is not a computation.
        if is_table_lookup_evidence(detail):
            return False
        # 2. Reproducing a tabulated code is a rediscovery, not a discovery.
        if trivial_rediscovery(detail, self.n, self.k, d):
            return False
        # 3. The witness must independently re-check against the submitted generator.
        if not self._witness_holds(detail, d):
            return False
        # 4. It must STRICTLY beat the sourced record.
        if d <= self._baseline():
            return False
        # 5. It must not exceed the PROVEN upper bound. Beating a published nonexistence proof is
        #    overwhelmingly likelier to be our bug than a discovery — refuse it and let a human look.
        if d > self.record.upper:
            return False
        return True

    def _witness_holds(self, detail: dict[str, Any], d: int) -> bool:
        """Re-derive the result from the evidence alone, exactly as a third party would."""
        g = detail.get("generator_matrix")
        message = detail.get("witness_message")
        witness = detail.get("witness_codeword")
        if g is None or message is None or witness is None:
            return False
        arr = np.asarray(g)
        if arr.ndim != 2 or arr.shape != (self.k, self.n):
            return False
        valid, _ = is_valid_generator(arr)
        if not valid:
            return False
        recheck = recompute_distance_of_witness(arr, list(message))
        if not recheck.get("ok"):
            return False
        if recheck.get("weight") != d:
            return False
        if list(recheck.get("recomputed_codeword") or []) != list(witness):
            return False
        # The witness proves d <= its weight. That the code has NO lighter codeword is what the
        # exhaustive enumeration establishes — so re-run it; it is cheap and it is the whole claim.
        recomputed = compute_min_distance(arr)
        return recomputed.get("min_distance") == d

    # -- seeds -------------------------------------------------------------- #
    def seed_programs(self) -> list[str]:
        return [
            _PRELUDE.format(n=self.n, k=self.k) + body
            for body in (
                _SEED_FAMILY_SWEEP,          # first: it is the SHAPE every descendant should copy
                _SEED_NAMED_CONSTRUCTIONS,
                _SEED_COLUMN_MULTISET,
                _SEED_QUASI_CYCLIC,
                _SEED_CYCLIC_POLYNOMIAL,
                _SEED_GREEDY_COLUMN_SWAP,
                _SEED_SYSTEMATIC_RANDOM,
            )
        ]


# --------------------------------------------------------------------------- #
# Seed programs — self-contained Python (numpy + stdlib only), each defines build().
# These are the LLM's mutation surface, so they are written to be READ and EDITED.
# --------------------------------------------------------------------------- #
_PRELUDE = '''\
import itertools
import time

import numpy as np

N = {n}   # block length
K = {k}   # dimension


def gf2_rank(a):
    """Rank over GF(2)."""
    a = (np.asarray(a, dtype=np.int64) % 2).copy()
    rows, cols = a.shape
    r = 0
    for c in range(cols):
        piv = -1
        for i in range(r, rows):
            if a[i, c]:
                piv = i
                break
        if piv < 0:
            continue
        a[[r, piv]] = a[[piv, r]]
        for i in range(rows):
            if i != r and a[i, c]:
                a[i] = (a[i] + a[r]) % 2
        r += 1
        if r == rows:
            break
    return r


_MSG_CACHE = {{}}


def all_messages(k):
    """All 2^k - 1 nonzero messages as a (2^k - 1) x k bit matrix. Cached."""
    m = _MSG_CACHE.get(k)
    if m is None:
        idx = np.arange(1, 2 ** k, dtype=np.int64)
        m = (idx[:, None] >> np.arange(k, dtype=np.int64)[None, :]) & 1
        _MSG_CACHE[k] = m
    return m


def min_distance(g):
    """Minimum distance = the smallest nonzero codeword weight (exact, same definition the
    verifier uses). Vectorised over all 2^k - 1 messages in one matmul — use it freely."""
    g = np.asarray(g, dtype=np.int64) % 2
    k = g.shape[0]
    weights = ((all_messages(k) @ g) % 2).sum(axis=1)
    return int(weights.min())


def take_rank_k(rows, k=K, n=N):
    """Greedily keep k independent rows (a subcode); top up with unit rows if short."""
    rows = np.asarray(rows, dtype=np.int64) % 2
    chosen = []
    for row in rows:
        if len(chosen) == k:
            break
        if gf2_rank(np.array(chosen + [row], dtype=np.int64)) == len(chosen) + 1:
            chosen.append(row)
    i = 0
    while len(chosen) < k and i < n:
        e = np.zeros(n, dtype=np.int64)
        e[i] = 1
        if gf2_rank(np.array(chosen + [e], dtype=np.int64)) == len(chosen) + 1:
            chosen.append(e)
        i += 1
    if len(chosen) != k:
        return None
    return (np.array(chosen, dtype=np.int64) % 2).astype(np.uint8)


def fit(g, n=N, k=K):
    """Adapt any generator matrix to the target [n, k] shape.

    Columns: puncture if too long, extend by repeating columns if too short.
    Rows:    take a rank-k subcode (or top up to rank k).
    Returns None if it cannot be made into a valid [n, k] generator.
    """
    g = np.asarray(g, dtype=np.int64) % 2
    if g.ndim != 2 or g.size == 0:
        return None
    cols = g.shape[1]
    if cols > n:
        g = g[:, :n]
    elif cols < n:
        extra = np.array([g[:, i % cols] for i in range(n - cols)], dtype=np.int64).T
        g = np.hstack([g, extra])
    out = take_rank_k(g, k, n)
    if out is None or out.shape != (k, n) or gf2_rank(out) != k:
        return None
    return out


def from_columns(cols, k=K, n=N):
    """Build G from an explicit list of n column vectors in GF(2)^k."""
    g = (np.array(cols, dtype=np.int64).T % 2)
    if g.shape != (k, n) or gf2_rank(g) != k:
        return None
    return g.astype(np.uint8)


def int_to_vec(c, k=K):
    """Integer 1..2^k-1 -> its binary column vector in GF(2)^k."""
    return [(c >> b) & 1 for b in range(k)]


'''

_SEED_FAMILY_SWEEP = '''\
def build():
    """FAMILY SWEEP -- yields HUNDREDS of candidates from one call. This is the shape to copy.

    Do not guess one matrix. Parameterise a construction and yield the whole sweep: the verifier
    checks ~20 codes/sec, and a program that emits one object wastes the machine. Point-search also
    simply does not work on these landscapes -- the neighbourhood of the best-known code is a
    deceptive local optimum. Sweeping a family's PARAMETERS is what crosses the valley.

    Here: systematic G = [I_K | A], A a K x R matrix built from a cyclic seed row. Sweep the seed
    polynomial and the per-row rotation step -- a 2-parameter family; every member is a different
    code.

    BUDGET, and it is not optional. Two hard limits bound how much one program may emit:
      * generation must finish well inside the sandbox timeout (~10s), and
      * the verifier costs ~46 ms per code, so ~300 candidates is already ~14s of checking.
    Exhausting this family would emit ~49,000 codes and take ~94s just to BUILD -- the sandbox would
    kill it and the whole program would score nothing. Emitting many BAD candidates is also worse
    than emitting a few good ones. So: sweep widely, but stride the space and stop at the budget.
    """
    r = N - K
    if r <= 0:
        return
    budget = 400
    emitted = 0
    # High-weight seed rows first: a dense A-block spreads the parity checks, which is where the
    # distance comes from. Stride the polynomial space rather than exhaust it.
    polys = sorted(range(1, 1 << min(14, r)), key=lambda p: (-bin(p).count("1"), p))
    for poly in polys:
        row = [(poly >> b) & 1 for b in range(r)]
        for step in (1, 2, 3, 5):                  # how far each successive row rotates
            a = np.array([np.roll(row, (i * step) % r) for i in range(K)], dtype=np.int64) % 2
            g = np.hstack([np.eye(K, dtype=np.int64), a]) % 2
            if gf2_rank(g) == K:
                yield g.astype(np.uint8)
                emitted += 1
                if emitted >= budget:
                    return
'''

_SEED_NAMED_CONSTRUCTIONS = '''\
def hamming(r):
    """[2^r - 1, 2^r - 1 - r, 3]."""
    cols = [int_to_vec(c, r) for c in range(1, 2 ** r)]
    h = np.array(cols, dtype=np.int64).T
    ident = {1 << b for b in range(r)}
    order = [c for c in range(1, 2 ** r) if c not in ident] + [1 << b for b in range(r)]
    h = np.array([int_to_vec(c, r) for c in order], dtype=np.int64).T
    kk = 2 ** r - 1 - r
    a = h[:, :kk]
    return np.hstack([np.eye(kk, dtype=np.int64), a.T]) % 2


def extended_hamming(r):
    """[2^r, 2^r - 1 - r, 4] — Hamming plus an overall parity bit."""
    g = hamming(r)
    return np.hstack([g, (g.sum(axis=1) % 2).reshape(-1, 1)])


def simplex(r):
    """[2^r - 1, r, 2^(r-1)] — every nonzero codeword has the SAME weight."""
    return np.array([int_to_vec(c, r) for c in range(1, 2 ** r)], dtype=np.int64).T % 2


def reed_muller_rm1(m):
    """RM(1, m) = [2^m, m + 1, 2^(m-1)]."""
    pts = list(itertools.product((0, 1), repeat=m))
    rows = [np.ones(2 ** m, dtype=np.int64)]
    for i in range(m):
        rows.append(np.array([p[i] for p in pts], dtype=np.int64))
    return np.array(rows, dtype=np.int64) % 2


def repetition(n):
    """[n, 1, n]."""
    return np.ones((1, n), dtype=np.int64)


def parity_check(k):
    """[k + 1, k, 2] — G = [I_k | 1]."""
    return np.hstack([np.eye(k, dtype=np.int64), np.ones((k, 1), dtype=np.int64)])


def build():
    """The classical constructions, each ADAPTED to the target [n, k] by shortening,
    puncturing and column-extension. These are the ancestors — recombine them."""
    out = []
    for r in range(2, 8):
        for fn in (hamming, extended_hamming, simplex):
            try:
                out.append(fit(fn(r)))
            except Exception:
                pass
    for m in range(2, 8):
        try:
            out.append(fit(reed_muller_rm1(m)))
        except Exception:
            pass
    out.append(fit(repetition(N)))
    out.append(fit(parity_check(K)))
    return [g for g in out if g is not None]
'''

_SEED_COLUMN_MULTISET = '''\
def build():
    """Columns-first. G is N columns chosen from GF(2)^K \\\\ {0}; the minimum distance is

        d = min over nonzero u of  #{j : <u, c_j> = 1}

    so the game is to spread the columns so that no hyperplane through the origin swallows too
    many of them. Vary WHICH columns you take — that is the whole search."""
    out = []
    pool = list(range(1, 2 ** K))

    # (a) the first N nonzero vectors — a punctured simplex code
    out.append(from_columns([int_to_vec(c) for c in pool[:N]]))

    # (b) heaviest columns first: high-weight columns hit more hyperplanes
    by_weight = sorted(pool, key=lambda c: (-bin(c).count("1"), c))
    out.append(from_columns([int_to_vec(c) for c in by_weight[:N]]))

    # (c) evenly spaced through the pool — a crude 'spread'
    if len(pool) >= N:
        step = len(pool) // N
        picked = [pool[(i * step) % len(pool)] for i in range(N)]
        if len(set(picked)) == N:
            out.append(from_columns([int_to_vec(c) for c in picked]))

    return [g for g in out if g is not None]
'''

_SEED_QUASI_CYCLIC = '''\
def circulant(first_row, k=K):
    """k x k circulant: row i is first_row rotated right by i."""
    return np.array(
        [np.roll(np.asarray(first_row, dtype=np.int64), i) for i in range(k)],
        dtype=np.int64,
    ) % 2


def build():
    """Quasi-cyclic: G = [C_1 | C_2 | ... | C_m | R], each C a K x K circulant. QC codes hold a
    large share of the current records in this region — mutate the circulant seed rows."""
    out = []
    blocks = N // K
    rest = N - blocks * K
    seeds = (
        [1, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
    )
    for offset in range(len(seeds)):
        cols = []
        for b in range(blocks):
            row = np.zeros(K, dtype=np.int64)
            pat = seeds[(b + offset) % len(seeds)]
            for i, bit in enumerate(pat):
                if i < K:
                    row[i] = bit
            cols.append(circulant(row))
        if rest:
            tail = np.zeros((K, rest), dtype=np.int64)
            for i in range(K):
                for j in range(rest):
                    tail[i, j] = (i + j + offset) % 2
            cols.append(tail)
        if not cols:
            continue
        g = np.hstack(cols) % 2
        if g.shape == (K, N) and gf2_rank(g) == K:
            out.append(g.astype(np.uint8))
        else:
            fitted = fit(g)
            if fitted is not None:
                out.append(fitted)
    return out
'''

_SEED_CYCLIC_POLYNOMIAL = '''\
def cyclic_from_poly(poly, n=N, k=K):
    """Cyclic [n, k] code: rows are the n - deg shifts of the generator polynomial g(x),
    where deg g(x) = n - k. poly is a bit list, lowest degree first."""
    deg = len(poly) - 1
    if deg != n - k:
        return None
    rows = []
    for i in range(k):
        row = np.zeros(n, dtype=np.int64)
        for j, bit in enumerate(poly):
            row[i + j] = bit % 2
        rows.append(row)
    g = np.array(rows, dtype=np.int64) % 2
    if gf2_rank(g) != k:
        return None
    return g.astype(np.uint8)


def build():
    """Cyclic codes from a generator polynomial of degree N-K. Classic BCH-style structure:
    mutate the polynomial's coefficients (and try its reciprocal)."""
    out = []
    deg = N - K
    polys = []
    # a few dense/structured degree-(N-K) polynomials, always with constant term 1
    polys.append([1] + [(i % 2) for i in range(1, deg)] + [1])
    polys.append([1] + [(1 if i % 3 else 0) for i in range(1, deg)] + [1])
    polys.append([1] + [(1 if (i * i) % 5 < 2 else 0) for i in range(1, deg)] + [1])
    polys.append([1] * (deg + 1))
    for p in polys:
        if len(p) != deg + 1:
            continue
        g = cyclic_from_poly(p)
        if g is not None:
            out.append(g)
        g = cyclic_from_poly(list(reversed(p)))
        if g is not None:
            out.append(g)
    return out
'''

_SEED_GREEDY_COLUMN_SWAP = '''\
# Wall-clock budget. The sandbox hard-kills a slow program, and a killed program scores NOTHING,
# so a search seed must bound its own runtime rather than its iteration count — the cost of one
# iteration scales as 2^K, so a fixed iteration count that is fine at K=10 is fatal at K=16.
BUDGET_S = 5.0


def build():
    """Hill-climb the column multiset: swap one column at a time, keep the swap unless it makes
    the minimum distance worse (ties are kept, so the walk can cross plateaus).

    This is a real, deterministic baseline — the engine has to beat IT, not just beat random.
    Obvious ways to improve it: choose the column to replace from the SUPPORT of a
    minimum-weight codeword instead of uniformly at random; anneal instead of hill-climbing;
    or restart from a structured (quasi-cyclic / cyclic) code instead of a random one."""
    deadline = time.monotonic() + BUDGET_S
    rng = np.random.default_rng(20260714)

    # Lookup table: every column vector of GF(2)^K, indexed by its integer encoding.
    vecs = np.array([int_to_vec(c) for c in range(2 ** K)], dtype=np.int64)
    pool = np.arange(1, 2 ** K, dtype=np.int64)

    best_g = None
    best_d = -1
    while time.monotonic() < deadline:
        cols = rng.choice(pool, size=N, replace=True)
        g = vecs[cols].T
        while gf2_rank(g) != K and time.monotonic() < deadline:
            cols[int(rng.integers(N))] = int(rng.choice(pool))
            g = vecs[cols].T
        if gf2_rank(g) != K:
            break
        d = min_distance(g)

        while time.monotonic() < deadline:
            i = int(rng.integers(N))
            old = cols[i]
            cols[i] = int(rng.choice(pool))
            g2 = vecs[cols].T
            if gf2_rank(g2) != K:
                cols[i] = old
                continue
            d2 = min_distance(g2)
            if d2 >= d:            # accept improvements AND lateral moves
                g, d = g2, d2
            else:
                cols[i] = old

        if d > best_d:
            best_d = d
            best_g = (g % 2).astype(np.uint8)

    return [best_g] if best_g is not None else []
'''

_SEED_SYSTEMATIC_RANDOM = '''\
def build():
    """Baseline: G = [I_K | R] for pseudo-random R. Systematic form is always full rank, so this
    always produces a VALID code — it just will not produce a good one. Beat it."""
    out = []
    for seed in (1, 2, 3, 5, 8):
        rng = np.random.default_rng(seed)
        r = rng.integers(0, 2, size=(K, N - K), dtype=np.uint8)
        out.append(np.hstack([np.eye(K, dtype=np.uint8), r]) % 2)
    return out
'''


# --------------------------------------------------------------------------- #
# The recommended first cell — chosen by MEASUREMENT, not by table-reading.
#
# There are 122 open cells now. Two axes decide which to attack first, and they PULL APART:
#   * gap size    -- a gap of exactly 1 means any single-unit improvement CLOSES the cell.
#   * verifier cost -- verification is 2^k per candidate, independent of n. k=10 is ~16x cheaper
#                     than k=14, ~64x cheaper than k=16.
#   * distance from the frontier -- where the EXISTING seed pool lands vs the record. A cell whose
#                     seeds already tie the record starts the search at the open question itself; a
#                     cell whose seeds sit 3-6 below it spends the whole campaign just RE-climbing
#                     to known territory before it can even attempt the open part. And point-search
#                     on these landscapes is deceptive (see the seed docstrings), so that re-climb
#                     is not free.
#
# The naive read of "cheap verifier" points at the k=10 cells. But the seed pool was actually run on
# every gap-1 cell (2^k enumeration, family_sweep/greedy/quasi_cyclic/...), and the cheap cells are
# NOT close to the frontier:
#     k=10:  [35,10] seeds 10 vs record 12 (climb 3);  [36,10] 11 vs 13 (climb 3); ... climb 3-6.
#     k=11:  [36,11] 10 vs 12 (climb 3);  rest climb 3-6.
#     k=12:  [37,12] 10 vs 12 (climb 3);  rest climb 3-6.
# whereas exactly three cells have the seeds ALREADY TIED to the record (climb 1), all with gap 1:
#     [32,14]  seeds reach d=8 == record 8, upper 9.   k=14.
#     [33,15]  seeds reach d=8 == record 8, upper 9.   k=15.
#     [34,16]  seeds reach d=8 == record 8, upper 9.   k=16.
# On any of these the very first generation is at the frontier; a single +1 (d=9) closes the cell.
# Among them [32,14] has the cheapest verifier (k=14), so it is the single best first target: it is
# simultaneously gap-1, climb-1, and the least expensive of the climb-1 set. (Measurements come from
# the wall-clock-budgeted seeds, so the exact seed distances wobble a little run to run; the climb-1
# vs climb-3+ SEPARATION is stable and is the thing that matters.)
#
# If instead you want to exploit raw throughput (~460 cand/sec sustained) rather than a warm start,
# the cheapest-verifier gap-1 cells are [35,10] and [36,10]: 16x more candidates/sec than [32,14],
# at the cost of a 3-unit climb before the open question is even in reach. That is a different bet,
# not a better one.
#
# HONESTY: an open cell means NOBODY KNOWS — it does NOT mean a better code exists. The true optimum
# may equal the lower bound, with the upper bound merely not yet tightened. If the [32,14,9] code
# does not exist, no amount of search finds it, and the engine cannot emit the nonexistence proof
# that would close the cell the other way (that is an argument, not an object). Expected yield on any
# ONE cell is genuinely uncertain; the case for this target is the BASE RATE across many cells.
# --------------------------------------------------------------------------- #
RECOMMENDED_CELL = (32, 14)


def recommended_problem() -> ECCProblem:
    """[32,14]: record d=8, proven upper bound d=9 — the seed pool already ties the record, so the
    search starts at the frontier and only needs +1 to close the cell."""
    return ECCProblem(*RECOMMENDED_CELL)
