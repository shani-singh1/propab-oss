# Math domain: `math_combinatorics` + `coding_theory`

Two deterministic, real-compute domains. Both build an **actual mathematical object**,
compute a property of it, emit an **independently re-checkable witness**, and report an
**honest** comparison to the best-known literature value (below / matches / exceeds).
Neither reads an answer from a table and presents it as a computation, and neither
oversells a match to a known optimum as a discovery.

- Sources: `packages/propab-core/propab/domain_modules/math_combinatorics/`
  and `.../coding_theory/`.
- Tests: `tests/test_math_combinatorics_plugin.py`, `tests/test_coding_theory_plugin.py`.
- Live end-to-end transcript: `artifacts/math_domain_live_e2e.json`.

---

## 1. `math_combinatorics` — cap sets in F_3^n (and Sidon / AP-free analyses)

### What it computes
A **real cap set** in F_3^n: a subset of F_3^n with no three collinear points
(equivalently, no non-trivial solution to x + y + z = 0). The construction is genuine,
never a table value:

- **n ≤ 4** — near-exhaustive, deterministic branch-and-bound (origin-fixed DFS with a
  candidate-set bound), budget `CAP_BB_TIME_BUDGET = 4.0s`.
- **n = 5** — deterministic random-restart greedy (fixed seed).
- **6 ≤ n ≤ 10** — product / tensor construction from fully-computed base caps
  (dims from `_decompose_dim`, favouring the size-20 optimal cap in F_3^4).

Every returned cap is **independently validated** by `is_valid_cap` — an O(|S|²) check
on the ACTUAL point set that the completing point `c = -(a+b) mod 3` of every pair is
never a third member. For large product caps (size > 1000) it validates every base
factor fully and spot-checks a random sample of product points; product validity then
follows from base validity plus the product theorem. The reported size is always the
size of the real set, never a claimed or tabulated number.

### What it CAN discover
- A **genuine beat-best-known**: if a computed, independently-validated cap in F_3^n is
  strictly larger than the tabulated `CAP_SET_BEST_KNOWN[n]`, that is reported as
  `exceeds_best_known` and is discovery-worthy — a real result with a checkable witness.
- **Honest refutations**: a falsifiable size/ratio claim (e.g. "|A_max(7)| ≥ 300")
  that the real computation cannot meet is `refuted`.
- Reproducing a known optimum (F_3^4 = 20) as a real construction with a full witness.

### What it CANNOT discover (stated honestly)
- The construction is a **lower-bound engine**. For 5 ≤ n ≤ 8 the product cap is
  genuinely *below* the best-known value (see table), so it does not currently beat the
  literature there — it reports the honest gap, it does not invent a larger cap.
- It does **not** resolve the hard open problems: the exact maximum cap for n ≥ 7 and,
  above all, the **cap-set growth exponent** (the true value of `a_3(n)^{1/n}`, known
  only to lie between ≈ 2.2202 and 2.756). Those are open and out of reach of a
  bounded construction. Fourier / slice-rank / SAT-style methodologies named in a
  hypothesis are refused as unimplemented rather than faked.
- For n ≤ 4 the B&B *reaches* the optimum size within the budget but does not always
  certify optimality by exhausting the whole tree (only F_3^2 and F_3^3 finish
  completely on typical hardware); `proven_optimal` is reported honestly.

### Supported range and computed vs best-known
`MAX_FULL_CAP_DIM = 10`. Best-known values are **OEIS A090245** (max cap in AG(n,3));
n ≤ 6 are proven maxima, n = 7, 8 are best-known lower bounds. A dimension above 10 is
dropped as unsupported (routes to the default) — never silently re-dimensioned.

| n  | computed cap size | best-known (A090245) | label            |
|----|-------------------|----------------------|------------------|
| 1  | 2                 | 2                    | matches          |
| 2  | 4                 | 4                    | matches          |
| 3  | 9                 | 9                    | matches          |
| 4  | 20                | 20                   | matches          |
| 5  | 38                | 45                   | below            |
| 6  | 80                | 112                  | below            |
| 7  | 180               | 236                  | below            |
| 8  | 400               | 496                  | below            |
| 9  | 800               | — (no table value)   | no_table_value   |
| 10 | 1600              | — (no table value)   | no_table_value   |

(Computed sizes are deterministic under a fixed seed; the table above is reproducible.)

---

## 2. `coding_theory` — binary linear [n, k] codes over GF(2)

### What it computes
The **TRUE minimum distance** of a binary linear code, by exhaustively enumerating all
`2^k − 1` nonzero codewords of an ACTUAL k×n generator matrix over GF(2) and taking the
minimum Hamming weight (exact, since for a linear code min distance = min nonzero
codeword weight). The achieving codeword is emitted as a **witness** and is
**independently recomputed** from the generator and message; a distance whose witness
fails re-checking is refused, never certified.

Constructible named families (each built and distance-checked here):
repetition [n,1,n], single-parity-check [k+1,k,2], identity [k,k,1],
Hamming [2^r−1, 2^r−1−r, 3], extended Hamming [2^r, 2^r−1−r, 4],
simplex [2^r−1, r, 2^{r−1}], first-order Reed–Muller RM(1,m) = [2^m, m+1, 2^{m−1}].
A caller may also pass an explicit generator, or request a systematic random [n,k].

### What it CAN discover
- A **genuine beat-best-known**: an explicit generator whose independently-recomputed
  minimum distance **strictly exceeds** the best-known Brouwer/Grassl lower bound for
  that [n, k], with a re-checked witness, is discovery-worthy.
- **Honest refutations**: a claim "[n,k] has d ≥ D" that the real computation falsifies
  is `refuted` (witness weight < D).
- Reproducing a known optimum ([7,4,3] Hamming, [8,4,4] extended Hamming, [15,4,8]
  simplex, [16,5,8] RM(1,4)) as a real code with a witness — reported honestly as a
  **rediscovery**, not a discovery.

### What it CANNOT discover (stated honestly)
- It is bounded to **k ≤ 16** (`MAX_EXHAUSTIVE_K`): 2^16−1 ≈ 65k codewords is ~2s;
  k = 18/20 cost ~7s/~25s and are refused as an unacceptable per-verification runaway.
  Above k = 16 it **refuses to certify** a distance rather than lie. So it cannot attack
  open gaps that require large k, and the systematic-random path only *finds* a beat by
  luck — it is not an optimizer/search over generators.
- It does not implement SAT / MILP / neural / annealing constructions; hypotheses naming
  those are rejected as unimplemented, not faked.
- Table entries with k > 16 (e.g. length-31 BCH [31,21], [31,26]) exist only as
  best-known **reference** values for rediscovery rejection; they are never enumerated.

### Reference table (rediscovery guard)
`BEST_KNOWN_TABLE` holds 130 entries for n = 2..31, from the Brouwer / Grassl tables of
bounds on the minimum distance of linear codes (M. Grassl, codetables.de, accessed
2026-07; and A. E. Brouwer's classic tables). Every entry satisfies the Griesmer bound
(a necessary consistency check enforced by a test), and every constructible optimal
family this module builds matches its table value **exactly** — no construction
spuriously "beats" the table (also enforced by a test). A wider correct table lets the
engine reject more rediscoveries and honestly recognise a genuine beat.

Verified constructor anchors: [7,4,3], [8,4,4], [15,4,8] (simplex), [15,11,3] (Hamming),
[16,5,8] (RM(1,4)), [16,11,4] (extended Hamming).

---

## How to run

Tests (all green):

```
PYTHONPATH="packages/propab-core;." python -m pytest \
  tests/test_math_combinatorics_plugin.py tests/test_coding_theory_plugin.py -q
# or, by keyword, exactly as the v1 checklist specifies:
PYTHONPATH="packages/propab-core;." python -m pytest tests/ -k "math_comb or coding" -q
```

Compute a single object directly:

```python
from propab.domain_modules.math_combinatorics.verifier import compute_cap_set
compute_cap_set(4)   # -> real F_3^4 cap, size 20, full witness, matches_best_known

from propab.domain_modules.coding_theory.constructors import hamming_code, compute_min_distance
compute_min_distance(hamming_code(3))   # -> [7,4,3], min_distance 3, witness codeword
```

End-to-end through a plugin (build + verify + honest verdict):

```python
from propab.domain_modules.coding_theory.plugin import CodingTheoryPlugin
from propab.domain_modules.coding_theory.verifier import run_coding_experiment
ev = run_coding_experiment({"statement": "The [7,4,3] Hamming code has d >= 3",
                            "test_methodology": "hamming"})
CodingTheoryPlugin().classify_verdict("hamming", ev)   # -> ("inconclusive", ..., 0.4)  # rediscovery-demoted
```

---

## Known limits (true production-v1 gaps)

1. **Lower-bound engines, not optimizers.** Both compute and honestly compare real
   objects; neither *searches* aggressively for a beat. A genuine beat-best-known is
   reportable with a witness, but is not actively hunted, so at the supported small
   parameters the honest verdict is usually "rediscovery" or "below/honest-gap".
2. **The hardest bounds stay open.** The cap-set growth exponent and exact maxima for
   n ≥ 7 are unresolved and out of scope of a bounded construction.
3. **Hard compute caps.** Cap sets to n ≤ 10; codes to k ≤ 16. These bound worst-case
   wall-clock (~4s / ~2s) for production safety; larger parameters are refused, not
   approximated, so the engine never emits an uncertified number.
