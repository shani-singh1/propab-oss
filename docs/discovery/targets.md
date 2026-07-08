# Discovery Targets — math_combinatorics: where Propab could set a FIRST computational record

**Author:** discovery-target research agent
**Date:** 2026-07-08
**Status:** LOCAL / gitignored (confirmed via `git check-ignore`). Not pushed. Intended.

---

## TL;DR

The engine's real, differentiated asset is a **cheap, deterministic witness verifier**:
given an explicit set, checking "is this a valid cap / Sidon / B_h set" is O(|S|^2) or
O(|S|^h) and runs in milliseconds (see `math_combinatorics/verifier.py`:
`is_valid_cap`, `is_sidon_set`). Its *finders* are weak (greedy, random-restart,
product construction) — so it will **not** out-search records set by dedicated
SAT/ILP/distributed efforts.

Therefore the only honest path to a FIRST record is a **specific small OPEN cell**
where (a) the current best value is genuinely uncomputed or is only a *provisional
lower bound improvable by search*, and (b) verifying a witness is trivially cheap.

**TOP PICK:** **A396704 — maximum size of a B_3 set in {0,1}^n** — improve the
published lower bound at **n = 7** (currently `a(7) >= 16`, *not proven optimal*) by
finding an explicit B_3 set of size **17**. A 17-vector witness in {0,1}^7 is verified
in milliseconds; the search space (128 points) is small; the current record is a
provisional local-search bound, not an exhaustive one. **Primary risk: this exact
sequence is being actively record-chased by one competitor (William Blair,
"verified-combinatorics"), created 2026-06-28.**

Runner-up (uncontested, but exact-optimality is the hard part): **A385931 — "weak B_3
Golomb ruler", next uncomputed term a(11).**

---

## Part 1 — Ranked shortlist (candidate target families)

Ranking criterion = (cheap deterministic witness) x (record genuinely open/improvable)
x (reachable in modest compute) x (low competition). Every "best-known value" below
carries a source URL.

### #1 — Max B_3 set in {0,1}^n (OEIS A396704) — **lower-bound record via witness**
- **Object:** subset S of the binary cube {0,1}^n (added componentwise over the
  integers) such that all threefold sums a+b+c are distinct as vectors in {0,1,2,3}^n.
  Maximize |S|.
- **Current best-known:** exact & proven optimal only for n=0..6:
  `1,2,3,4,6,8,11`. For n>=7 only provisional lower bounds:
  **`a(7) >= 16`, `a(8) >= 19`** — from *iterated local search, NOT proven optimal*.
  Source: OEIS A396704 (created 2026-06-28), https://oeis.org/A396704
- **Why a first/improved record is reachable:** the n=7 lower bound is a local-search
  artifact in a 128-point space; a smarter search (ILS/SAT/ILP restart) plausibly finds
  size 17. Beating a *provisional* published bound is a clean, cheaply-verifiable record.
  Witness verification = check all C(|S|+2,3)-ish threefold sums distinct → milliseconds.
- **Competition (against):** the SAME family is being actively colonized by William
  Blair's "verified-combinatorics" project; a(7)>=16 is his ILS result.
  https://github.com/willblair0708/verified-combinatorics/tree/main/b3-binary

### #2 — "Weak B_3 Golomb ruler" (OEIS A385931) — **next uncomputed exact term**
- **Object:** least integer L such that some n-element set in [0,L] has *all sums of any
  three distinct elements distinct* ("weak B_3"). Minimize L.
- **Current best-known:** exact for n=1..10: `0,1,2,3,7,13,22,39,69,113`.
  **a(11) is uncomputed** (marked `hard,more`). Source: OEIS A385931,
  https://oeis.org/A385931
- **Why reachable:** no visible competitor. Growth ratio ~1.65-1.77 ⇒ a(11) ≈ 185-195.
  Witness = an 11-element set; verify C(11,3)=165 triple-sums distinct → sub-millisecond.
  **Caveat:** a first *exact* value needs the exhaustive "no shorter ruler exists" proof
  (the expensive, not-cheaply-verifiable half). An explicit short 11-set is still a novel
  datum (an upper bound where the sequence currently has none).

### #3 — Modular / cyclic Golomb rulers (OEIS A004135 & A004136) — **stale open next term**
- **Object (A004136):** least k such that Z_k contains an n-subset with all pairwise
  sums (incl. repeats) distinct mod k (a *perfect/modular Golomb ruler*). A004135 is the
  distinct-pairs variant.
- **Current best-known:**
  A004135 known through **n=17 (a(17)=255)**; a(18) open — https://oeis.org/A004135
  A004136 known through **n=18 (a(18)=307)**; a(19) open — https://oeis.org/A004136
  Both frontiers frozen since Fausto A. C. Cariboni's 2017-2018 submissions; **not marked
  `hard`.** Witness = a set in Z_k; the modular-Sidon check is cheap and deterministic.
- **Caveat:** like #2, exact = witness (upper bound) + exhaustive over all smaller k
  (expensive). But the next term is genuinely open and uncontested.

### #4 — Max Sidon (B_2) set in {0,1}^n (OEIS A309370) — **lower-bound record, but HOT**
- **Object:** max Sidon subset of {0,1}^n (all pairwise sums distinct).
- **Current best-known:** exact only n<=6 (`...,15`); n>=7 lower bounds only, e.g.
  `a(7)>=24` (Sievers, Sep 2025); a(16)>=505, a(24)>=7179 (Blair, Jun 2026).
  Source: OEIS A309370, https://oeis.org/A309370
- **Why lower priority:** cheap witness, but this is the *most actively contested*
  sequence in the family — William Blair submitted improved lower bounds as recently as
  June 2026. Very hard to be "first" against a live weekly record-pusher.

### #5 — Cap sets in F_3^7 (lower bound > 236) — **native verifier, wrong finder**
- **Object:** largest cap (no-3-collinear set) in F_3^7. Maximize.
- **Current best-known:** exact known only n<=6 (`2,4,9,20,45,112`); **n=7 open: LB 236,
  UB <= 288** (no 289-cap 7-flats). Sources: exact-to-6 & UB: Cameron/… "The cap set
  problem up to dimension 7," https://arxiv.org/abs/2206.09804 ; cap-set lower-bound
  constructions: Tyrrell, "New Lower Bounds for Cap Sets," https://arxiv.org/abs/2209.10045
- **Why here despite ranking:** Propab has a *native* O(|S|^2) cap validity checker
  (`is_valid_cap`) — the best in-house verification story. BUT its finder maxes out around
  the product construction (cap(4)xcap(3)=20x9=180 for n=7) — **far below 236**. Beating
  the record needs affine/local-search machinery well beyond the current engine and likely
  beyond modest compute. Great demo of the *verifier*; not a plausible first *record*.

### #6 — Max pairwise sums that are powers of 2 (OEIS A352178) — open but theory-closing
- **Object:** over n distinct integers, maximize #{i<j : t_i+t_j is a power of 2}.
- **Best-known:** lower bounds tabulated to n=100 (Pratt); several terms proven. Marked
  `more`. Source: OEIS A352178, https://oeis.org/A352178
- **Why lower:** recent heavy theory (Alekseyev, J. Comp. Sys. Sci. 2026,
  https://arxiv.org/abs/2303.02872) is actively pinning values — a moving, expert-occupied
  target. Cheap verify, but low first-mover odds.

### #7 — Greedy B_h analogues (A005282 Mian-Chowla, A051912 greedy B_3)
- **Excluded from serious contention:** these are *greedy-defined* sequences, not
  extremal records; recomputing them is rediscovery, not a record. The engine already
  flags single-point greedy computations as `trivial_rediscovery`
  (`verifier.py`). Listed only to mark them as explicit non-targets.

---

## Part 2 — TOP PICK in full: A396704, improve the n=7 lower bound to 17

### (a) Exact problem statement
Let addition on the binary cube {0,1}^n be **componentwise ordinary integer addition**
(so a threefold sum lands in {0,1,2,3}^n). A subset S ⊆ {0,1}^n is a **B_3 set** iff all
sums a+b+c (a,b,c ∈ S, not necessarily distinct) are distinct as integer vectors —
equivalently, a+b+c = d+e+f with a..f ∈ S forces the multisets {a,b,c} = {d,e,f}.
Define **a(n) = max |S|** over all B_3 sets S ⊆ {0,1}^n.
**Goal:** exhibit a B_3 set of size **17** in {0,1}^7 (the published record is 16).

### (b) Current best-known value + the exact source proving it's the record/open
- Exact/proven-optimal terms: a(0..6) = 1,2,3,4,6,8,11 (exhaustive IP over the
  hyperoctahedral symmetry, each re-verified).
- **n=7: a(7) >= 16, "from computer search by iterated local search; NOT proven
  optimal."** n=8: a(8) >= 19 (same status).
- Upper bound context: every B_3 set is a B_2 (Sidon) set, so a(7) <= A309370(7), and
  A309370(7) itself is only known as `>= 24` — so a(7) lives in roughly [16, 24].
- **Source (authoritative, and proves openness):** OEIS **A396704**, comment block and
  keywords `hard,more,new`, created 2026-06-28. https://oeis.org/A396704
  Companion data/code: https://github.com/willblair0708/verified-combinatorics/tree/main/b3-binary

### (c) Computable parameter regime (what fits a Python sandbox in minutes–hours)
- **n=7:** ambient space is 2^7 = **128 points**. Searching for a size-17 B_3 subset is a
  constraint-satisfaction problem over 128 binary-inclusion variables with distinct-sum
  constraints. Fully in scope for ILS / a SAT or ILP solver / branch-and-bound in
  minutes–hours. (n=8 → 256 points, still feasible; n>=9 grows fast.)
- **Verification of any candidate** (any n up to ~12) is trivially cheap — see (e).

### (d) Witness format (what explicit object certifies a record)
A JSON list of 17 distinct 7-bit vectors, e.g.
```json
{"n": 7, "claimed_size": 17,
 "set": [[0,0,0,0,0,0,0],[1,0,0,0,0,0,0], ... 15 more ...]}
```
Certifies **a(7) >= 17** the instant it passes the B_3 check — beating the published 16.

### (e) Verifier sketch (deterministic; certifies the witness beats the record)
```
def is_B3(S):                       # S: list of tuples in {0,1}^n
    if len({tuple(v) for v in S}) != len(S):   # distinctness
        return False
    sums = {}
    for i in range(len(S)):
        for j in range(i, len(S)):
            for k in range(j, len(S)):         # unordered multisets {i,j,k}
                key = tuple(S[i][t] + S[j][t] + S[k][t] for t in range(len(S[0])))
                if key in sums:                # collision => not B_3
                    return False
                sums[key] = (i, j, k)
    return True

def certifies_record(witness, published_best=16):
    S = witness["set"]
    return (all(x in (0,1) for v in S for x in v)      # in {0,1}^n
            and len(S) > published_best                 # strictly beats record
            and is_B3(S))                                # and is genuinely B_3
```
Cost for |S|=17: C(17+2,3)+lower-order ≈ a few thousand vector sums → sub-millisecond.
**Note the asymmetry the task wants:** *finding* size 17 may be hard; *checking* it is
trivial and fully deterministic — no false confirms possible. This mirrors the engine's
existing `is_valid_cap` design exactly.

### (f) Realistic search strategy — smart search required (naive won't do)
- **Naive greedy / random-restart** (what the engine ships): will likely reproduce ~16,
  not beat it — same regime the competitor already exhausted with ILS.
- **Needed:** iterated local search with B_3-preserving moves (swap-in/swap-out with an
  incremental collision index), OR encode "∃ B_3 set of size 17 in {0,1}^7" as SAT/ILP
  and let a solver settle it. Exploit the **hyperoctahedral symmetry** of {0,1}^7 (bit
  permutations + coordinate complementations) to fix a canonical prefix and shrink the
  space. This is a *modest* engineering lift on top of the existing verifier, and is the
  honest prerequisite: the record is reachable **only** with a smarter finder than the
  one currently wired in.

---

## Part 3 — Blunt HONESTY: is this actually reachable?

**Probability Propab sets a first (even provisional) record here in modest compute:
~25-40%.** Not a slam dunk. Reasoning:

**Strongest evidence FOR:**
1. The n=7 bound is explicitly **"not proven optimal"** and came from local search, not
   exhaustion — provisional bounds are the ones that fall to a fresh, better search.
   (Source: A396704 comments, https://oeis.org/A396704)
2. The gap to the ceiling is real: a(7) ∈ [16, ~24] (B_2 bound A309370(7) >= 24), so 17
   is not squeezed against a proven wall. https://oeis.org/A309370
3. Verification is trivially deterministic and cheap — the engine's core competency —
   so any hit is instantly, honestly certifiable (no risk of a false headline).
4. The search space (128 points) is genuinely small; this is *not* an asymptotic
   theorem-level barrier, which the task explicitly told us to avoid.

**Strongest evidence AGAINST:**
1. **Active competition, same niche.** A396704 was created 2026-06-28 and A309370 lower
   bounds were pushed by William Blair through June 2026. The person who set 16 is
   *actively iterating*; he may reach 17+ (or prove 16 optimal) first, or concurrently.
   A "first" here is provisional and could be short-lived.
   (https://github.com/willblair0708/verified-combinatorics)
2. **The engine's shipped finder cannot do this as-is.** Beating 16 requires bolting on
   ILS/SAT and symmetry reduction — the current greedy/product path will not. So the
   "record" is contingent on new search code, not just running a campaign.
3. **It might already be optimal.** If a(7)=16 truly (plausible — B_3 is far sparser than
   B_2), then no witness of size 17 exists and the effort yields only a *proof of
   optimality*, which needs the expensive exhaustive/ILP branch, not a cheap witness.

**Why still the top pick despite the caveats:** it is the single cleanest instance of the
profile the task demands — *a specific, small, genuinely-open cell where a witness beats a
provisional record and is machine-verified in milliseconds*. If the competition risk is
judged too high, the **uncontested fallback is #2 (A385931, a(11))**: no rival is chasing
it, but the payoff is an exact next term whose optimality proof is the harder,
not-cheaply-verifiable half — a slower, surer, less headline-clean path.

---

## Appendix — sources (all URLs)
- Max B_3 set in {0,1}^n: OEIS A396704 — https://oeis.org/A396704
- Competitor code/data: https://github.com/willblair0708/verified-combinatorics
- Weak B_3 Golomb ruler: OEIS A385931 — https://oeis.org/A385931
- Modular Golomb rulers: OEIS A004135 — https://oeis.org/A004135 ; A004136 — https://oeis.org/A004136
- Max Sidon set in {0,1}^n: OEIS A309370 — https://oeis.org/A309370
- Optimal Golomb rulers (B_2): OEIS A003022 — https://oeis.org/A003022 ; A227590 — https://oeis.org/A227590 ; A143824 — https://oeis.org/A143824
- Cap sets to dim 7 (exact/UB): https://arxiv.org/abs/2206.09804 ; cap-set lower bounds: https://arxiv.org/abs/2209.10045
- Powers-of-2 pair sums: OEIS A352178 — https://oeis.org/A352178 ; Alekseyev — https://arxiv.org/abs/2303.02872
- Mian-Chowla (non-target, greedy): OEIS A005282 — https://oeis.org/A005282
- Engine verifier referenced: `packages/propab-core/propab/domain_modules/math_combinatorics/verifier.py` (`is_valid_cap`, `is_sidon_set`, `is_B3`-style checks)

---

## CP-SAT attempt result (2026-07-08) — honest

An exact OR-tools CP-SAT B_3 finder was built and validated against the proven values:
- **a(4)=6** and **a(5)=8** solved as **proven-optimal** in seconds; size+1 proven **UNSAT**
  (sound infeasibility proofs) — the encoding is correct and complete.
- **a(6)=11**: the correct value is *found*, but CP-SAT does **not close the optimality gap
  within 5 minutes** (proven_optimal=False), and "no size-12" does not resolve in that budget.
- **a(7), size-17 decision**: **timeout (unknown) after 10 minutes** — neither a 17-witness
  found nor 16 proven optimal. No claim.

**Finding:** exact CP-SAT is trustworthy but does not scale past n=5 for full closure at modest
budgets; n=7/size-17 is well beyond reach here. Reaching a(7) would need substantially longer
runs (hours/days, uncertain), a much tighter encoding (stronger symmetry breaking / better
constraints), or a pivot to a more search-friendly reachable target (e.g. the uncontested
modular Golomb rulers A004135/A004136, whose open next terms may be more CP-SAT-tractable).
The finder (`discovery/cp_sat_finder.py`) is retained as a verified exact backend; the ILS
finder remains for larger instances. Nothing here is a record.
