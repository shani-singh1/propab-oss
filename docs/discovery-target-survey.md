# Discovery-target survey — can general tools set a real, certifiable record?

**Date:** 2026-07-09 · **Type:** read-only assessment (no code built, no campaign launched)
**Question:** across the whole `record_registry.py`, which improvable terms could a
*general* system plausibly turn into a genuine, independently-certifiable NEW result —
using only reusable capabilities, not a per-question solver?

---

## Capabilities actually in hand (from the discovery package)

| Capability | File | What it does | Empirically |
| --- | --- | --- | --- |
| (a) Metaheuristic finder | `finder.py` | greedy + Dynamic Local Search, incremental collision index | **plateaus at the KNOWN bound** — 6 min could not push B_3 a(7) 16→17 |
| (b) Exact CP-SAT decide/optimize | `cp_sat_finder.py`, `modular_golomb.py` | SAT witness / UNSAT proof / maximize, with sound symmetry breaking | **proof direction weak** — couldn't close a(6)=11 optimality (5 min); a(7) size-17 *feasibility* also timed out **unknown** (10 min) |
| (c) Independent certifier | `verifier.py`, `modular_golomb.py` | `is_B3`, `certify_b3_record`, `is_modular_sidon`, `certify_modular_ruler` — cheap, deterministic, re-derives from scratch | solid; the real differentiated asset |
| (d) Construction synthesis | `construction_synthesis.py` | model WRITES `construct(n)`, sandboxed, gated by the exact oracle | general but **unproven** to cross any bound |

**Consequence (the honest asymmetry):** *FIND* is self-certifying and safe, but only
succeeds if the object genuinely exists **and** is reachable by search. *PROVE*
(optimality / minimality) is out of reach beyond tiny sizes. Clean self-certifying
**records exist only for the maximize + `provisional_lower_bound` terms**; every
minimize/`open` term yields at best an *upper-bound witness*, not the proven sequence value.

---

## Every improvable term in the registry (status `open` or `provisional_lower_bound`)

| Seq | n / param | best_known | dir | status | FIND tractable? | PROVE tractable? | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A396704 | **a(7)** | 16 | max | prov. LB | low — **both engines already stuck here** | no (worse than a(6)) | clean record but most-tried/most-stuck; 16 may be optimal |
| A396704 | **a(8)** | 19 | max | prov. LB | low-moderate — freshest, least-hammered | no | **cleanest fresh maximize record** (odds still low) |
| A385931 | a(11) | — | min | open | witness = upper bound only; no wired finder | no (needs "no shorter ruler") | soft; upper-bound datum, not a term |
| A004135 | **a(18)** | — | min | open | **moderate** — smallest, most CP-SAT-friendly model; `attempt_open_term` wired | no (needs k-scan) | tractable but soft (upper bound, not proven term) |
| A004136 | a(19) | — | min | open | value tied to **existence of a projective plane of order 18** (famous open Q) | no | wrong tool; exclude |
| A309370 | **a(7)** | 24 | max | prov. LB | moderate — weakest constraint = densest, most search-amenable object | no | clean record, but "most contested" + **no B_2 finder wired** |
| A309370 | a(16) | 505 | max | prov. LB | ~none — construction-limited object in a 65k-pt space | no | local search can't BUILD it; exclude |
| A309370 | a(24) | 7179 | max | prov. LB | ~none — construction-limited, 16.7M-pt space | no | local search can't BUILD it; exclude |
| A090245 | a(7) | 236 | max | prov. LB | ~none — in-house finder ~180, far below 236 | no | decades-studied polynomial-method frontier; exclude |

Key distinction the space forces: **local search finds SMALL extremal objects; it does
not BUILD LARGE structured ones.** That kills a(16)/a(24) (huge, algebraically-constructed
bounds) regardless of compute, and it is why the tractable cells are all small (n=7,8; k≈280).

---

## Ranked shortlist (best-first — all honestly low-probability)

**1. A396704 `a(8)` — find a B_3 set of size ≥ 20 in {0,1}^8.**
Cleanest *fresh* self-certifying maximize record (no proof needed). Author effort and both
in-house engines concentrated on a(7); a(8)=19 is a lower-effort ILS bound with more
construction slack, and the cube is still only 256 points. Odds low (both engines weaker at
n=8), so this is a longer-compute + construction-synthesis lottery — but it is the least
picked-over clean target. **Attack: n=8, target 20**, long parallel DLS restarts + `construct(n)`
seeded with GF(2) generator-matrix / Sidon-difference structure.

**2. A004135 `a(18)` — certified modular-Sidon witness (upper bound a(18) ≤ k), smallest k reachable.**
Best *structural* fit for "strong SAT-find + cheap certifier": the model is a flat
"distinct pairwise sums mod k" over ~k booleans (far smaller than the B_3 cube), uncontested,
frozen since Cariboni 2017-2018, and `attempt_open_term` is already wired. **Honest caveat:** a
witness is an *upper bound*, not the proven a(18); the frontier was computed *exactly*, so a
non-minimal witness is not a genuine new term. Highest chance of a certified OBJECT; lowest
chance that object is a headline result. **Attack: n=18, scan k upward from 256.**

**3. A309370 `a(7)` — find a Sidon (B_2) set of size ≥ 25 in {0,1}^7.**
Weakest constraint ⇒ densest, most search-amenable object type, in the smallest space (128
points); a clean self-certifying record if found. Two real drags: it is the *most actively
contested* sequence (thin slack), and **no B_2 finder/CP-SAT model exists in the package** —
only B_3 — so the general B_2 tool must be stood up first (an easy analogue of the B_3 code,
but it is real work). **Attack: n=7, target 25.**

**4. A396704 `a(7)` — size 17, via a much longer CP-SAT feasibility run or the construction loop.**
The flagship, kept only for completeness: it has already resisted the metaheuristic (6 min)
**and** CP-SAT feasibility (10 min, unknown). A fresh attempt is not a new lever — it is a
compute gamble (hours/days) or a bet on construction-synthesis, and there is a real chance
16 is simply optimal (so no witness exists and only the out-of-reach UNSAT proof settles it).
**Do not lead a campaign here.**

**5. (Outside the registry) a best-known lower bound in a coding-theory table.**
The class best matched to our profile: **constant-weight binary codes A(n,d,w)** (Brouwer's
tables) or **covering codes K(n,R)** — a bigger code / smaller cover self-certifies with a
cheap deterministic pairwise-distance / coverage check and needs **no optimality proof**, so
it is a clean lower-bound record. Honest caveat: the easy entries are long swept; remaining
gains need clever constructions or specialist clique/tabu compute that general local search
rarely beats. Requires live-table target selection (not derivable from this registry) and is
a lottery — but it is the right *shape* of problem.

**Explicitly excluded as unreachable with current general tools:** A090245 a(7) (cap set;
finder ~180 vs 236, decades-studied), A309370 a(16)/a(24) (construction-limited large objects
local search cannot build), A004136 a(19) (entangled with the projective-plane-of-order-18
open problem), A385931 a(11) (upper-bound-only, no wired finder, needs the minimality proof).

---

## Bottom line

**No registry target is a high-probability, cleanly-certifiable NEW *sequence-term* record
with the current general tools at modest compute.** The flagship a(7) 16→17 is the
most-tried, most-stuck cell (both engines timed out) and may already be optimal. What remains
splits into (A) longer-compute lotteries on fresher maximize bounds — best shot **A396704
a(8)→20** — and (B) certified *upper-bound witnesses* on uncontested minimize/open terms —
best shot **A004135 a(18)** — which are real, checkable objects but not proven sequence
values. If a bounded experiment is run at all, run one of those two; do **not** launch an
optimality-PROVE campaign, and do **not** launch a large-object local-search campaign
(a(16)/a(24)/cap-set) — those are structurally out of reach for local search.

**What general capability would move the needle:** a **construction-aware / construction-
synthesizing search** — inject algebraic/recursive structure (Singer/difference-set/product
seeds, or model-written `construct(n)` gated by the exact oracle) instead of random starts.
The wall is precisely that pure local search plateaus at known bounds and pure CP-SAT can't
scale the proof; only injected *structure* changes the reachable frontier, and it is the one
lever (capability (d)) that has not been genuinely exercised. Secondarily, a **SAT-tuned
witness-finder** (streamlining + strong symmetry breaking + massive parallel restarts) aimed
at the FIND (satisfiability) direction, where SAT is often far easier than the UNSAT proofs
that have blocked us — applied to the small cells (a(8)→20, B_2 a(7)→25).
