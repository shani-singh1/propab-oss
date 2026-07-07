---
name: growth-exponent-open-bounds
description: Design multi-n / multi-dimension experiments that address the OPEN growth-exponent gap rather than rediscovering a tabulated value at one size
phase: experiment
scope: math_combinatorics
priority: 34
---
Extremal combinatorics questions are almost always about an *exponent*, not a single
value: the cap-set growth rate c_n = |A_max(n)|^(1/n), the Sidon ratio F(n)/√n, the
AP-free density decay. The open frontier is where the best lower-bound exponent and the
best upper-bound exponent still disagree. When you specify how a hypothesis about such a
quantity is tested, design the experiment so a confirmation is about the exponent's
BEHAVIOUR, and so it cannot be a rediscovery of a tabulated point.

1. **Test a trend across a sweep, not a point.** One size at one n/dimension is, by
   construction, a known-range computation and the verifier will flag it as a trivial
   rediscovery. Specify at least three sweep points into the open regime (F_3 caps:
   dimensions ≥ 5, with the frontier at n ≥ 7; Sidon: n ≥ 500 across several values) and
   state the trend the data must show — c_n strictly increasing toward the CLP ceiling,
   F(n)/√n monotone-decreasing, density decaying sub-exponentially. The trend is the
   falsifiable object.

2. **State the exponent gap explicitly as the null-to-beat.** Name the current
   lower-bound exponent and upper-bound exponent (for F_3 caps: ≈ 2.2174 lower via
   product constructions vs ≈ 2.7551 CLP/EG upper). A result is only interesting if it
   measurably moves a construction's per-dimension rate within that gap — write the gap
   into the hypothesis so a skeptic sees exactly what "progress" would mean.

3. **Anti-rediscovery is structural, not cosmetic.** The cap-set sizes for F_3^n are
   TABULATED and proven up to n = 6 (and best-known through n = 8). A sweep whose points
   are all table lookups confirms arithmetic but is a REDISCOVERY — the engine demotes
   best-known-table evidence to trivial_rediscovery even when a claim is "supported". To
   be discovery-worthy the improving point must be produced by a genuine construction
   that COMPUTES the object (a real cap / real Sidon set) whose measured size the verifier
   independently checks — with the object as the witness.

4. **Provide a witness for the exact claims, statistics for the trend claims.** If the
   claim is exact ("a construction achieves size K in F_3^7"), the cap set itself is the
   witness and the engine recomputes it; a size K that merely equals the table value with
   no constructed witness is a table lookup, not a discovery. If the claim is a trend
   ("c_n increases across dimensions 5–8"), the sweep of computed exponents is the
   evidence — report the per-dimension rates, not just a pass/fail.

5. **Name the failure regime before confirming.** Every exponent claim has a size where
   the construction should stop beating the baseline (products saturate; greedy stalls).
   State that regime and check it: a monotonicity claim that was never tested at the size
   where it is expected to break is not a finding.

The bar: the experiment decides where a construction's growth exponent sits inside the
open lower/upper gap, from computed objects the engine re-checks — not by re-reading a
number the literature already tabulated.
