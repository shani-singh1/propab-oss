---
name: distance-bounds-and-open-regime
description: Use Singleton, Hamming/sphere-packing, Gilbert–Varshamov, and Plotkin bounds to judge whether a target [n,k,d] is settled or genuinely open before proposing a code
phase: hypothesis
scope: coding_theory
priority: 31
---
Before proposing a binary linear [n, k, d] code, locate the target between the classical
bounds. The bounds tell you the ceiling (no code can beat it) and the floor (a code is
guaranteed to exist), and the OPEN regime is the band between the best-known constructive
distance and the upper bounds — that band is the only place a discovery can live.

**Upper bounds (ceilings — d cannot exceed these):**
- **Singleton:** d ≤ n − k + 1. Binary codes meeting it (MDS) are essentially only the
  trivial ones (repetition, parity, whole space), so for binary a claim near Singleton at
  intermediate k is almost certainly false — a fast falsifier.
- **Hamming / sphere-packing:** for a t-error-correcting code (d = 2t + 1),
  2^k · Σ_{i=0}^{t} C(n, i) ≤ 2^n. A code meeting it with equality is *perfect*
  (Hamming codes, Golay) — those are tabulated, so meeting the sphere-packing bound is a
  rediscovery, not a discovery.
- **Plotkin:** when d is large relative to n (roughly 2d > n), d ≤ (n · 2^(k−1)) /
  (2^k − 1); tight for simplex-type codes. Use it to reject over-optimistic
  large-distance claims at low n.

**Lower bound (floor — a code at least this good exists):**
- **Gilbert–Varshamov:** a binary linear [n, k, d] code exists whenever
  Σ_{i=0}^{d−2} C(n − 1, i) < 2^(n − k). GV is the benchmark random/greedy constructions
  meet on average. **A construction that merely matches the GV floor is not exciting** —
  the interesting claims either (a) BEAT GV at parameters where the best-known table has
  a gap, or (b) constructively achieve a distance a random code would rarely hit.

**How to judge "open" and form the hypothesis:**
1. Compute the Singleton/Hamming/Plotkin ceiling and the GV floor for the target [n, k].
2. Look up the best-known lower bound d* for [n, k] (Brouwer/Grassl range the engine
   carries for small n, k). If your target d ≤ d*, it is SETTLED — proposing it is a
   rediscovery. If d* < d ≤ ceiling, the [n, k, d] is OPEN and worth a construction.
3. Phrase the hypothesis as "[n, k] admits a code with computed d ≥ D" where D strictly
   exceeds the best-known d* for that [n, k] but respects every ceiling above. State the
   ceiling you are NOT allowed to cross as the automatic falsifier.

**Anti-rediscovery guardrails:**
- The best-known distances for small [n, k] are TABULATED (codetables.de / Brouwer).
  A computed d that only meets-or-falls-below the table value is a trivial rediscovery,
  even if a claim "d ≥ (table value)" is technically satisfied — the engine demotes it.
- Never assert a bound to "verify" it. Bounds SCOPE the hypothesis (they say which [n,k,d]
  is even possible); the discovery must still be a real code the engine builds and whose
  distance it computes with a witness.
- Target the band d* < d ≤ ceiling at parameters where the table has room. A witnessed
  code whose computed distance strictly beats the best-known lower bound there is a real
  result; anything at or below the table is not.
