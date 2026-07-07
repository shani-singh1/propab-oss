---
name: sidon-sets-and-additive-energy
description: Frame Sidon / B_h set and additive-energy hypotheses that the verifier can compute (greedy vs Bose–Chowla, F(n)/√n, sumset growth)
phase: hypothesis
scope: math_combinatorics
priority: 33
---
A Sidon set (B_2 set) has all pairwise sums distinct; a B_h set has all h-fold sums
distinct. The extremal question — how large can a Sidon subset of {1,…,n} be — is
governed by the ratio F(n)/√n, and the asymptotics of that ratio are a live open
problem. Additive energy E(A) = #{(a,b,c,d) : a+b = c+d} is the dual lens: a Sidon set
is exactly a set of *minimum* additive energy. Build hypotheses the engine can compute.

**Computable techniques (each maps to real verifier work):**

- **Greedy vs algebraic (Bose–Chowla) at matched scale.** The greedy construction and
  the Bose–Chowla set (size q for prime q, living in {0,…,q²−1}) both realize Sidon
  sets. A sharp hypothesis compares their F(n)/√n at the *same* n and predicts which is
  denser — and by how much. This routes to a real matched-n comparison; state the
  direction and margin so a confirmation is falsifiable.

- **Ratio asymptotics as a sweep, never a single point.** A single greedy size at one n
  is a known-range computation, not evidence about the open limit. Phrase the claim as a
  trend across a multi-n sweep (n ≥ 500, several values): does F(n)/√n stay in a band
  [lo, hi], decrease monotonically, or cross below a target? Each is a concrete,
  checkable claim the verifier evaluates against the computed ratios.

- **Threshold crossing as a discovery.** "The smallest n where F(n)/√n first drops
  below t" is a genuine, falsifiable quantity — the verifier searches for the crossing.
  Pick a t that is uncertain (not one the greedy trend obviously satisfies).

- **Additive energy / sumset growth as the structural signal.** |A+A|/|A| is a proxy
  for additive energy: Sidon-like sets maximize sumset growth (near |A|(|A|+1)/2),
  arithmetic progressions minimize it (≈ 2|A|). A hypothesis that a Sidon-like set's
  sumset growth strictly exceeds both a random subset's and an AP's at fixed n is
  computable and decides a real structural claim. Frame B_h generalizations as sumset
  claims on the h-fold sumset when h > 2.

**Anti-rediscovery guardrails:**
- The Erdős–Turán result F(n) ~ √n is established; restating "greedy Sidon sets have
  size about √n" is a REDISCOVERY. The open part is the second-order term / the exact
  constant and its convergence — target that, not the leading order.
- A construction that merely reproduces a known Sidon set (Bose–Chowla, Singer,
  Mian–Chowla) at tabulated size is a rediscovery. Discovery-worthy means a measured
  ratio, trend, or threshold that is NOT a restatement of a settled asymptotic — and it
  must come with the actual set as a checkable witness, not an asserted size.
- Never "confirm" a Sidon claim by asserting the set is Sidon; the engine verifies the
  Sidon property by checking all pairwise sums are distinct on the real set.
