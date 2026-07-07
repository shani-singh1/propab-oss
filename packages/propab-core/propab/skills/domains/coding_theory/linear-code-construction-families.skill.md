---
name: linear-code-construction-families
description: Choose the right binary-linear-code construction family (Hamming, extended Hamming, simplex, Reed–Muller, cyclic/BCH, LDPC) for the [n,k] target, and know which the engine builds natively
phase: hypothesis
scope: coding_theory
priority: 30
---
A hypothesis about a binary linear [n, k, d] code is only testable if it names a
construction the engine can turn into a real generator matrix over GF(2). Match the
family to the [n, k] regime, and know the difference between families the verifier
builds natively and families you must supply as an explicit generator matrix.

**Families the engine constructs natively (name them in the statement):**
- **Hamming [2^r − 1, 2^r − 1 − r, 3].** High rate, minimum distance fixed at 3
  (single-error-correcting). The right move when you want maximal k at d = 3 and n is
  one less than a power of two (n = 7, 15, 31 …).
- **Extended Hamming [2^r, 2^r − 1 − r, 4].** Adds an overall parity bit → d = 4 (SEC-DED).
  The right move when you need to *detect* double errors at almost the Hamming rate.
- **Simplex [2^r − 1, r, 2^(r−1)] (dual of Hamming).** Very low rate but every nonzero
  codeword has the same large weight 2^(r−1) — the move when distance, not rate, is the
  objective, or as an equidistant-code building block.
- **First-order Reed–Muller RM(1, m) = [2^m, m + 1, 2^(m−1)].** Low rate, large distance,
  strong symmetry — the move for very noisy channels and when you want a self-dual-ish,
  highly structured code.
- **Repetition [n, 1, n]** and **single-parity-check [k+1, k, 2]** — the two extreme
  trade-offs; use them as honest baselines, not as discoveries.

**Families the engine does NOT build from a name — supply an explicit generator matrix:**
- **Cyclic / BCH codes.** Defined by a generator polynomial g(x) | x^n − 1; BCH gives a
  designed distance via consecutive roots. If you hypothesize a BCH [n, k] beats a
  baseline, you must construct its k×n generator matrix yourself (rows = shifts of g)
  and pass it as the witness — the engine will compute its TRUE distance, which may be
  larger than the designed distance, so claim the designed lower bound and let the real
  computation confirm or exceed it.
- **General Reed–Muller RM(r, m), r ≥ 2, and LDPC codes.** Also supply as an explicit
  generator (LDPC: a sparse generator/parity matrix). The engine verifies whatever
  generator you give it; it will not invent the family from the word "LDPC".

**When each is the right move (rate–distance intuition):**
- Need high rate, tolerate d = 3? Hamming. Need d = 4 detection? Extended Hamming.
- Need large guaranteed distance at low rate? Simplex or RM(1, m).
- Targeting a specific designed distance at moderate rate and n = 2^m − 1? BCH (explicit).
- Long code, iterative decoding, near-capacity? LDPC (explicit, sparse).

**Anti-rediscovery guardrail:** Naming a textbook family at its textbook parameters and
"confirming" its known distance is a REDISCOVERY — [7,4,3] Hamming, [8,4,4] extended
Hamming, [24,12,8] Golay are tabulated. A family is a hypothesis ingredient, not a
result: propose it only where its computed [n, k, d] would sit in the OPEN regime for
those parameters (see the distance-bounds skill), with the generator as the witness.
