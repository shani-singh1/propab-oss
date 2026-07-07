---
name: polynomial-slice-rank-method
description: Use the polynomial / slice-rank method (Croot–Lev–Pach, Ellenberg–Gijswijt) to frame — and honestly test — extremal upper bounds in F_q^n
phase: hypothesis
scope: math_combinatorics
priority: 32
---
The polynomial method and its slice-rank refinement (Croot–Lev–Pach 2016;
Ellenberg–Gijswijt 2016) are how the modern *upper* bounds on progression-free and
cap-set problems were proven. When you reach for this method, be precise about which
side of the bound you are working on and what the engine can actually certify.

**What the method proves (upper bounds).** For a progression-free set A ⊆ F_q^n, one
builds a polynomial that vanishes on the difference structure, then bounds |A| by the
*slice rank* of a related tensor. The output is a statement of the form
|A| ≤ c·(λ_q)^n with an explicit λ_q < q (for F_3, λ_3 ≈ 2.7551, so caps have size
≤ 2.756^n up to constants). This is a CEILING on any construction, proven by a rank
argument, not a search.

**How to form a hypothesis with it — legitimately:**

- **Sharpen the constant, not the exponent, on a computed slice.** The exponent λ_q
  is fixed by the CLP/EG optimization; the open room is in the multiplicative constant
  and in whether a *specific structured family* meets or approaches the ceiling. Propose
  that a named construction's measured size sits at a specific fraction of the 2.756^n
  ceiling across a dimension sweep, and let the verifier compute that ratio_to_clp
  honestly. That ratio is the real, checkable quantity.

- **Use the ceiling as an adversarial null for a lower-bound claim.** A construction
  claim "|A_max(n)| ≥ K" is only interesting if K is genuinely large relative to what
  greedy/product baselines reach AND strictly below the CLP ceiling (a claim at or
  above 2.756^n would be refuted by the method itself — flag that as an automatic
  falsifier). State the ceiling explicitly so a confirmation cannot be a just-so story.

- **Slice-rank certificate as a witness, when you can compute it.** If you claim a
  polynomial/slice-rank certificate tightens a bound, the certificate (the polynomial's
  support, the multiset of monomials, the rank count) is the witness — supply it so a
  skeptic can recompute the rank. A rank argument you cannot exhibit is not evidence.

**Anti-rediscovery guardrails (specific to this method):**
- Re-deriving λ_q ≈ 2.7551 for F_3, or restating the EG bound as if it were a new
  result, is a REDISCOVERY. The theorem is proven; asserting it is not a discovery.
- The verifier COMPUTES lower-bound constructions (real caps, greedy, products) and
  compares their measured size to the 2.756^n ceiling — it does NOT re-prove the
  upper bound. Do not claim the engine "verified the CLP bound"; claim only what it
  measured: a construction's size and its ratio to the known ceiling.
- The genuinely open frontier here is the *gap between the best lower-bound growth
  exponent (≈ 2.2174, Edel-type products) and the CLP upper exponent (≈ 2.7551)*. A
  hypothesis that measurably moves a construction's per-dimension growth rate toward the
  ceiling — with the cap itself as a checkable witness at n in the open regime (n ≥ 7
  for F_3) — is discovery-worthy; matching a tabulated size is not.
