---
name: cap-set-and-extremal-constructions
description: Technique menu for improving extremal set constructions (cap sets, Sidon sets, AP-free sets)
phase: hypothesis
scope: math_combinatorics
priority: 30
---
When proposing hypotheses about extremal configurations in F_q^n (cap sets, Sidon
sets, progression-free sets), reach for genuine construction/analysis techniques —
not another parameter of the greedy baseline:

- **Product / tensor constructions.** Combine optimal small-dimension objects (a cap
  in F_3^a × a cap in F_3^b) and ask whether a structured perturbation or local repair
  of the product beats the direct product size. This is a concrete, computable path to
  improving a lower bound.
- **Symmetry-restricted search.** Restrict the search to subsets invariant under a
  non-trivial subgroup (affine group AGL(n,q), cyclic shifts): a hypothesis that an
  invariant family beats the lexicographic/greedy baseline is novel and testable by
  direct construction.
- **The polynomial / slice-rank method.** Frame upper-bound questions in terms of the
  slice rank or the polynomial method (Croot–Lev–Pach / Ellenberg–Gijswijt). Propose
  whether a specific polynomial certificate tightens a known bound.
- **Fourier-analytic / pseudorandomness criteria.** Order candidate points to control
  the largest Fourier coefficient of the indicator, delaying density collapse; hypothesize
  a measurable improvement over cardinality-only greedy.
- **Growth-exponent framing.** Phrase claims about the growth exponent (a_q(n)^{1/n}) and
  where it sits between the best known lower and upper bounds — the real open frontier.

Anti-rediscovery guardrails specific to this domain:
- The maximum cap sizes for small n are TABULATED and PROVEN up to n=6 (F_3). A claim
  that "the size in F_3^k equals/at-least the known value" is a REDISCOVERY unless it
  is produced by a genuinely new construction that COMPUTES the object with a witness.
- Never propose to "verify" a tabulated bound by asserting it; propose a construction
  that computes a real cap/set and compare its measured size honestly to best-known
  (below / matches / exceeds), with the object as a checkable witness.
- Target the OPEN regime (n ≥ 7 for F_3 caps; the growth-exponent gap) where a genuine
  improvement would be a real result.
