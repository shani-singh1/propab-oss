---
name: symmetry-reduction-for-search-and-proof
description: Prune a search by orbits / canonical forms to make it tractable — but treat soundness as sacred, because an unsound symmetry break turns an UNSAT/optimality proof into a false theorem
phase: experiment
scope: mathematics
priority: 24
---
Most combinatorial search spaces are riddled with symmetry — relabelings, permutations,
rotations, reflections — so the honest object is an orbit, not an individual. Collapsing each
orbit to one canonical representative can shrink a search by orders of magnitude and is often the
only way a proof becomes feasible. But symmetry reduction is the single most dangerous step in a
search, because getting it wrong does not slow you down — it silently gives you the wrong answer.

1. **Know the asymmetric stakes of an unsound break.** For a FIND, an over-aggressive reduction
   only costs you witnesses: you might prune the orbit of a valid object and wrongly report "none
   found", but any witness the pruned search *does* return is still real. For a PROVE, the failure
   is catastrophic and silent: if your reduction removes a satisfying assignment that lies in a
   pruned orbit, the solver returns UNSAT / "optimal" over a space that was missing a real
   solution — and you announce a THEOREM that is false. A false "no better exists" is far worse
   than a missed witness.

2. **Use a canonical form that is a true invariant of the orbit.** The representative you keep
   must be a well-defined function of the orbit (e.g. the lexicographically least labeling under
   the full symmetry group), so that every object maps to exactly one kept representative and no
   object maps to none. An ad-hoc "looks equivalent, skip it" pruning rule that is not a genuine
   orbit invariant is exactly how satisfying assignments vanish.

3. **VALIDATE the reduction against every known value before trusting a proof on it.** Run the
   reduced search on every case whose answer is already known and confirm it reproduces each one —
   the same counts, the same optima, the same known extremal objects. A symmetry reduction that
   misses even one established value is unsound, and every "proof" built on top of it is void. Do
   this before, not after, you rely on it for a new claim.

4. **Prefer symmetry breaking the certifier can re-check.** Encode the break as explicit
   constraints (lex-leader / ordering constraints added to the model) rather than as hidden
   pruning inside search code, so the UNSAT/optimality certificate is over a model an independent
   checker can inspect and confirm is symmetry-sound. Reductions that live only in imperative
   search logic cannot be audited.

5. **When in doubt for a PROVE, don't break.** If you cannot show the reduction is sound, run the
   proof without it (or over a smaller but fully covered space) rather than claim optimality over a
   space you may have holed. A slower sound proof beats a fast false one.

The bar: a symmetry reduction is admissible in a proof only after it has reproduced every known
value and is expressible as a checkable invariant or constraint. The dishonest outcome is an
optimality or UNSAT claim resting on a break you never validated — it reads as a theorem but is a
bug, and it is the worst failure this whole system can produce: a confident, false "proved".
