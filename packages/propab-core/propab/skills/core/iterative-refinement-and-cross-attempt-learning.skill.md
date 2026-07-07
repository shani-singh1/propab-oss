---
name: iterative-refinement-and-cross-attempt-learning
description: Carry the abstracted lesson from each attempt forward, and keep a change only if it beats an evaluator the search never optimized against
phase: iteration
scope: core
priority: 50
---
Iteration is not re-sweeping parameters until something scores well — that just overfits the
feedback signal. Iteration is turning each refuted or confirmed attempt into a reusable lesson
that constrains the next attempt, under a discipline that prevents the search from fooling
itself.

1. **Abstract the lesson; do not just record the number.** After every attempt, write down not
   only what happened but WHY: the mechanism the result supports or bounds. "Configuration X
   scored 0.4" is a log entry. "Approaches that rely on <assumption> collapse once the confound
   is broken" is a lesson. Carry the lesson upward so it shapes the whole line of inquiry, not
   just the next tweak — a refuted attempt should retire an entire family of ideas, not send you
   to the neighboring parameter value.

2. **Turn failures into negative constraints.** A refutation is informative: it tells you a
   region of the space is dead. Record it as an explicit constraint ("do not pursue
   <direction>, because <mechanism> makes it fail") so the search never wastes effort
   re-entering ground already proven barren. Accumulated negative constraints are often worth
   more than the positive wins.

3. **Separate a search/dev evaluator from a held-out gate.** Use a fast, repeatable evaluator
   FREELY during the search to steer exploration — but this is exactly the signal iteration
   tends to overfit. A change is only KEPT if it also beats a separate held-out evaluator the
   search did NOT get to optimize against (different seeds, different splits, an out-of-
   distribution condition). A candidate that scores high on the search evaluator but fails the
   held-out gate is evidence you are exploiting the feedback, not improving the science —
   discard it and record why.

4. **Condition the next hypothesis on accumulated evidence, not a fresh guess.** Propose the
   next attempt from the current state of confirmed lessons and negative constraints, so each
   step builds on the last. Favor the attempt that is most INFORMATIVE (most likely to
   discriminate between live hypotheses), not merely the one predicted to score highest — a
   test that can only confirm what you already believe teaches nothing.

5. **Promote only what clears the gate; keep everything else as a lesson.** The best result is
   updated only when a candidate passes the held-out gate. Everything that does not pass is not
   waste — it is a recorded constraint that makes the search cumulative rather than a random
   walk. Never quietly promote a search-evaluator win to "the result"; that is how overfitting
   masquerades as progress.

The test of good iteration: after N attempts you can state, in one sentence each, the general
lessons learned and the directions ruled out — and every kept improvement survived an evaluator
your search never touched.
