---
name: what-counts-as-a-proof
description: Draw the exact line between computation-as-proof (exhaustive, exact, certified) and computation-as-evidence (sampled, heuristic, bounded-range), and let only the former license a "confirmed"
phase: evidence
scope: mathematics
priority: 32
---
Before you attach a verdict to a mathematical claim, classify what your computation actually
established. Some computations ARE proofs; most are evidence. The failure mode this skill exists to
prevent is emitting "confirmed" / "optimal" / "holds for all n" on the back of a computation that
only ever sampled or heuristically explored the space.

1. **Computation-as-proof — can CONFIRM.** Only these qualify:
   - an *exhaustive enumeration* over a space you can argue is COMPLETE (every case covered, no
     orbit silently pruned — see `symmetry-reduction-for-search-and-proof`);
   - *exact* arithmetic / symbolic computation with no floating-point or truncation error;
   - a *certified witness* re-checked by an independent checker;
   - a *sound UNSAT* from an encoding you have shown faithfully models the claim.

2. **Computation-as-evidence — can only SUPPORT or stay inconclusive.** Sampling, Monte Carlo, a
   heuristic or local-search plateau, or a check over a finite range with no argument that the range
   is exhaustive. These raise or lower plausibility; they never confirm a universal.

3. **Know precisely what each certificate proves.**
   - A *certified witness* proves a **lower bound / existence**: there IS an object at least this
     good. It says nothing about whether a better one exists.
   - An *exhaustive search or sound UNSAT* proves **optimality / a universal**: no better object
     exists, or the property holds for every case — but only because the space was provably complete
     and the encoding sound.
   - A *heuristic plateau* (best-known did not improve after long search) proves **neither** bound.
     It is suggestive of optimality at most, and only if you say so honestly.

4. **Respect the line in the verdict.** "Confirmed / proved / optimal" is reserved for
   computation-as-proof. A witness earns "lower bound established, optimality open". A plateau earns
   "best found, not proven optimal". A bounded-range check earns "verified for n ≤ N, unproven
   beyond". This mirrors `core/evidence-honesty-and-calibration`: the verdict must be entailed by the
   kind of evidence, no stronger.

5. **When a claimed proof depends on an assumption, surface it.** Exhaustiveness, encoding
   soundness, and symmetry-reduction validity are the assumptions that turn evidence into proof — if
   any is unverified, the result is evidence, not proof, and must be reported as such.

The bar: name the kind of computation you ran, and let it dictate the verdict. The dishonest moves
are the two boundary crossings — calling a heuristic plateau "optimal", and calling a sampled or
bounded-range check "proved for all n". A "confirmed" that a skeptic could not reproduce as a
proof is a false confirmation.
