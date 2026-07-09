---
name: conjecturing-from-computation
description: Turn computed small cases into a conjecture — OEIS, a recurrence, a generating function, an integer relation — then validate it on held-out terms it was never fit on; a fitted formula is never an established one
phase: hypothesis
scope: mathematics
priority: 12
---
Computation is where conjectures come from: compute enough small cases exactly, and a pattern
often becomes visible as a sequence, a recurrence, or an exact constant. But a formula that
merely reproduces the terms you handed it is fitted, not proven — the whole risk of this workflow
is mistaking interpolation for discovery. The discipline is to force the guess to predict terms
it has never seen.

1. **Compute the small cases EXACTLY, and more than you think you need.** Use exact enumeration /
   arithmetic (`combinatorial_enumeration`, `number_theory`, `symbolic_algebra`) so the terms
   carry no float error. Compute several more terms than the model you intend to fit has degrees
   of freedom — those extra terms are your validation set, and you must have them before you fit.

2. **Guess a form from part of the data.** Look the sequence up (`sequence_oracle` → OEIS);
   guess a linear recurrence or rational generating function (`find_linear_recurrence`); or, for
   a numeric constant, hunt an integer relation / closed form (`integer_relation`, PSLQ). Fit the
   guess on an INITIAL segment of the terms — deliberately hold the rest back.

3. **VALIDATE on the held-out terms.** Use the fitted recurrence / formula to PREDICT the terms
   you withheld, then recompute those terms independently and compare. A guess that predicts
   held-out terms it was never fit on has earned the label "conjecture, well-supported". A guess
   that only reproduces its own fitting set has earned nothing — a k-term recurrence can always be
   made to fit k terms, so agreement there is vacuous.

4. **For PSLQ / integer relations, re-derive at higher precision.** A relation found at one
   precision is a coincidence until it survives being recomputed at higher precision with a small
   residual; report the precision and the residual, not just the relation.

5. **Report it as a CONJECTURE, always.** Held-out validation raises confidence; it does not make
   the formula a theorem. Label the output "conjectured recurrence / closed form", state how many
   terms it was validated against, and flag that a proof (induction, a bijection, an exact
   generating-function argument) is still owed. An OEIS match is a lead, not a citation of proof.

The bar: never emit a formula as established from computation alone. The dishonest move is
reporting a fitted expression — one that only interpolates the terms it was trained on — as an
identity or a "known" result. A conjecture is honest exactly when you state the terms it predicted
correctly, the terms it was fit on, and the fact that it remains unproven.
