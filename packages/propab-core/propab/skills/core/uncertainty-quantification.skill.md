---
name: uncertainty-quantification
description: Attach honest uncertainty to every estimate — intervals, bootstrap, error propagation — and never report a point estimate as if it were exact
phase: evidence
scope: core
priority: 28
---
A number without its uncertainty is not a result — it is a guess dressed as a fact. Every estimate
you report is a draw from a distribution of estimates you could have gotten; your job is to say how
wide that distribution is. A result whose uncertainty overlaps the null is INCONCLUSIVE, no matter
how appealing the central value.

1. **Never report a point estimate as if it were exact.** Attach an interval to every quantity that
   was measured or estimated: an error bar, a confidence interval, or a credible interval. State
   what the interval means (its coverage level, e.g. 95%) and what it is over (sampling variability,
   measurement noise, both). A lone number silently claims infinite precision — which is always false.

2. **Distinguish the two kinds of uncertainty, because they behave differently.**
   - *Aleatoric* (irreducible): genuine randomness in the system or measurement — more data narrows
     your estimate of it but never removes it. It sets the noise floor.
   - *Epistemic* (reducible): uncertainty from limited data, model choice, or unknown parameters —
     it shrinks as you learn more, and it is where most overconfidence hides.
   Say which dominates. Reporting only the aleatoric part (e.g. measurement error) while ignoring the
   epistemic part (e.g. "which model / which sample") understates your true uncertainty, often badly.

3. **Quantify uncertainty by a method matched to the quantity.**
   - When a closed-form standard error exists and its assumptions hold, use it.
   - When the estimator is complex, nonlinear, or the distribution is unknown, BOOTSTRAP: resample
     the data (respecting its structure — resample the independent unit, e.g. the cluster/subject,
     not individual rows) and read the interval off the spread of the recomputed estimates.
   - When a result is a FUNCTION of several uncertain inputs, PROPAGATE their errors through the
     computation (analytically via the standard rules, or by Monte Carlo — sample each input from
     its distribution and push the samples through). Do not report the output as exact because the
     formula was exact; the inputs were not.

4. **Uncertainty in, uncertainty out.** If an estimate feeds a downstream calculation, the downstream
   result inherits and usually AMPLIFIES that uncertainty — carry it through, never drop it at a step
   boundary. A pipeline that discards error bars at each stage ends in false precision.

5. **Compare against the null WITH the uncertainty, not just the central value.** The decision is not
   "is the point estimate on the interesting side?" but "does the interval EXCLUDE the null / the
   trivial value?" If the interval straddles the null, the honest verdict is inconclusive — say so,
   and report the width so a reader sees whether the study was too imprecise to decide, versus
   genuinely centered on no effect. Widening the interval to be safe is honest; hiding it is not.

6. **Report the uncertainty even when it hurts the story.** State the interval alongside the estimate
   in every headline claim, including the ones that came out the way you hoped. Round the point
   estimate to the precision the uncertainty actually supports — extra decimal places past the error
   bar are noise pretending to be signal.

The bar: a skeptic reading your result should be able to see not just what you estimated, but how
much it could plausibly have been otherwise — and should reach "inconclusive" exactly when your
interval fails to clear the null.
