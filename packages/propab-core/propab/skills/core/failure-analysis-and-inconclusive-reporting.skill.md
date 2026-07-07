---
name: failure-analysis-and-inconclusive-reporting
description: Diagnose why a test came out null or ambiguous, and report inconclusive results as first-class evidence rather than forcing a verdict
phase: evidence
scope: core
priority: 40
---
Not every test yields a clean confirm or refute. A large share of results are genuinely
inconclusive, and the honest failure mode to avoid is manufacturing a verdict the data does
not support. Treat "inconclusive" as a real, informative outcome — and diagnose WHY before
you move on.

1. **Distinguish "refuted" from "could not decide".** A refutation means the predicted
   outcome demonstrably failed under a test that had the power to detect it. Inconclusive means
   the test could not distinguish the hypothesis from the null — often because it was
   underpowered, the signal and the confound were entangled, the witness was missing, or the
   pipeline had a bug. Do not report an inconclusive test as a refutation, and never report it
   as a confirmation.

2. **Triage a null result before believing it.** When a test comes back null, run the checklist
   before concluding the effect is absent:
   - **Sanity / positive control** — does the pipeline detect an effect you KNOW is present? If
     not, the null is a broken measurement, not a real absence.
   - **Power** — could the test have detected a meaningful effect at this sample size? An
     underpowered null is silence, not evidence of absence.
   - **Specification** — was the hypothesis operationalized correctly, or does the test measure
     something adjacent to what was claimed?
   - **Leakage in reverse** — did an over-aggressive control also remove the real signal?
   Only after these pass is a null credible as evidence the effect is small or absent.

3. **Extract the diagnosis, not just the outcome.** Every non-confirming result should yield a
   stated reason: which specific assumption failed, which condition broke it, what the observed
   value was versus what was predicted. "It didn't work" is not a result; "the effect was
   present on the fit distribution but vanished on the hold-out, consistent with overfitting"
   is. The diagnosis is what makes the next attempt smarter (see iterative refinement).

4. **Report inconclusive honestly and completely.** State what WAS shown, what remains
   undecided, and precisely what evidence would resolve it. An inconclusive result with a clear
   "here is the experiment that would settle it" is more valuable than a false confirmation.

5. **Do not p-hack or rescope toward a verdict.** When a test is inconclusive, resist swapping
   metrics, dropping inconvenient cases, or narrowing the claim post hoc until something crosses
   a threshold. Any reframing that was chosen BECAUSE it produced a positive result is not
   evidence — it must be pre-committed and retested, or reported as exploratory only.

An inconclusive result you diagnosed and reported truthfully advances the research. A confident
verdict you forced out of ambiguous data sets it back — because someone will act on it.
