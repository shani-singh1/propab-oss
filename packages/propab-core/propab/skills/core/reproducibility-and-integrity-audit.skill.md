---
name: reproducibility-and-integrity-audit
description: The self-audit an agent runs on its OWN result before claiming it — can it be reproduced from the recorded witness, and was it confirmed or merely found in the data?
phase: evidence
scope: core
priority: 38
---
Before you report a result as your own, put it through the audit a hostile reviewer would run — but run
it on YOURSELF, first, honestly. Adapted from formal reproducibility-and-integrity review: the point is
to catch a result that looks confirmed but was manufactured by the analysis, by chance, or by selective
telling. Three questions decide it.

1. **Can it be REPRODUCED from the recorded witness?** A result you cannot regenerate is a claim, not a
   finding. Confirm that everything needed to reproduce it is recorded and sufficient: the seed(s), the
   exact data/split used, the code/procedure, the parameters, and the resulting witness object. Then run
   the check: does re-executing from the recorded state yield the SAME result? If regeneration drifts,
   the effect depends on something you did not record (hidden state, an unfixed seed, an ordering
   artifact) and the finding is not yet real. Record the witness so an independent party could recompute
   it without your narration.

2. **Was the analysis EXPLORATORY or CONFIRMATORY — and which data decided it?** This is the sharpest
   distinction and the easiest to blur.
   - **Exploratory** — the hypothesis was found by looking at the data (searching conditions, subgroups,
     thresholds, or transforms until something popped). Exploratory results generate hypotheses; they
     cannot confirm them.
   - **Confirmatory** — the hypothesis and the decision rule were fixed BEFORE the deciding data were
     seen, and the result was evaluated once on data untouched by that search.
   A hypothesis discovered in a dataset cannot be confirmed by the SAME dataset — that is circular. If the
   claim was found and tested on the same data, label it exploratory and demand a fresh hold-out before
   any confirmatory language. State honestly which mode produced the result.

3. **Screen for selective reporting, p-hacking, and HARKing.** These are the ways a null becomes a
   "finding" without any single lie:
   - **Selective reporting** — showing the conditions/seeds/subgroups that worked and omitting those that
     did not. Report ALL of them, including the null and failed ones; a result that only survives when the
     failures are hidden is not a result.
   - **P-hacking** — trying many analyses (tests, cutoffs, exclusions, covariates) and reporting the one
     that crossed threshold. Count how many analyses were actually run and correct for it, or pre-commit
     the single analysis. An effect that needed a search to appear probably will not replicate.
   - **HARKing** — writing up a hypothesis-found-after-the-fact as if it were predicted in advance. Keep
     the discovery order honest: a post-hoc explanation of a pattern is a conjecture for the NEXT test,
     not a confirmation of this one.

The verdict this audit produces: a result is reportable as confirmed only if (a) it reproduces from the
recorded witness/seed/data/code, AND (b) it was decided confirmatorily on data untouched by the search
that found it, AND (c) it survives with every attempted analysis and every failed condition disclosed.
Fail any one and downgrade the wording to exploratory / suggestive / inconclusive. The purpose is not to
pass the audit — it is to find out, before anyone else does, whether your own result is real.
