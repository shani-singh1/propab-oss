---
name: exploratory-data-analysis
description: Interrogate the data for distributions, missingness, outliers, imbalance, and — critically — leakage BEFORE modeling, so an artifact is caught before it becomes a false confirmation
phase: experiment
scope: core
priority: 26
---
Exploratory data analysis is not "load the data and print summary statistics." It is a deliberate
hunt for the artifacts that would otherwise masquerade as a finding. Do it BEFORE any modeling or
testing: the point is to catch the boring, mechanical explanation — a leak, an imbalance, a
mislabeled subset — while it is still cheap to catch, not after it has confirmed a false claim.

1. **Inspect the distribution of every variable, don't just summarize it.** A mean and standard
   deviation hide bimodality, skew, floor/ceiling effects, and impossible values. Look at the actual
   shape (a histogram or empirical CDF) per variable and per group. Flag values outside physical or
   logical bounds, spikes at sentinel values (0, -1, 999) that really mean "missing," and units that
   silently differ across sources. These shape the choice of test later.

2. **Map missingness and treat it as data, not a nuisance.** Quantify how much is missing per variable
   AND per unit, then ask WHY it is missing: is it missing at random, or does missingness itself
   correlate with the group or the outcome? Missingness that tracks the label is a leak and a confound
   — dropping those rows silently biases the result. Never listwise-delete without checking, and never
   impute in a way that lets information cross from evaluation data into the fit.

3. **Find outliers and decide their status before they decide your result.** Identify extreme points
   and, for each, determine whether it is an error (fix or remove, with the rule stated in advance) or
   a real tail (keep — it may be the phenomenon). A single leverage point can create or destroy an
   apparent effect. Pre-commit the handling rule; do not delete points because they weaken the story.

4. **Check class / group balance and base rates.** Report the size of each group and the base rate of
   the outcome. Severe imbalance makes accuracy meaningless, inflates chance agreement, and lets a
   model "win" by predicting the majority. Know the balance before choosing a metric or a test, and
   pick metrics that survive the imbalance.

5. **HUNT FOR LEAKAGE — this is the step that saves you.** Before any model sees the data, ask whether
   any feature encodes the answer:
   - *Label/outcome leakage:* does a feature proxy the group label or the outcome — an identifier, a
     source/batch marker, a timestamp, an ordering artifact, or a value computed downstream of the
     outcome? If a nonsense feature that carries no real signal but tracks the group would predict the
     outcome, you have a leak, not a mechanism.
   - *Contamination across the fit/evaluation boundary:* do the same unit (subject/source/cluster)
     appear on both sides of a split? Was any normalization, feature selection, or thresholding fit
     using the evaluation data? Both leak the answer in and manufacture effects that vanish on truly
     unseen data. Split by the unit that could carry the confound, and fit every transform on the
     training portion only.
   Name the leak channels you checked and how you closed each; an unchecked channel is an open door.

6. **Look at joint structure, and keep exploration honest.** Examine how variables co-vary (pairwise
   and against the outcome) to spot redundancy, collinearity, and suspiciously perfect predictors — a
   feature that predicts too well is usually a leak, not a discovery. But remember EDA is exploratory:
   patterns you find here are hypotheses to be tested on held-out data with a pre-committed rule, never
   confirmations. Testing a pattern on the same data that suggested it is circular.

The bar: by the end of EDA you can state the data's distributions, its missingness mechanism, its
outlier handling, its balance, and — above all — the leakage channels you ruled out, so that any
effect found later cannot be dismissed as an artifact you never looked for.
