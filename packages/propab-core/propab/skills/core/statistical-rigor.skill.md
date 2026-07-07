---
name: statistical-rigor
description: Frame the question first, pick the right test, check its assumptions, correct for multiple comparisons, and report effect sizes with CIs — never a bare p-value
phase: experiment
scope: core
priority: 24
---
A p-value is not a finding. When a claim is statistical, the statistic must sit ON TOP OF a
real adversarial null — it does not replace it. A tiny p on a trivial effect, or a "significant"
result whose assumptions are violated, is an artifact, not evidence. Work the pipeline in order;
skipping a step manufactures false confirmations.

1. **Frame the question BEFORE you touch the data.** State the hypothesis, the outcome and the
   predictor(s), the design (independent vs paired, how many groups), and the ONE test you will
   run — in advance. Choosing the test after peeking at the results is p-hacking: it invalidates
   the p-value. Pre-commit the decision rule (the threshold, the direction, the correction) so
   the analysis is confirmatory, not a search for something quotable.

2. **Pick the test with a decision tree, not by habit.** Match the test to the data type and design:
   - *Two groups, continuous:* normal → t-test (paired vs independent as the design dictates);
     non-normal → Mann-Whitney U (independent) or Wilcoxon signed-rank (paired).
   - *Three-or-more groups, continuous:* normal → one-way / repeated-measures ANOVA;
     non-normal → Kruskal-Wallis (independent) or Friedman (paired).
   - *Categorical outcome:* chi-square, or an exact test when any expected cell count is small.
   - *Association between two continuous measures:* Pearson (normal) or Spearman/rank (non-normal).
   - *Outcome modeled from predictors:* linear regression (continuous outcome) or logistic
     regression (binary outcome).
   Counts, time-to-event, and factorial designs have their own tests — do not force everything
   through a t-test.

3. **CHECK THE ASSUMPTIONS and report the checks.** A test is only valid where its assumptions hold.
   - *Normality* (for parametric tests): inspect a quantile plot and the distribution, backed by a
     formal test; at large n trust the plot over the test's p-value, since the test flags trivial
     departures. On violation, transform or switch to the rank-based alternative above.
   - *Homogeneity of variance:* test it; on violation use the variance-robust variant (e.g. the
     unequal-variance t-test) rather than the pooled one.
   - *Independence of observations:* the deadliest and least testable assumption. Repeated measures,
     clustering, or shared sources make observations non-independent and shrink the true sample
     size far below the row count — model the structure (paired/mixed/clustered) or the p-value is
     fiction. State how independence is justified.
   Do not report a result whose assumptions you did not check.

4. **CORRECT FOR MULTIPLE COMPARISONS and say which correction you used.** Running many tests and
   reporting the ones that "hit" guarantees false positives. When you test a family of hypotheses,
   control the error rate: Holm (or another family-wise method) to bound the chance of ANY false
   positive; Benjamini-Hochberg to control the false-discovery rate when you expect several true
   effects. Count every test you ran, not just the ones you liked, and disclose the method.

5. **Report EFFECT SIZE + CONFIDENCE INTERVAL alongside the p-value.** The p-value only signals
   whether an effect is distinguishable from noise; the effect size says whether it MATTERS, and
   the interval says how precisely you know it. Lead with the standardized effect size and its 95%
   confidence interval; a result is scientifically interesting only if the effect is meaningful AND
   its interval excludes the trivial. A minuscule p on a negligible effect is not a discovery.

6. **Do a POWER analysis — a priori when you can.** Before running, compute the sample size needed
   to detect the smallest effect that would matter, given your alpha and target power. A study
   too small to see the effect proves nothing when it comes out "null": non-significant is NOT
   evidence of no effect. When planning is impossible, report the minimum detectable effect for
   your actual n. Do NOT report post-hoc "observed power": it is a deterministic function of the
   p-value and is circular and misleading.

The bar: the reported statistic must be the honest output of a pre-committed, assumption-checked,
multiplicity-corrected test on real computed data, expressed as an effect size with its interval —
and it must clear a genuine adversarial null, not stand in for one.
