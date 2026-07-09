---
name: multiple-testing-and-effect-size-rigor
description: Across genome-wide tests FDR correction is mandatory and an effect size with a CI is reported beside every p-value; guard the "significant but biologically trivial" trap and the double-dipping / winner's-curse / p-hacking failures
phase: evidence
scope: biology
priority: 31
---
When you test thousands of genes or millions of variants, an uncorrected p-value is meaningless — at
α = 0.05 a genome-wide scan returns hundreds of "hits" from noise alone. Multiple-testing correction
is not optional, and a small p is not a finding without an effect size that matters. This is where
most false biology should die.

1. **FDR by default across genome-wide tests.** Apply Benjamini-Hochberg and report q-values, not raw
   p, to any family of simultaneous tests — every gene in a DE scan, every variant, every gene set.
   Use Bonferroni / family-wise control when the tests are few or a single false positive is costly
   (GWAS 5×10⁻⁸ is essentially this). Report the corrected value, always.

2. **Correct against the RIGHT denominator.** The correction must count every test actually performed,
   including the ones you filtered out after peeking. Testing 20 000 genes and then correcting only
   the 50 that looked interesting is a hidden multiplicity that quietly reinflates the false-discovery
   rate back toward chance.

3. **Report effect size AND its CI beside every p.** log2 fold-change, odds/hazard ratio, Cohen's d,
   variance explained. With large n a "significant" q < 0.05 can sit on a log2FC of 0.05 — real but
   biologically trivial. Significance answers "is it non-zero?"; only the effect size answers "does it
   matter?" A bare threshold verdict hides that distinction.

4. **Do not double-dip (circular analysis).** Selecting features on the full dataset and then testing
   them on the same data is circular — defining clusters from a set of genes and then testing those
   same genes for cluster differences guarantees significance. Select on one split, test on another.

5. **Expect the winner's curse.** The effect sizes of your top hits are upward-biased precisely
   because they were selected for being extreme. The honest magnitude comes from an independent
   replication set, not from the discovery set that crowned them.

6. **No p-hacking.** Trying several normalizations, tests, covariate sets, or outlier rules and
   reporting only the one that crossed threshold is fabrication by selection. Pre-specify the pipeline
   and report every test run, not the survivor.

The dishonest moves here are quiet: an uncorrected or wrongly-denominated p, a "significant" hit whose
effect size is nil, a feature tested on the very data that selected it, or the best of many pipelines
reported as the only one. The bar: every genome-wide claim carries an FDR-corrected value against the
full test count, an effect size with a confidence interval, and a selection procedure that never
peeked at the outcome it is judged on. Significant-but-trivial is not a discovery.
