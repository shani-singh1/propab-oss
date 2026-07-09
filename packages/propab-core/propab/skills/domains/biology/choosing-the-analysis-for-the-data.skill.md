---
name: choosing-the-analysis-for-the-data
description: Match the statistical model, normalization, and null to the data-generating process — bulk RNA-seq vs scRNA-seq vs proteomics vs GWAS vs survival each demand a different test; the wrong one silently manufactures or erases an effect
phase: experiment
scope: biology
priority: 24
---
There is no default test. The right analysis is dictated by how the data were generated — counts or
continuous, paired or unpaired, zero-inflated, censored, compositional — and using a method built for
a different process is a silent way to fabricate or destroy a signal. Name the data type and its
noise model first; only then pick the test, the normalization, and the null.

1. **Identify the data-generating process before choosing a test.** Count vs continuous,
   over-dispersed, zero-inflated, right-censored, paired/repeated-measures, or compositional — each
   rules whole families of tests in or out. A t-test on raw counts, or on censored survival times, is
   simply the wrong model, and its p-value is meaningless however small.

2. **Bulk RNA-seq / count data** → a negative-binomial GLM (DESeq2 / edgeR / limma-voom) with
   library-size normalization (median-of-ratios / TMM), NOT a t-test on raw or naively-scaled counts.
   Use dispersion shrinkage on low-n designs; report shrunken log2 fold-change, not just the p.

3. **scRNA-seq** → normalize for sequencing depth (log-normalize / SCT), expect dropout, cluster
   then call markers. Critically, for condition-vs-condition DE, aggregate to PSEUDOBULK per sample:
   treating individual cells as replicates is pseudoreplication (n is the number of donors, not
   cells) and yields absurdly tiny, meaningless p-values.

4. **Proteomics (MS)** → log-transform, then handle missing-not-at-random explicitly (missingness
   correlates with abundance, so naive zero/mean imputation biases results), median/quantile
   normalize, and use a moderated t (limma-style) rather than a per-protein t-test.

5. **GWAS** → per-variant regression (linear/logistic) with population structure controlled
   (principal components or a linear mixed model), QC first (MAF, call rate, HWE), and the
   genome-wide threshold 5×10⁻⁸ — not 0.05. Uncontrolled ancestry is the classic confound that
   yields floods of false associations.

6. **Survival / time-to-event** → Kaplan-Meier + log-rank and Cox proportional-hazards; you MUST
   model censoring and check the proportional-hazards assumption. Report hazard ratios with CIs, never
   a mean-difference test that discards the censored cases.

7. **Enrichment (GO / KEGG / GSEA)** → hypergeometric or rank-based GSEA against an EXPLICIT, correct
   background (the genes actually testable in your assay, not all genes in the genome), FDR-corrected.

The dishonest move is letting the method invent a result the data cannot support — a t-test that
ignores over-dispersion or censoring, cells counted as replicates, or an enrichment against the wrong
background. The bar: state the data-generating process, justify that the chosen test's assumptions
match it, and name the normalization and the null. If the model does not fit the data, the number it
produces is not evidence.
