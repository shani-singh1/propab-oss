# Genomics dataset provenance

The genomics adapter serves a **real subset of the GTEx v8 median gene-level
TPM atlas** — not synthetic data. `GenomicsPlugin.uses_synthetic_data()`
returns `False` whenever the real dataset is on disk (it reads the `synthetic`
flag written into the cache `.meta.json`).

## Source

- **Dataset:** GTEx v8 median gene-level TPM,
  `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz`.
- **Origin:** GTEx Consortium. *The GTEx Consortium atlas of genetic
  regulatory effects across human tissues.* **Science** 369, 1318–1330 (2020).
  https://doi.org/10.1126/science.aaz1776
- **Fetched URL:** `https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz`
- **Portal:** https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression
- **Version:** GTEx Analysis v8 (dbGaP accession phs000424.v8).
- **Access date:** 2026-07-07.
- **License:** GTEx summary/median expression matrices are open-access
  (no controlled-access restriction applies to the median-TPM summary; the
  GTEx portal provides it for public download). Only a small transformed
  subset is cached locally.

## What is cached (`data/genomics/gtex_subset_v1.csv`)

Long format, one row per (gene, tissue):

| column | meaning | provenance |
|---|---|---|
| `gene_id` | Ensembl gene id (version stripped) | real (GTEx) |
| `tissue` | one of 10 organ labels | real GTEx tissue columns |
| `expression` | log2(median TPM + 1) | derived from real median TPM |
| `expression_variance`, `mean_expression`, `cv_across_tissues` | gene-level summaries | derived |
| `tissue_specificity_tau` | Yanai τ index (0 housekeeping … 1 specific) | derived |

Ten organs are represented by one detailed GTEx tissue column each
(Brain–Cortex, Heart–Left Ventricle, Liver, Lung, Muscle–Skeletal,
Skin, Whole Blood, Adipose–Subcutaneous, Thyroid, Colon–Sigmoid). Genes are
filtered to those expressed (>0.1 TPM in ≥5 tissues) and reduced to the 1000
most variable genes.

## Why the LOFO structure is unchanged

The verifier still does **leave-one-tissue-out** ridge + **tissue-label-shuffle
null**, grouping each gene by its dominant (max-expression) tissue. Only the
data changed from synthetic to real. On the real data the gene-level summary
relationships are largely within-row (tautological under the null), so the
tissue-shuffle null is not rejected — the correct, non-rediscovery outcome.
