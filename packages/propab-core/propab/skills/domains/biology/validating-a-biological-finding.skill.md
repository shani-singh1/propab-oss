---
name: validating-a-biological-finding
description: Promote a candidate to a finding only after it replicates in an independent cohort, is confirmed by an orthogonal assay, and survives our leakage discipline — leave-one-group-out plus a within-group label-shuffle null; a single dataset is never enough
phase: evidence
scope: biology
priority: 34
---
A result from a single dataset is a hypothesis, not a finding. Expression atlases, GWAS cohorts, and
screens are riddled with dataset-specific batch structure, so a signal that lives in one dataset is as
likely an artifact of that dataset as a fact about biology. Validation means the effect survives being
taken out of the exact conditions that produced it.

1. **Replicate in an independent cohort/dataset.** Re-find the effect in a separately collected sample
   — a different GEO series, a different biobank, a different platform. Cross-validation WITHIN the
   discovery set is not external validation: it shares the batch structure and the winner's curse, so
   a good CV score there proves the model memorized the dataset, not the biology.

2. **Confirm with an orthogonal assay.** A finding should not hinge on one measurement modality. An
   RNA-seq hit checked by qPCR or protein (Western / IHC); a GWAS locus supported by an eQTL or a
   functional readout. Agreement across modalities that fail in DIFFERENT ways is what rules out a
   shared technical artifact — agreement within one modality does not.

3. **Apply our leakage discipline — the right split, the right null.** Split by the unit that carries
   the confound (patient / donor / batch / cohort), never by individual samples, so no group straddles
   train and test. Establish significance against a within-group label-shuffle permutation null under a
   leave-one-group-out (LOGO/LOFO) evaluation: the effect must exceed the shuffle's 95th percentile,
   not merely "look predictive". A model that only works when the same donor appears on both sides is
   leaking, not predicting.

4. **Pre-state where it should break.** Name, in advance, a held-out group in which the effect should
   hold and one in which it should fail, then report the per-group result honestly — including the
   groups that did not carry it. A finding that expects to win everywhere is untested.

5. **Distrust a single batch.** If the signal collapses under leave-one-batch-out, or concentrates in
   one batch or site, it is a batch effect wearing biology's clothes — report it as such.

The dishonest move is calling a discovery-set result "validated" on the strength of internal
cross-validation, a single assay, or a null that the leakage never let fail. The bar: an independent
cohort re-finds it, an orthogonal assay confirms it, it beats a within-group-shuffle null under a
group-wise hold-out, and you have reported the groups where it broke. Anything short of that is a
candidate, and must be labeled as one.
