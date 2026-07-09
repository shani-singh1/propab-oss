---
name: experimental-design-in-biology
description: Design a biology experiment so a confirmation cannot be a batch effect, a technical-replicate illusion, or an uncontrolled biological confound — biological replication, balanced batches, and the right positive/negative controls (extends core/experimental-design-and-controls)
phase: experiment
scope: biology
priority: 22
---
Biology's characteristic false finding is not fraud — it is a batch effect, a pseudoreplicated n, or
a confound (sex, age, passage, sample quality) that rode along with the condition. This skill extends
`core/experimental-design-and-controls` for the specific ways wet-lab and omics data manufacture
effects. Build the controls into the design; you cannot rescue a confounded experiment with a
cleverer test afterward.

1. **Count BIOLOGICAL replicates, not technical ones.** n is the number of independent biological
   units (animals, patients, independently grown cultures), not repeated measurements, lanes, or
   aliquots of one sample. Averaging technical replicates is fine; treating them as independent n is
   pseudoreplication and inflates significance. Three flow cells from one mouse is n = 1, not n = 3.

2. **Batch effect is the dominant confound — never confound batch with condition.** If all cases run
   on day 1 and all controls on day 2, biology and batch are inseparable and no downstream analysis
   recovers the truth. Randomize or balance samples across batches, plates, lanes, and operators;
   record every batch variable; model or regress out batch. A perfectly confounded batch is
   unrecoverable, so the only fix is at design time.

3. **Install the bio-specific controls.** Negative: a non-targeting sgRNA / scrambled siRNA /
   vehicle / IgG that MUST show no effect if the pipeline is clean — if it fires, the pipeline is
   detecting an artifact. Positive: a manipulation with a known-true response (a validated knockdown,
   a canonical DE gene) that MUST fire, or your assay is blind and a null means nothing. Spike-ins
   (e.g. ERCC) calibrate the technical null.

4. **Enumerate and neutralize the standard biological confounds.** Sex, age, genetic background,
   cell-line passage number, RNA integrity (RIN), sample-handling and freeze-thaw time, ancestry
   (for any genetic association), and collection site. For each, balance it across arms or measure
   and adjust for it. A confounder you did not name is one you did not control.

5. **Randomize processing order and physical layout; block by batch.** Plate position, extraction
   order, and sequencing lane all imprint signal — assign them at random and compare conditions
   WITHIN a block so the block-level nuisance cancels.

6. **Size the study before you run it.** Omics is expensive, but a 3-vs-3 tested genome-wide confirms
   noise as readily as signal. State the effect size worth detecting and whether this n could see it
   (`power_analysis`); an underpowered "null" has refuted nothing.

What makes such an experiment dishonest is presenting a confounded or pseudoreplicated result as a
clean one — reporting technical replicates as biological n, or a case/control difference that is
really a day-of-processing difference. The bar: for every arm, name the biological unit of
replication, show that no batch or handling variable aligns with the condition, and point to the
negative control that stays silent and the positive control that fires. A confirmation a batch could
have produced is not a finding.
