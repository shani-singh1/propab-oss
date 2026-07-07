---
name: cross-clade-holdout-and-redundancy-artifact
description: Design cross-RT-family holdouts that separate a real biophysical signal from sequence-redundancy artifacts and family-ID proxies, under small n
phase: experiment
scope: mandrake
priority: 40
---
A within-family RT-activity signal is only a finding if it is not (a) the sequence-redundancy
artifact — nearest-neighbour leakage between related sequences — and (b) not a feature that
secretly encodes which family a sequence belongs to. The dataset is small (~56 sequences, 7
families), so the design must be conservative and the two rival explanations must be pitted
against each other, not confirmed in isolation.

**The two artifacts you must rule out:**
1. **Sequence-redundancy artifact.** Sequences within a clade are similar, so a model can
   "predict" a held-out sequence by memorising a near-neighbour rather than learning biophysics.
   Guard: a low-identity split. Restrict the test to held-out sequences below a sequence-identity
   threshold (~<25–50%, echoing the characteristically low cross-RT-class identity ceiling) to the
   training set; if the signal collapses (R² < 0) under the low-identity split, it was redundancy,
   not mechanism.
2. **Family-ID proxy (fold/geometry leakage).** Foldseek TM scores to named references and
   catalytic-triad geometry can encode family identity directly. A confirmation driven by such a
   feature is the clade label in disguise. Guard: require the feature to beat the FAMILY-MEAN
   baseline out-of-family, and treat a feature whose only value is separating clades as a
   surrogate, not a discovery.

**Adversarial design bar (the two rivals must move in opposite directions):**
1. **State both nulls.** The observed leave-one-family-out (LOFO) R² must exceed the family-label-
   shuffle null (permutation p < 0.05) AND survive the low-identity split. Report the per-family
   LOFO breakdown, the LOFO gap (within-family minus LOFO), the family-mean baseline, the bootstrap
   CI, and the permutation p — a single mean number is not evidence on n this small.
2. **Design the critical experiment as a discriminator.** Choose the next test because its result
   would push "family-specific mechanism (signal survives clustered split)" and "redundancy
   artifact (signal collapses under low-identity holdout)" in OPPOSITE directions — not because it
   refines one belief. Do not silently fall back to another cross-family feature-combination search;
   both rivals must clear the same artifact-verification bar.
3. **Read the disposition honestly.** A confirmed result whose features are thermal-only is likely
   the obvious thermostability axis (archive, not discovery). A confirmed null with negative LOFO
   and a large within-vs-LOFO gap is a family surrogate (reject). The interesting "gold" outcome is
   a mixed feature set with POSITIVE LOFO that survives group removal AND the low-identity split.
   Beware "fake diversity" where nearly all confirmations reduce to the thermal axis.
4. **Compute on the real panel; never on hand-typed numbers.** Agent-supplied values cannot confirm.

**Small-n discipline:** with 7 families and ~56 sequences, one family can dominate a mean and a
single sequence can swing a within-family fit. Widen intervals, distrust a signal carried by one
family, and prefer conservative, discriminating claims. The bar: a skeptic re-running your LOFO
could not re-explain the result as redundancy OR as the family label — because the low-identity
split and the family-mean baseline leave no room for either.
