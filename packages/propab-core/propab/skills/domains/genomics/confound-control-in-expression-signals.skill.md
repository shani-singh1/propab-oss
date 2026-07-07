---
name: confound-control-in-expression-signals
description: Separate a real cross-tissue expression signal from technical artifacts (library size, gene length, mean-variance coupling) before claiming a finding
phase: experiment
scope: genomics
priority: 40
---
An apparent expression signal is worthless until you have shown it is not a technical
artifact. When you specify how a genomics hypothesis will be tested under leave-one-tissue-out
(LOFO) against the tissue-label-shuffle null, design the test so a confirmation cannot be
explained by a nuisance variable rather than by biology.

**Name the confounds explicitly and neutralise each:**
- **Library size / sequencing depth.** Raw counts scale with how deeply a sample was
  sequenced. A "signal" that is really depth is a normalization artifact. Confirm the effect
  survives on depth-normalized values (TPM/CPM/log2 as the atlas provides) and does not track
  a per-tissue depth covariate.
- **Gene length.** Longer genes accrue more reads under count-based quantification. If your
  feature or target correlates with gene length, a length-driven association can masquerade as
  a mechanism. Test whether the effect holds after conditioning on (or residualizing against)
  gene length.
- **Mean–variance coupling.** In expression data, variance grows with the mean, so
  "high-variance genes" and "high-expression genes" are entangled. A claim about variance that
  is really about mean is not a finding. Break the coupling (e.g. compare within mean-matched
  bins, or use CV / a variance-stabilized transform) before attributing signal to variance.
- **Dominant-tissue assignment leakage.** The verifier groups each gene by its max-expression
  tissue. If a feature encodes which tissue is the maximum, the model can recover the group
  label without any transferable biology — a leakage path that inflates within-sample fit but
  should still collapse under a proper hold-out. Make sure your feature is not a proxy for the
  grouping variable.

**Design bar (adversarial):**
1. State the null concretely: the observed mean leave-one-tissue-out R² must exceed the 95th
   percentile of the tissue-label-shuffle null (permutation p < 0.05). "It looks predictive"
   is not evidence; the shuffle p is.
2. Add a confound control as a SECOND null wherever a nuisance variable is plausible: re-run the
   LOFO on values with the confound removed (depth-normalized, length-residualized, or
   mean-matched) and confirm the effect does not vanish. A signal that disappears once the
   confound is controlled was the confound.
3. Run the test on the real computed atlas values, never on numbers you supplied by hand;
   agent-typed inputs cannot confirm.
4. Declare, before running, at least one held-out tissue in which the effect SHOULD break, and
   report the per-tissue LOFO honestly (which tissues carried it, which did not).

If the effect survives the tissue-shuffle null AND the relevant confound control AND transfers
to a tissue it never saw, it is a candidate finding. If it only survives before the confound
control, report it as a technical artifact, not biology.
