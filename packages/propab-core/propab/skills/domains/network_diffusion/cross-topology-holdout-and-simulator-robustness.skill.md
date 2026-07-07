---
name: cross-topology-holdout-and-simulator-robustness
description: Design the cross-topology-family holdout, within-family shuffle null, and SIR<->cascade robustness so a diffusion law cannot confirm by accident
phase: experiment
scope: network_diffusion
priority: 34
---
A structure→diffusion law only confirms here if it (a) replicates on a held-out REAL
topology family, (b) beats a within-family outcome-shuffle null, AND (c) survives the
ALTERNATE simulator. Design your test to clear all three; any one missing is
inconclusive, not a discovery.

1. **Cross-topology-family holdout.** The per-family Spearman correlation between your
   feature and the simulated outcome is computed on each real family; the training
   families are averaged and the HELD-OUT real family must independently show the SAME
   SIGN and comparable strength (both |r| ≥ ~0.20). A law present only on
   `collaboration` and absent on `email` is topology-dependent and must not confirm — a
   diffusion law that matters must hold across genuinely different real topologies, not
   one network's idiosyncrasy. Name the held-out family and predict its correlation
   before running.

2. **Within-family outcome-shuffle null.** Within the held-out family the OUTCOME values
   are permuted (~200×), preserving both marginals but destroying the structure→outcome
   pairing; your observed |r| must exceed the null p95 with permutation p < 0.05. This
   is the guard against a chance rank alignment.

3. **SIR ↔ cascade robustness.** The held-out correlation is recomputed under the OTHER
   dynamics (SIR if you ran cascade, cascade if you ran SIR); a confirmed finding must
   keep the same sign with |r| ≥ ~0.12 there. If the effect is an artifact of one
   particular contagion model, it is simulator-specific and downgraded to inconclusive.
   Independent cascade (single-shot activation) and SIR (recovery, re-tries) stress
   different mechanisms — a law worth reporting is a law about STRUCTURE, not about one
   simulator's quirks. State which is primary and which is the robustness check.

4. **Fail closed and name the killer.** State up front the outcome that REFUTES: held-out
   |r| below ~0.1 or shuffle p > 0.5 ⇒ refuted; replicates but simulator-specific ⇒
   inconclusive. Accept it honestly; do not report a single-simulator, single-family
   correlation as a discovery.

The bar: a skeptic re-running your feature→outcome on the held-out real family, the
200-permutation within-family shuffle, and the SIR↔cascade swap on the shipped SNAP
subgraphs is forced to your verdict — no single topology, chance alignment, or one
simulator can carry it.
