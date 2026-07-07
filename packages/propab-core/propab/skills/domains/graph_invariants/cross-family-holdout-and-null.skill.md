---
name: cross-family-holdout-and-null
description: Design the cross-network-family holdout + label-shuffle null so an invariant correlation cannot confirm by accident
phase: experiment
scope: graph_invariants
priority: 34
---
The verifier decides an invariant claim with a cross-network-family leave-one-out plus
an adversarial permutation null. Design your test so a "confirmed" cannot be a
single-family fluke or a chance correlation.

1. **Name the source and target invariant explicitly.** The check reads the two named
   invariants from the hypothesis text; if it can identify none it REFUSES rather than
   defaulting, and the result maps to inconclusive. Write the claim so exactly the two
   invariants you mean are named (e.g. "algebraic connectivity" and "modularity"),
   using their standard phrasing, and state the claimed sign
   (positive / negative / holds-on-all-families).

2. **Require replication on the held-out real family.** Per-family correlation is
   computed on each real SNAP network; training correlation averages the non-held
   families and the HELD-OUT family must independently show the same relationship. A law
   that holds only on `collaboration` and vanishes on `communication` (or vice-versa) is
   topology-dependent and must NOT confirm. Name which family you hold out and predict
   its correlation before running.

3. **Beat the label-shuffle null.** Within the held-out family the target invariant is
   permuted against the source (~200 permutations), preserving each column's marginal
   distribution while destroying any real pairing. Your observed |correlation| must
   exceed the null's 95th percentile with permutation p < 0.05. "It looks correlated" is
   not enough; only clearing the shuffle null counts. When the held-out family is too
   small or degenerate to build a null, the check emits NO null stats and fails closed to
   inconclusive — so a claim that only "works" on a family that cannot support a null is
   not a finding.

4. **Fail closed, not open.** A correlation that clears the training families but not the
   holdout, or clears the holdout but not the shuffle null, is inconclusive/refuted — not
   a discovery. State up front the outcome that would REFUTE the claim (e.g. "held-out r
   within the shuffle null band ⇒ refuted") and accept it honestly.

The bar: a skeptic re-running your exact source→target invariant, holdout family, and
200-permutation null on the shipped SNAP frame is forced to your verdict — because the
held-out family and the shuffle null leave no room for a family-specific or chance story.
