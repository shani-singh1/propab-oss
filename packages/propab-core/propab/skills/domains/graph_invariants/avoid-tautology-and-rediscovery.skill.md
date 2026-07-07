---
name: avoid-tautology-and-rediscovery
description: Guardrail against tautological invariant pairs and re-confirming a Newman-2003 textbook correlation
phase: hypothesis
scope: graph_invariants
priority: 32
---
In this domain two failure modes disguise themselves as findings. Reject both BEFORE
you commit to a hypothesis.

**1. Tautological invariant pairs are not a finding.** A "relationship" between two
invariants that are defined in terms of each other, or that are algebraically forced to
move together, tells you nothing about real networks — it would hold on ANY graph. The
verifier's six invariants were deliberately decoupled so a real relationship is
testable:
- `modularity` is a genuine Newman Q of a partition, NOT a closed-form function of the
  clustering coefficient. Do NOT propose "modularity correlates with clustering" as a
  discovery on the premise that they measure the same thing — if there IS a cross-family
  law linking them it must be argued mechanistically and survive the holdout, not
  asserted as definitional.
- `spectral_gap` (adjacency λ1 − λ2) and `algebraic_connectivity` (Laplacian λ2) live on
  DIFFERENT operators. Do not claim one "equals"/"is" the other, or that they trivially
  track — a claim that they DIVERGE is interesting; a claim that they are the same is
  wrong.
- Never pair an invariant with a monotone re-expression of itself (e.g. avg_degree vs a
  density that is a function of avg_degree at fixed size). r≈1 by construction is an
  artifact, not a law.

**2. Re-confirming a textbook correlation is rediscovery, not discovery.** Classical
network-science results (Newman 2003 and the community-detection literature) already
establish, on canonical graphs, relationships such as high clustering co-occurring with
community structure, and expander-like graphs having a large spectral gap. A hypothesis
whose ENTIRE content is "the known Newman-2003 correlation holds" is a rediscovery even
if it passes the holdout — it re-derives a settled fact on new data.
- To be novel, target what is genuinely OPEN across these real families: does the
  relationship's SIGN, STRENGTH, or existence CHANGE between a real collaboration
  network and a real communication network? A regime shift, a reversal, or a documented
  ABSENCE on a held-out real family is a real result; a textbook restatement is not.
- If the honest expected outcome is "yes, the known correlation reappears", either
  sharpen the claim to a quantitative, family-contingent prediction the textbook does
  NOT make, or pick a different pair. The engine's holdout will confirm a rediscovery
  just as readily as a discovery — the anti-rediscovery burden is on the hypothesis.

Litmus test before you submit: (a) could this "relationship" be true of every graph by
definition? → tautology, discard. (b) Is the whole claim already in a network-science
textbook for canonical graphs? → rediscovery, re-scope to a family-contingent,
currently-unknown prediction.
