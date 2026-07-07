---
name: avoid-density-confound-and-rediscovery
description: Guardrail against outcomes that track network scale/density and against re-confirming the textbook epidemic-threshold law
phase: hypothesis
scope: network_diffusion
priority: 32
---
Two disguises for a non-finding in contagion experiments. Reject both before committing.

**1. Scale/density confound is leakage, not a structural law.** Bigger, denser
subgraphs simply spread more. If your "predictor" is really a proxy for
`mean_degree`/size and the outcome tracks scale rather than TOPOLOGY, you have found the
obvious, not a law — and the within-family shuffle + cross-family holdout exist to catch
it. Guard against it in the hypothesis:
- Prefer heterogeneity / mixing predictors (⟨k²⟩/⟨k⟩ shape, gini, assortativity,
  hub-dominance) that are NOT just density. If you test `mean_degree` itself, say so and
  expect the holdout to expose it as a scale effect, not a discovery.
- The cross-topology holdout trains on one real family and tests another with a
  DIFFERENT absolute-outcome regime; a predictor that only tracks each family's scale
  will not replicate with the same sign and strength. Design the claim so its content is
  the STRUCTURE→outcome shape, not the level.

**2. Re-confirming the epidemic-threshold law is rediscovery.** That ⟨k²⟩/⟨k⟩ lowers the
epidemic threshold and heterogeneity aids spreading is TEXTBOOK
(Pastor-Satorras & Vespignani 2001, heterogeneous mean field). A hypothesis whose entire
content is "high ⟨k²⟩/⟨k⟩ ⇒ larger outbreak" re-derives a settled result even if it
passes the holdout.
- To be novel, go past the textbook: predict where the law is STRONGEST vs where it
  WEAKENS or REVERSES across the two real families, a contrast between competing
  predictors on the held-out family, an outcome-specific split (governs `outbreak_prob`
  but not `final_size`), or a divergence between the two simulators that theory does not
  anticipate.
- Mean-field threshold theory assumes uncorrelated, tree-like mixing; real
  collaboration/email subgraphs have clustering and assortativity that VIOLATE those
  assumptions — a claim about how the real law departs from the mean-field prediction is
  genuinely open and worth testing.

Litmus test before you submit: (a) would a denser/bigger subgraph pass this "law" for
free? → scale confound, re-scope to a structure-shape predictor. (b) Is the whole claim
just the mean-field threshold result restated? → rediscovery; sharpen to a
family-contingent, simulator-contingent, or outcome-contingent prediction the textbook
does not make.
