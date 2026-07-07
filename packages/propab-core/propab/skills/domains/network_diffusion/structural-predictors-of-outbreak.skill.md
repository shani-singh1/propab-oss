---
name: structural-predictors-of-outbreak
description: Epidemic-threshold theory and which structural predictors of outbreak size are worth testing on real topologies
phase: hypothesis
scope: network_diffusion
priority: 30
---
You are proposing which STRUCTURAL feature of a real network predicts a simulated
diffusion outcome. The verifier ships two real SNAP topology families (`collaboration`
= arXiv GR-QC, sparse/high-clustering/assortative; `email` = EU-email, denser/lower
clustering), simulates outcomes with Monte-Carlo SIR or independent cascade on real
subgraphs, and measures the per-family rank correlation between one structural feature
and the outcome. The exposed structural features are:

- `k2_over_k1` — ⟨k²⟩/⟨k⟩, the degree-distribution moment that sets the epidemic
  threshold. Heterogeneous-mean-field theory (Pastor-Satorras & Vespignani 2001) says
  the critical transmissibility scales as ⟨k⟩/⟨k²⟩, so high ⟨k²⟩/⟨k⟩ ⇒ vanishing
  threshold and easy takeoff. This is the canonical heterogeneity predictor.
- `degree_gini`, `degree_cv` — degree inequality / heterogeneity by other measures.
- `max_degree_ratio` — hub dominance (k_max / ⟨k⟩): does a single super-spreader drive
  the outbreak?
- `mean_degree` — density/scale (a confound to be wary of — see the leakage guardrail).
- `clustering` — local transitivity: triangles can trap or slow spread.
- `assortativity` — degree–degree mixing: assortative hubs form a resilient core.

Outcomes: `final_size` (fraction ever infected/active) and `outbreak_prob` (fraction of
seeds reaching a macroscopic outbreak). Simulators: `sir` and `cascade`.

Reach for mechanistic, falsifiable structural laws:

- **Threshold-theory predictions with a sign.** Hypothesize, with a mechanism, that
  ⟨k²⟩/⟨k⟩ POSITIVELY predicts final size across real topology families — but sharpen it
  past the textbook by predicting where it should be STRONGEST or should BREAK (e.g.
  "the ⟨k²⟩/⟨k⟩→final-size law is strong in the clustered collaboration family but
  weakened in email because clustering redundifies transmission paths").
- **Competing predictors.** Pit heterogeneity (⟨k²⟩/⟨k⟩, gini) against hub dominance
  (max_degree_ratio) or mixing (assortativity): which one still predicts the outcome on
  a HELD-OUT real family? Naming the rival predictor makes the claim falsifiable.
- **Clustering / assortativity as suppressors.** A claim that clustering or assortative
  mixing NEGATIVELY predicts outbreak size (redundant edges, hub-core trapping) is a
  real, sign-testable conjecture the holdout can kill.
- **Outcome-specific laws.** A predictor may govern `outbreak_prob` (takeoff near
  threshold) but not `final_size` (super-critical saturation), or vice-versa — state
  which outcome, and predict the other as a failure mode.

Every claim must name: the structural feature, the outcome, the claimed sign, and the
real family whose held-out replication would decide it.
