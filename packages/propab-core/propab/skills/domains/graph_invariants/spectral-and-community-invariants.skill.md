---
name: spectral-and-community-invariants
description: Technique menu for cross-network-family invariant hypotheses (Laplacian spectrum, algebraic connectivity, spectral gap, Newman modularity)
phase: hypothesis
scope: graph_invariants
priority: 30
---
You are proposing a claim about how graph invariants relate ACROSS real network
families (the verifier ships two structurally distinct SNAP networks:
`collaboration` = arXiv GR-QC co-authorship, sparse/high-clustering; `communication`
= EU-email, dense/low-clustering). Each row is an invariant fingerprint of a real
connected induced subgraph. The exposed invariants are exactly six, each an
INDEPENDENT structural quantity (none is a closed-form function of another):

- `spectral_gap` — gap between the two largest ADJACENCY eigenvalues (λ1 − λ2).
  Governs mixing / expansion; large gap ⇒ good expander, fast random-walk mixing.
- `algebraic_connectivity` — the Fiedler value, the SECOND-SMALLEST LAPLACIAN
  eigenvalue λ2(L). Controls how hard the graph is to disconnect and the speed of
  consensus/synchronization. It is a DIFFERENT spectrum from `spectral_gap` (Laplacian
  vs adjacency) — do not treat the two as interchangeable.
- `modularity` — real Newman Q of a Fiedler (spectral) bipartition: how much more
  densely nodes connect within their community than a degree-preserving null predicts.
- `clustering_coefficient` — global transitivity (triangle density).
- `diameter`, `avg_degree` — geometric / density scale.

Reach for genuine spectral / community-structure hypotheses, not a bare "A correlates
with B":

- **Spectral–structural bridges.** Hypothesize a sign and mechanism linking a spectral
  invariant to a combinatorial one — e.g. "algebraic connectivity grows with average
  degree SUBLINEARLY in real communication subgraphs but supralinearly in collaboration
  subgraphs", or "high modularity SUPPRESSES the adjacency spectral gap because
  community walls slow mixing". State WHY the mechanism should hold, then let the
  cross-family holdout decide.
- **Sign-and-strength claims, not existence claims.** The verifier tests a per-family
  correlation and a cross-family holdout, so a useful claim names a direction
  (`correlation_positive` / `correlation_negative`) or asserts it holds on ALL families
  (`holds_all_families`), and predicts the held-out family will replicate it.
- **Community-vs-spectrum contrasts.** Because `modularity` and
  `algebraic_connectivity` are computed from different objects (a partition's Q vs the
  Laplacian's λ2), a claim that they DIVERGE across families (e.g. modular but
  well-connected communication subgraphs) is a real, non-obvious structural conjecture.
- **Regime-dependence as the finding.** The most interesting claims say the
  relationship itself CHANGES between real families (present in collaboration, absent or
  reversed in communication) — that is exactly what the cross-family holdout can confirm
  or kill.

Anchor every claim on the two REAL families and name the held-out family whose
replication would decide it. A finding that was never at risk of failing on the
held-out family is not a finding.
