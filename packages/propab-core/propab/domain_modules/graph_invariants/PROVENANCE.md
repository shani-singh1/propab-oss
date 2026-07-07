# graph_invariants — real network data provenance

The `graph_invariants` domain computes its cross-family invariant frame from
**real** public networks, not seed-generated textbook graph families. Each row of
`snap_subset_v1.csv` is the invariant fingerprint of a genuine connected induced
subgraph (randomised snowball / BFS sample) of one of the real networks below.

> Note: the repository's `.gitignore` excludes `data/`, so the raw edge-list files
> are **not** committed to git — they are cached on the operator's disk under
> `data/v1_candidates/`. This file is the committed, authoritative provenance
> record. `GraphInvariantsAdapter.ensure_cache()` also writes machine-readable
> provenance into `data/graph_invariants/snap_subset_v1.meta.json`
> (`synthetic: false`, `data_provenance: "real"`, per-family `sources`).

## Sources

| family          | file on disk (`data/v1_candidates/`) | network                                                   | SNAP dataset | citation |
|-----------------|--------------------------------------|-----------------------------------------------------------|--------------|----------|
| `collaboration` | `ca-GrQc.txt.gz`                     | arXiv General Relativity (GR-QC) co-authorship network    | [`ca-GrQc`](https://snap.stanford.edu/data/ca-GrQc.html) | Leskovec, Kleinberg, Faloutsos, "Graph Evolution: Densification and Shrinking Diameters", ACM TKDD 1(1), 2007 |
| `communication` | `email-Eu-core.txt.gz`               | e-mail network of a large European research institution   | [`email-Eu-core`](https://snap.stanford.edu/data/email-Eu-core.html) | Leskovec, Kleinberg, Faloutsos, ACM TKDD 1(1), 2007; Yin, Benson, Leskovec, Gleich, "Local Higher-order Graph Clustering", KDD 2017 |

Both edge lists are undirected (self-loops dropped); the largest connected
component is used as the sampling source for each family.

## How the invariant frame is built

For each real family, `adapter._real_frame()`:

1. loads the SNAP edge list and takes its largest connected component;
2. draws `SUBGRAPHS_PER_FAMILY` (=30) connected induced subgraphs of
   `SUBGRAPH_NODES` (=100) nodes via randomised snowball/BFS sampling
   (deterministic per-family seed derived from `RANDOM_SEED`=42);
3. computes the six exposed invariants on each subgraph — `spectral_gap`
   (adjacency λ1−λ2), `algebraic_connectivity` (Fiedler value), global
   `clustering_coefficient` (transitivity), `diameter`, `avg_degree`, and real
   Newman `modularity` of a Fiedler bipartition.

Every subgraph is a genuine piece of a real network — no topology is fabricated.

## Why this makes the cross-family question novel

The retired implementation generated 4 synthetic textbook families (Erdős–Rényi,
Barabási–Albert, Watts–Strogatz, grid lattice). A "confirmed" cross-family
invariant correlation there was a Newman-2003 textbook fact the engine could only
**rediscover**. With real collaboration and communication networks — structurally
very different (sparse high-clustering coauthorship vs. dense communication) — the
leave-one-network-family-out (LOFO) test in `verifier.py` asks whether an invariant
relationship holds across *real* network types, which is a genuinely open question.
The V3 label-shuffle permutation null in `verifier.py` is preserved unchanged and
is now fed these real invariants.
