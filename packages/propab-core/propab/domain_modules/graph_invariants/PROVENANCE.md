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

| family           | file on disk (`data/v1_candidates/`) | network                                                   | dataset | citation |
|------------------|--------------------------------------|-----------------------------------------------------------|---------|----------|
| `collaboration`  | `ca-GrQc.txt.gz`                     | arXiv General Relativity (GR-QC) co-authorship network    | SNAP [`ca-GrQc`](https://snap.stanford.edu/data/ca-GrQc.html) | Leskovec, Kleinberg, Faloutsos, "Graph Evolution: Densification and Shrinking Diameters", ACM TKDD 1(1), 2007 |
| `communication`  | `email-Eu-core.txt.gz`               | e-mail network of a large European research institution   | SNAP [`email-Eu-core`](https://snap.stanford.edu/data/email-Eu-core.html) | Leskovec, Kleinberg, Faloutsos, ACM TKDD 1(1), 2007; Yin, Benson, Leskovec, Gleich, "Local Higher-order Graph Clustering", KDD 2017 |
| `infrastructure` | `power-US-Grid.txt.gz`               | US Western States power grid (nodes=generator/transformer/substation, edges=power supply line) | KONECT [`opsahl-powergrid`](http://konect.cc/networks/opsahl-powergrid) | Watts & Strogatz, "Collective dynamics of 'small-world' networks", Nature 393, 440–442 (1998) |

- **Fetched URL:** `http://konect.cc/files/download.tsv.opsahl-powergrid.tar.bz2`
  (KONECT TSV bundle; the `out.opsahl-powergrid` edge list has 4,941 nodes and 6,594
  edges). The `%`-commented header lines were stripped and the two integer edge
  columns re-written as a gzipped, `#`-commented whitespace edge list matching the
  loader's format; **no edges were added, removed or reweighted**.
- **Access date:** 2026-07-07.
- **License:** The KONECT collection redistributes this dataset for open research
  use (the underlying network is from Watts & Strogatz 1998, via Tore Opsahl's
  dataset page). The two social nets come from Stanford SNAP (open research use;
  BSD-2-Clause on the SNAP software). Only a small transformed subset (invariant
  fingerprints of sampled induced subgraphs) is cached locally; the raw edge lists
  are git-ignored.

All three edge lists are undirected (self-loops dropped); the largest connected
component is used as the sampling source for each family (the power grid is a single
connected component).

### Why `power-US-Grid` is a genuinely distinct topology class

Collaboration and communication are both compact social graphs (small diameter,
high clustering, hub-bearing degree distributions). The power grid is the opposite:
a near-planar infrastructure mesh. On 100-node snowball subgraphs the three families
separate cleanly:

| family           | mean avg-degree | mean clustering | mean diameter |
|------------------|-----------------|-----------------|---------------|
| `collaboration`  | ~7.3            | ~0.48           | ~5            |
| `communication`  | ~24             | ~0.46           | ~3            |
| `infrastructure` | ~2.6            | ~0.11           | ~9            |

So the leave-one-network-family-out holdout is now a genuine 3-way cross-topology
test (train on 2, hold out 1) rather than a weak train-on-1/hold-out-1 split.

## How the invariant frame is built

For each real family, `adapter._real_frame()`:

1. loads the edge list and takes its largest connected component;
2. draws `SUBGRAPHS_PER_FAMILY` (=50) connected induced subgraphs of
   `SUBGRAPH_NODES` (=100) nodes via randomised snowball/BFS sampling
   (deterministic per-family seed derived from `RANDOM_SEED`=42);
   50 (not 30) subgraphs/family so the label-shuffle null retains power on the
   sparse power-grid subgraphs — it only adds real subgraphs, fabricating nothing;
3. computes the six exposed invariants on each subgraph — `spectral_gap`
   (adjacency λ1−λ2), `algebraic_connectivity` (Fiedler value), global
   `clustering_coefficient` (transitivity), `diameter`, `avg_degree`, and real
   Newman `modularity` of a Fiedler bipartition.

Every subgraph is a genuine piece of a real network — no topology is fabricated.

## Why this makes the cross-family question novel

The retired implementation generated 4 synthetic textbook families (Erdős–Rényi,
Barabási–Albert, Watts–Strogatz, grid lattice). A "confirmed" cross-family
invariant correlation there was a Newman-2003 textbook fact the engine could only
**rediscover**. With three real, structurally very different networks —
sparse high-clustering coauthorship, dense communication, and a near-planar
low-degree power grid — the leave-one-network-family-out (LOFO) test in
`verifier.py` asks whether an invariant relationship holds across *real* network
types, which is a genuinely open question. Adding the third (infrastructure)
family strengthens the holdout from train-on-1/hold-out-1 to train-on-2/hold-out-1
and, empirically, is discriminating: `clustering_coefficient → avg_degree`
survives the label-shuffle null on all three held-out families (denser subgraphs
pack in both more edges per node and more closed triangles), whereas `spectral_gap →
avg_degree` — which held across the two social families — collapses on the sparse
power-grid subgraphs, and `diameter → modularity` survives on none. The V3
label-shuffle permutation null in `verifier.py` is preserved unchanged and is now
fed these real invariants.
