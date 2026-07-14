"""evolve/targets — thin adapters wrapping existing domain-module verifiers as `Problem`s.

Keep these THIN. The verifier logic already exists and is battle-tested; a target's job is to expose
it through the `Problem` contract, source a REAL best-known record, and supply good seed programs.

  ecc.py        Target A — best-known [n,k,d] linear codes (coding_theory.compute_min_distance,
                BEST_KNOWN_TABLE, trivial_rediscovery).  Base rate: many open cells, steady output.
  graph_conj.py Target B — counterexample hunting on open graph conjectures (graph_invariants).
                Asymmetric: a finite counterexample is FULLY settled by the verifier.
  erdos143.py   Erdős #143 side-run — separation-condition sets, finite-prefix verifier.
                NOTE: a true counterexample is INFINITE, so this can only produce evidence /
                a candidate construction pattern — never a settled result. Bounded side-bet.
"""
