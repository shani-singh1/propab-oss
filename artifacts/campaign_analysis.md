# Campaign #2 Deep Analysis — What Propab Is ACTUALLY Doing (honest, node-by-node)

Analyzed all 131 nodes of campaign `5f99b96d` from the raw DB/tree data — NOT the
paper's "21 supported findings" headline.

## Verdict: a rigorous, HONEST experiment-runner — but NOT a discovery engine.
The verdict machinery is mechanically honest (it computes real ratios, refutes wrong
claims, keeps inconclusive dominant). But the EPISTEMIC content is near-empty: it
confirms known constants and measures a known algorithm's behavior. The "findings" are
not discoveries.

## The evidence

1. **18 of 21 "confirmed" findings are `best_known_table` lookups — circular.**
   `CONFIRMED-by-source = {combinatorial_computation: 3, best_known_table: 18}`.
   Example (verbatim evidence): *"greedy cap set size in F_3^6 will achieve ≥90% of the
   maximum known size (112)"* → evidence `cap_set_size: 112, construction_source:
   "best_known_table"`, notes *"best-known sizes=[112] … claim >=90 supported (computed
   112)"*. It LOOKED UP the known max (112), used it AS the greedy result, and "confirmed"
   that 112 ≥ 0.9·112. The answer was the input. This is a trivial rediscovery, not a
   computation of anything.

2. **The `trivial_rediscovery` detector FAILED.** That same node carries
   `"trivial_rediscovery": false, "discovery_worthy": true`. So the system confirmed a
   known value AND labelled it a worthy discovery. The detector doesn't catch
   best_known_table-sourced confirms.

3. **Zero novelty checking.** `event_counts` has NO `novelty` events — the campaign never
   called the literature service's `/novelty` endpoint. It cannot distinguish rediscovery
   from discovery, so re-deriving "cap set size at n=6 is 112" counts as a finding.

4. **Hypothesis monoculture — template sweep, not reasoning.** `hypothesis.generated: 1`
   (ONE LLM-generated hypothesis) vs 129 dispatched — 128 came from seed/synthesis
   template expansion. Top hypothesis skeletons across all 131 nodes: "randomized greedy
   search in F_q^n (top-k)" ×8, "forbidden-set ratio |A+A-A|/n" ×6, "greedy cap set CLP
   ratio" ×6, "greedy Sidon density" ×5, "k-step lookahead" ×4… It is parameter-sweeping
   ~8 experiment templates over (field q, dim n, lookahead k, top-k), not exploring
   conceptually distinct ideas.

5. **Generation is starved.** `hypothesis.tree_frontier_empty: 48` — the frontier ran
   dry 48 times and had to refill with more of the same templates. Only 58 LLM calls in 2h.

6. **Theme classification is domain-blind (corrupts convergence/diversity).** 17 nodes
   themed `diffusion_dynamics`, 3 `catalytic_geometry`, 1 `percolation` — ALL are actually
   cap-set/Sidon greedy-search hypotheses. The theme classifier slaps network/physics
   labels on additive-combinatorics work, so the diversity/monoculture engine is steering
   on garbage signals.

7. **Duplicate confirmations inflate the count.** Near-identical confirmed claims
   ("greedy cap set CLP ratio in F_q^n…" ×2, "sumset expansion ratio…" ×2) confirmed as
   separate findings; the 21 is padded by parameter-variants of the same result.

## What Propab NEEDS for REAL discovery (honest)

1. **Novelty gate (biggest lever).** Wire the literature `/novelty` endpoint into the
   confirmation/paper path: before a finding counts, ask "is this already known?" A claim
   that matches a best-known value or an established fact is a REDISCOVERY — demote it. The
   endpoint already exists and is unused.
2. **Kill the best_known_table circularity.** A verifier must COMPUTE the construction
   independently, never use the known answer as the result. A finding that re-derives a
   known constant must set `trivial_rediscovery=true` and NOT be `discovery_worthy`.
3. **Real hypothesis generation, not template sweeps.** Discovery needs conceptually novel
   hypotheses that target OPEN questions (beat a bound, find a denser construction, a new
   structural conjecture) — not (q,n,k) permutations of one greedy algorithm. 1 LLM
   hypothesis in 2h + 48 frontier-empties is a generation failure.
4. **Ambition: target the frontier of knowledge, not measurements of known algorithms.**
   "greedy reaches 90% of the known max" is a property of greedy, not new mathematics.
5. **Fix domain-aware theme classification** so the convergence/diversity engine steers on
   real structure.
6. **Dedup at the confirmation level** so parameter-variants don't inflate findings.

## Fix backlog (to loop on, no new campaign)
- **DISC1 · HIGH** — novelty gate: wire `/novelty` into confirmation; demote known claims.
- **DISC2 · HIGH** — best_known_table circularity + fix `trivial_rediscovery` detection.
- **DISC3 · HIGH** — hypothesis generation diversity/ambition (biggest, most open-ended).
- **DISC4 · MED** — domain-aware theme classification.
- **DISC5 · MED** — confirmation-level dedup (also seen as the paper's duplicate table rows).

---

## FIXES APPLIED (loop, no new campaign)

- **DISC1 · FIXED** (fix/disc1-novelty-gate, merged) — novelty gate wired: for each
  confirmed finding the finalize/paper path calls the literature `/novelty` endpoint;
  a `verdict=="known"` finding is demoted to a rediscovery and excluded from the
  headline discovery count. Honest non-blocking fallback (outage/uncertain never demotes).
  Live-verified `/novelty` works (returned "uncertain" for a specific numeric cap-set
  claim — the safe, conservative default; it won't false-demote a real discovery).
- **DISC2 · FIXED** (fix/disc2-rediscovery, merged) — `best_known_table`-sourced confirms
  are now flagged `trivial_rediscovery=True, discovery_worthy=False` and moved to the
  `rediscovery` bucket, excluded from the discovery count. **REPLAY on the real campaign's
  21 confirmed nodes: 19 flip to rediscovery, 2 remain genuine discoveries.**
- Merge unified both signals into one `rediscovery` bucket + one honest `counts["confirmed"]`
  (a finding is a rediscovery if the evidence is a table-lookup OR the literature knows it).

**Honest revised result of campaign 5f99b96d: ~2 discoveries + 19 rediscoveries, NOT 21
"confirmed findings."** The engine is now HONEST about not discovering.

### STILL OPEN (the hard part — what makes it actually discover)
- **DISC3 · HIGH** — real hypothesis generation. 1 LLM hypothesis vs 128 template
  expansions + 48 frontier-empties is the core gap; discovery needs conceptually novel,
  ambitious hypotheses targeting OPEN problems, not (q,n,k) sweeps of one greedy algorithm.
  This is a capability, not a patch — the biggest remaining lever.
- **DISC4 · MED** — domain-aware theme classification (cap-sets mislabeled diffusion/percolation).
- **DISC5 · MED** — confirmation-level dedup (near-identical parameter variants).

---

## VERIFICATION-LAYER AUDIT (trust nothing — findings, F1 verified by me)

A skeptical read-only audit of the verdict/verification layer. My earlier "0.0
false-confirm" benchmark tested `run_verdict_pipeline` — but 4/6 domains DON'T use it.

- **F1 · CRITICAL · VERIFIED (by me)** — `_plugin_verification_path` (sub_agent_loop.py:1285-1289)
  takes `domain_plugin.classify_verdict()` RAW; `run_verdict_pipeline`/the adversarial
  artifact gate runs ONLY in the `except` (i.e. only if classify_verdict crashes). So
  math_combinatorics, graph_invariants, genomics, enzyme_kinetics NEVER hit the
  permutation/label-shuffle/artifact gate — only materials + mandrake do. **The gate that
  underpins the honesty guarantee is not applied to most domains.** Fix: route every
  domain's verdict through the artifact gate (or an equivalent) after classify_verdict.
- **F2 · CRITICAL · LIKELY** — best_known_table circularity STILL confirms *claims*
  (constructors.py:329 via verifier.py cap-set sweep): `evaluate_numeric_claim` compares
  the claim threshold against the looked-up `cap_set_size` (table value) → "supported";
  the sweep+numeric-claim route may still force discovery_worthy. DISC2 fixed the plain
  single-lookup discovery flag; the sweep-claim path needs the same treatment. VERIFY the
  DISC2 coverage here myself.
- **F3 · HIGH** — deterministic path bypasses the gate on a gameable counter:
  `classify_evidence_type` calls it "deterministic" at `vt>=2`, and `scan_verification`
  counts `verified_true_steps` from ANY `{"verified":true}` output with no provenance/
  computed check → two boolean literals = "deterministic proof, confirmed", gate skipped.
- **F4/F5 · HIGH** — graph_invariants confirms TAUTOLOGIES in its own synthetic data:
  `algebraic_connectivity ≡ spectral_gap`, `modularity = 0.25·clustering + …` → a
  "modularity↔clustering correlation" is true by construction; confirmed at 0.90 with no
  gate. (DOM2b fixed modularity in the ADAPTER earlier — reconcile whether the verifier
  still uses the tautological version.)
- **F6 · HIGH** — confidence hard-pinned ≥0.92/0.95 for single-run deterministic checks;
  "replication (N independent checks)" re-runs the SAME deterministic function (not
  independent). 0.95 misrepresents a single-shot computation.
- **F7 · MED** — artifact-gate fallback branches "survive" with NO real null: label-shuffle
  heuristic survives on `lofo>0.05`; held-out-group branch on `lofo>-0.15` (a NEGATIVE R²
  passes); robustness branch on `lofo>0.0`.
- **F8 · MED** — gate strength depends on evidence SHAPE not quality: emit `verified:true`
  booleans (F3 path) to dodge the statistical de-rating.
**CIRCULAR-VERIFICATION SCAN (audit):** math cap-set numeric-claim = CIRCULAR (F2);
graph_invariants = CIRCULAR by construction (F4/F5); enzyme/genomics/materials/mandrake =
independent (real LOFO + label-shuffle null) — SAFE.
