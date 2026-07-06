# bench/

Measurement harnesses for individual layers of the Propab research engine. Each
harness DRIVES the real production code (imported from `propab` /
`services.orchestrator`) and reports deterministic metrics so future changes can
be measured — the same idea as LitQA2 for the literature layer.

Run everything from the repo root (or a worktree root) with core on the path:

```
PYTHONPATH="packages/propab-core;." python bench/bench_generation.py
```

Each bench writes a JSON baseline under `artifacts/bench/`.

---

## Generation layer — `bench_generation.py`

### What it measures (and what it deliberately does NOT)

Hypothesis generation must produce on-topic, implementable, non-duplicate,
well-scoped hypotheses. That job has two halves:

1. **Raw candidate proposal** — done by the LLM. This is non-deterministic and
   **OUT OF SCOPE** for this benchmark. Its quality shows up in campaign
   outcomes, not here.
2. **Post-generation filtering** — a chain of DETERMINISTIC filters that decide
   which raw candidates survive. This is where the generation-layer fixes live
   (scope/numeric-aware dedup, the relevance gate, the implementable-methodology
   gate, the scope-validity gate). **This benchmark measures exactly that chain.**

So the honest headline is: **this quantifies the deterministic post-generation
filters, not LLM creativity.** A perfect score here means the filters behave as
specified on a labelled bank; it says nothing about how good the raw ideas were.

### How it works

For five domains — math/Sidon, biology/mandrake, genomics, physics/graph-invariants,
and a generic/econ question with no domain plugin — the bench builds a **fixed,
hand-labelled candidate bank**. Every candidate carries a ground-truth tag
(`duplicate`, `narrowing_child`, `offtopic`, `unimplementable`,
`ungrounded_scope`, `distinct_survivor`) and is written with inline scope lines
(`Population:` / `Distribution:` / …) so the real scope parser can read it.

Each candidate is then fed through the **real** functions:

| Filter | Real function driven |
| --- | --- |
| Duplicate dedup | `campaign_synthesis._is_duplicate_frontier_candidate` |
| Relevance gate | `domain_modules.registry.hypothesis_is_on_topic` (each plugin's `hypothesis_on_topic`) |
| Implementable methodology | `synthesis_diversity.methodology_implementable` vs each domain's `implementable_methodologies()` |
| Scope validity | `scoped_claim.validate_scoped_claim` + `scoped_claim.is_boilerplate_scope` |
| Whole chain (cross-check) | `campaign_synthesis.apply_synthesis_to_frontier` (returns its own reject counters) |

The per-filter tallies are cross-checked against the counters that the full
`apply_synthesis_to_frontier` pipeline emits when the same bank is pushed through
it end-to-end — a faithfulness guard that the isolated measurements match the
chained production path.

The banks are ground-truth **by construction**; the labels ARE the ground truth.
They were validated to be defensible against the real functions before being
baked in — e.g. a `duplicate` is a pure rephrasing with the *same* claim title,
numbers, and scope tokens (so the real `text_similarity` ≥ 0.85 AND the parameter
fingerprint is identical); a `narrowing_child` keeps the claim title but changes
the numeric range (math) or the scope-line sub-population (biology), which is
exactly what the real dedup treats as a distinct experiment.

### Metrics

| Metric | Meaning | Target |
| --- | --- | --- |
| `duplicate_pass_rate` | Fraction of TRUE-duplicate candidates that SURVIVED dedup. **Key metric.** | **0** |
| `narrowing_survival_rate` | Fraction of genuine scope-narrowing children that survived dedup (a narrowing child must NOT be mistaken for a duplicate). | **1** |
| `offtopic_rejection_rate` | Fraction of off-topic candidates rejected by the relevance gate, over the domains that have a real gate. | high |
| `relevance_precision` | Of the candidates that survive the whole deterministic chain, the fraction that are on-topic. | 1 |
| `methodology_implementable_rate` | Fraction of non-implementable candidates correctly rejected, over the domains that supply an implementable-keyword filter. | 1 |
| `scope_validity_rate` | Fraction of ungrounded/boilerplate-scope candidates flagged as scope-invalid. | 1 |

`duplicate_pass_rate` and `narrowing_survival_rate` are two sides of the same
dedup coin: the fix that makes the dedup scope/numeric-aware must drive the first
to 0 **without** dropping the second below 1. Reporting both prevents a naive
"reject everything similar" from looking good.

### Honest limitations

- **Deterministic filters only.** Raw LLM idea quality is not measured here (see
  above). A high score does not mean the engine generates *good* hypotheses — it
  means the filters keep the good-vs-bad *labelled* candidates on the right side.
- **Off-topic gate coverage is partial.** `mandrake` and the generic/econ
  question do **not** override `hypothesis_on_topic` (they accept everything), so
  their off-topic candidates are **excluded** from `offtopic_rejection_rate`
  (measuring rejection where no gate exists would be dishonest). The JSON reports
  `offtopic_domains_with_gate` (3/5). Off-topic filtering for those domains
  happens later (campaign-level relevance scoring / scope gate), out of this
  bench's scope.
- **`methodology_implementable` is a substring OR-match** over the full scoped
  text. A candidate is only a clean `unimplementable` test if NO implementable
  keyword leaks into its scope lines; the banks are written to avoid that leak.
  This surfaced a real property of the filter (it cannot reject an unimplementable
  methodology if the scoped claim happens to name an implementable technique),
  documented here rather than hidden.
- **Small, fixed bank.** Rates are over a few dozen labelled candidates, chosen to
  exercise each filter's decision boundary — not a statistical sample of real LLM
  output. The value is regression sensitivity (does a code change move a metric?),
  not an absolute quality estimate.

### Metric-moves sanity check

The bench re-runs the banks with one real filter short-circuited and asserts the
corresponding metric moves, proving the metric is actually wired to that filter:

- disable dedup → `duplicate_pass_rate` jumps from 0 to 1 (every duplicate now
  survives).
- disable the relevance gate → `offtopic_rejection_rate` drops to 0 (no off-topic
  candidate is rejected).

Both are printed at the end of a run and stored under `sanity_check` in the JSON.

### Output

Writes `artifacts/bench/generation_baseline.json`:

```json
{
  "metrics": { ... },
  "counts": { ... },
  "n_seeds": 5,
  "config": { ... },
  "end_to_end_cross_check": { ... },
  "sanity_check": { ... },
  "git_sha": "<HEAD sha>",
  "timestamp": "<iso>"
}
```

`n_seeds` is the number of banks (one per domain). The run is fully deterministic,
so re-running reproduces the same metrics for a given code state.

> Note: the first import loads all domain plugins via the real registry; one
> plugin (`enzyme_kinetics`) pulls a heavy dependency and can take ~1–2 min to
> import the first time. This is a one-time startup cost, not per-bank work.
