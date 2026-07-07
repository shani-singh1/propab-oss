# bench/

Engineering-metric harnesses that quantify one layer of the Propab research
engine each. Every harness DRIVES the real production code (imported from
`propab` / `services`) and prints a small set of numbers that regress cleanly, so
a future change to that layer can be measured instead of eyeballed.

Convention (matches `scripts/bench_campaign_convergence.py`):

- Drives the REAL production functions under test. The only mocks are
  genuinely-external systems (raw experiment numbers, LLM output), clearly
  labeled as deterministic stand-ins. Everything that DECIDES a metric is the
  real code.
- Deterministic per `seed`; `main()` runs `>= 10` seeds and averages.
- Each metric is printed with a one-line plain-English meaning.
- Writes a baseline JSON under `artifacts/bench/`.

---

## Verdict layer — `bench_verdict.py`

Quantifies the **verification / verdict layer**: the code that must **CONFIRM**
genuine findings and **REFUSE** artifacts, null effects, fabricated inputs, and
un-adversarially-tested claims. This is the honesty gate a campaign's every
finding passes through.

### What is real (the code under test)

- `propab.verdict_pipeline.run_verdict_pipeline` — the single production entry
  point (classify → artifact gate → OOD gate → scope gate). Every case is scored
  by calling THIS, not a reimplementation.
- `propab.verdict_pipeline.classify_evidence_type` — routes a case to the
  deterministic / lofo / statistical / unknown branch. Carries the **V2 guard**
  (a bare `verified_true_steps` counter no longer bypasses the artifact gate).
- `propab.artifact_verification.run_artifact_gate` / `_survives_permutation` /
  `_survives_label_shuffle_lofo` — the adversarial artifact tests, reached inside
  the pipeline.
- `services.worker.permutation_null.compute_label_permutation_null` — the merged
  D2 label-permutation null. Every statistical case's `permutation_p` is computed
  by this function from the two real Gaussian arrays the case draws, so the gate
  reasons about a genuine Monte-Carlo p-value, not a hand-set one.

### The labeled corpus (8 evidence types)

Ground-truth-by-construction is the `should_confirm` label on each case. The
MEASURED quantity is the pipeline's verdict (a case is "confirmed" iff the
pipeline returns the literal verdict `"confirmed"`).

| # | case type | should_confirm | why the layer must decide this way |
|---|-----------|:---:|---|
| 1 | `real_effect_statistical` | ✅ | genuine +1.2 sd mean difference, n=120 ≥ 100, `computed` provenance; real passing permutation null |
| 2 | `null_effect_statistical` | ❌ | two groups, same distribution → permutation p large → significance gate fails |
| 3 | `fabricated_input_statistical` | ❌ | **W1b guard** — a PASSING null but `stat_input_provenance="agent_literal"` |
| 4 | `small_n_statistical` | ❌ | genuine effect but n=60 < 100; `_survives_permutation` needs n ≥ 100 |
| 5 | `genuine_lofo` | ✅ | `lofo_r2 > label_shuffle_null_p95` and `label_shuffle_permutation_p < 0.05` |
| 6 | `artifact_lofo` | ❌ | `lofo_r2 <= label_shuffle_null_p95` — the label shuffle explains it |
| 7 | `deterministic_proof` | ✅ | a real proof-method marker with ≥ 2 verified-true checks |
| 8 | `bare_counter_deterministic` | ❌ | **V2 hole** — `verified_true_steps > 0` but NO real proof method and no null |

### Metrics

- **`false_confirm_rate`** — fraction of `should_confirm=False` cases the pipeline
  CONFIRMED. **The key honesty metric; target 0.**
- **`confirm_recall`** — fraction of `should_confirm=True` cases confirmed. Target 1.
- **`balanced_accuracy`** — mean(recall on genuine, specificity on artifacts).
- **`per_evidence_type`** — confirm rate for each of the 8 types, to localize a regression.

### Run

```
PYTHONPATH="packages/propab-core;." python bench/bench_verdict.py
```

Writes `artifacts/bench/verdict_baseline.json`. `propab` is imported from the
**main checkout** (which carries the merged verdict-layer fixes), so the recorded
`git_sha` is the code actually under test.

### Baseline (12 seeds)

```
false-confirm rate  : 0.0    (0 artifact/null/fabricated cases confirmed)
confirm recall      : 1.0    (all genuine findings confirmed)
balanced accuracy   : 1.0
```

All 8 per-type rates exactly correct. **Metric-moves sanity check** (built into
the run): monkeypatching the real artifact tests to always `survived=True` lifts
`false_confirm_rate` **0.0 → 0.2**, proving the metric is wired to the real gate.
(Only 2 of 4 negative cases route through those two tests; the null-effect case is
stopped earlier at the significance gate and the fabricated case at the W1b
provenance check — the honest, expected behavior.)

> Note: `false_confirm_rate=0.0` measures the gate **given honest provenance**. It
> does not cover a *tool* that mislabels fabricated variance as `computed`
> provenance (issue TOOL1) — that leak is upstream in the tool, not in this gate.

---

## Evidence-binding layer — `bench_binding.py`

### What the layer does

Evidence binding is the **citation-integrity** gate. Before a node id is written
as supporting evidence for a belief, it must pass a deterministic subject-overlap
check: **ACCEPT** a node that genuinely supports the belief (so belief formation
is not starved), and **REJECT** irrelevant, fabricated, or cross-domain citations.

The historical failure was a **94.5% irrelevant-citation rate**. The merged **A4
fix** made binding domain-general: acceptance now requires *structured overlap* —
a shared mechanism/feature id, a shared scope subject term, or ≥2 salient content
terms (`_structured_overlap`), on top of the original biology tag path.

### What it drives (the real code)

```
propab.evidence_binding.filter_node_citations(belief, [node_id], {node_id: node})
    -> binding_check -> _structured_overlap  (A4 domain-general path)
                     -> biology test-target tag path
```

Nothing that decides bind-vs-reject is reimplemented. The bench **self-locates the
A4-merged** `evidence_binding.py` (the module defining `_structured_overlap`) and
asserts it loaded, so it can never silently measure a pre-A4 copy of core.

### The labeled corpus

5 domains (biology/mandrake, math, physics, econometrics, generic) × 4 classes = 20
pairs. Every label is ground-truth by construction; only the bind/reject *decision*
is measured.

| class | construction | correct outcome |
|-------|--------------|-----------------|
| `genuine`      | node whose structured fields genuinely bear on the belief | **BIND** |
| `irrelevant`   | an unrelated node | REJECT |
| `fabricated`   | node that echoes the belief's phrasing but shares no real subject term | REJECT |
| `cross_domain` | a genuine supporter, but for a different subject/domain | REJECT |

### Metrics

- **`binding_precision`** — of the citations that BOUND, the fraction that are
  genuine supporters. **Key integrity metric; target 1.0.**
- **`binding_recall`** — of the genuine supporters, the fraction that BOUND (target high).
- **`per_domain`** / **`non_biology_mean_recall`** — proves A4 domain-generality
  (recall in non-biology domains must be high, not ~0).

### Run

```
PYTHONPATH="packages/propab-core;." python bench/bench_binding.py
```

`sanity_always_accept()` monkeypatches the real `binding_check` to always accept
and confirms `binding_precision` drops (0.5 → 0.25), proving the metric is driven
by the binder.

### Baseline reading (honest)

Recall **1.0 in every domain** (A4's domain-generality holds; the pre-A4 ~0
non-biology recall is gone) and **precision 0.5**. The precision gap is entirely
**cross-domain false-accepts**: irrelevant and fabricated citations are correctly
rejected (0/5 each), but all 5 cross-domain "genuine-for-a-different-subject"
findings bind, because they share ≥2 relationship/methodology terms. **That is the
concrete improvement target this harness hands to future work: cross-domain
subject discrimination** (issue BND1).

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
