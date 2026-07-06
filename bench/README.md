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
