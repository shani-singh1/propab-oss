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

### What is mocked (external inputs only, labeled)

- The two-group numeric outcome arrays are drawn from `random.Random(seed)`
  Gaussians — a deterministic stand-in for a sandbox experiment's raw numbers.
- Deterministic-proof / LOFO metadata (proof-method marker, `lofo_r2`,
  `label_shuffle_null_p95`) mirror exactly the fields the real
  `_build_mandrake_evidence` / verification tools attach; they are set to
  ground-truth-by-construction values (a genuine effect vs. an artifact).

### The labeled corpus (8 evidence types)

Ground-truth-by-construction is the `should_confirm` label on each case. The
MEASURED quantity is the pipeline's verdict (a case is "confirmed" iff the
pipeline returns the literal verdict `"confirmed"`).

| # | case type | should_confirm | why the layer must decide this way |
|---|-----------|:---:|---|
| 1 | `real_effect_statistical` | ✅ | genuine +1.2 sd mean difference, n=120 ≥ 100, `computed` provenance; real passing permutation null |
| 2 | `null_effect_statistical` | ❌ | two groups, same distribution → permutation p large → significance gate fails |
| 3 | `fabricated_input_statistical` | ❌ | **W1b guard** — a PASSING null but `stat_input_provenance="agent_literal"` (agent typed the numbers, not sandbox-computed) |
| 4 | `small_n_statistical` | ❌ | genuine effect but n=60 < 100; `_survives_permutation` needs n ≥ 100 for a significance-only path |
| 5 | `genuine_lofo` | ✅ | `lofo_r2 > label_shuffle_null_p95` and `label_shuffle_permutation_p < 0.05` — real cross-group transfer |
| 6 | `artifact_lofo` | ❌ | `lofo_r2 <= label_shuffle_null_p95` — the label shuffle explains it (signal tracks group identity) |
| 7 | `deterministic_proof` | ✅ | a real proof-method marker (`symbolic_proof` / `exact_check` / …) with ≥ 2 verified-true checks |
| 8 | `bare_counter_deterministic` | ❌ | **V2 hole** — `verified_true_steps > 0` but NO real proof method (bare `significance` counter) and no adversarial null; must not bypass the gate |

### Metrics (and WHY each measures this layer's job)

- **`false_confirm_rate`** — fraction of `should_confirm=False` cases the pipeline
  CONFIRMED. **The key honesty metric; target 0.** A non-zero value means an
  artifact / null / fabricated result escaped the gate — precisely the failure
  the verdict layer exists to prevent.
- **`confirm_recall`** — fraction of `should_confirm=True` cases confirmed. Target
  1. Guards against the opposite failure: a gate so paranoid it refuses genuine
  findings, which would make the whole engine emit nothing.
- **`balanced_accuracy`** — mean(recall on genuine, specificity on artifacts). One
  number that regresses if EITHER honesty or recall breaks.
- **`per_evidence_type`** — confirm rate for each of the 8 types, so a regression
  localizes to the exact guard that broke (e.g. only `artifact_lofo` rising tells
  you the label-shuffle test regressed).

### Run

From the worktree root:

```
PYTHONPATH="packages/propab-core;." python bench/bench_verdict.py
```

Writes `artifacts/bench/verdict_baseline.json`
(`{"metrics", "n_seeds", "config", "git_sha", "timestamp"}`). `propab` is imported
from the **main checkout** (which carries the merged verdict-layer fixes), so the
recorded `git_sha` is the main checkout's HEAD — the code actually under test.

### Baseline (12 seeds, main checkout `97d8779`)

```
false-confirm rate  : 0.0    (0 artifact/null/fabricated cases confirmed)
confirm recall      : 1.0    (all genuine findings confirmed)
balanced accuracy   : 1.0
```

All 8 per-type rates are exactly correct (1.0 for the three confirm-cases, 0.0
for the five refuse-cases).

### Metric-moves sanity check (built into the run)

A metric that cannot fail is worthless, so the harness ends by monkeypatching the
real artifact tests (`_survives_permutation`, `_survives_label_shuffle_lofo`) to
always report `survived=True`, re-running the corpus, and confirming
`false_confirm_rate` jumps. Baseline **0.0 → broken 0.2** (the `artifact_lofo` and
`small_n_statistical` cases now leak through the crippled gate), proving the
metric is wired to the real gate. The other two negative cases stay refused
because they are stopped *earlier* by distinct guards — the null-effect case at
the significance gate, the fabricated case at the W1b provenance check — which is
the honest, expected behavior.
