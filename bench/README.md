# bench/

Measurement harnesses that quantify individual layers of the Propab research
engine by **driving the real production code offline**, deterministically, so
future changes to that layer can be measured. Same convention as
`scripts/bench_campaign_convergence.py`: import the real `propab` functions, mock
only genuinely-external systems (LLM / sandbox / DB), be deterministic per seed,
average over seeds in `main()`, and print each metric with a one-line meaning.

---

## Evidence-binding layer — `bench_binding.py`

### What the layer does

Evidence binding is the **citation-integrity** gate. Before a node id is written
as supporting evidence for a belief, it must pass a deterministic subject-overlap
check. The layer's job is binary:

- **ACCEPT** a node that genuinely supports the belief (so belief formation is
  not starved), and
- **REJECT** irrelevant, fabricated, or cross-domain citations (so beliefs are
  not backed by citations that do not bear on them).

The historical failure was a **94.5% irrelevant-citation rate** — citations were
accepted almost unconditionally. The merged **A4 fix** made binding
domain-general: acceptance now requires *structured overlap* — a shared
mechanism/feature id, a shared scope subject term, or ≥2 salient content terms —
computed by `_structured_overlap`, on top of the original biology tag path. That
domain-generality is the thing this harness is built to verify.

### What it drives (the real code)

The bench calls the production entry point directly:

```
propab.evidence_binding.filter_node_citations(belief, [node_id], {node_id: node})
    -> binding_check -> _structured_overlap  (A4 domain-general path)
                     -> biology test-target tag path (LOFO / within-family / …)
```

Nothing that decides bind-vs-reject is reimplemented. The **only** authored input
is a labeled corpus of `(belief, candidate_node)` pairs — the thing a harness has
to provide. The bench self-locates the **A4-merged** `evidence_binding.py` (the
module that defines `_structured_overlap`) and asserts it loaded, so it can never
silently measure a pre-A4 copy of core. It records that module's path and git sha
in the artifact for provenance.

### The labeled corpus

5 domains × 4 classes = 20 pairs. Domains span **biology/mandrake, math
(Sidon / cap-set), physics, econometrics, and a generic domain** — the non-biology
domains exist specifically to catch the pre-A4 "recall ≈ 0 outside biology"
failure. Every label is **ground-truth by construction** (documented in the case
comments); only the bind/reject *decision* is measured, by the real binder.

| class | construction | correct outcome |
|-------|--------------|-----------------|
| `genuine`      | node whose structured fields (verdict / metric / claim scope / mechanism-or-feature id / salient terms) genuinely bear on the belief | **BIND** |
| `irrelevant`   | an unrelated node | REJECT |
| `fabricated`   | node that superficially echoes the belief's phrasing but shares **no** real mechanism / scope / salient subject term | REJECT |
| `cross_domain` | a **genuine** supporter, but for a different subject/domain | REJECT |

### Metrics

- **`binding_precision`** — of the citations that BOUND, the fraction that are
  genuine supporters. **This is the key integrity metric; target 1.0.** A
  false-accept (an irrelevant/fabricated/cross-domain citation that binds) *is*
  the citation-integrity failure this layer exists to prevent.
- **`binding_recall`** — of the genuine supporters, the fraction that BOUND.
  Target high — if genuine evidence is rejected, belief formation is starved.
- **`per_domain`** — precision & recall for each domain. This is how the harness
  **proves A4's domain-generality**: recall in the non-biology domains must be
  high, not ~0 (the pre-A4 failure). The run output flags each non-biology row.
- **`non_biology_mean_recall`** — the single headline number for that claim.

Precision and recall pull in opposite directions here, which is why both are
reported: a binder that accepts everything has recall 1.0 but terrible precision
(that is the pre-A4 regime), and a binder that accepts nothing has precision 1.0
but zero recall. The layer's job is high on both.

### Metric-moves sanity check

`sanity_always_accept()` monkeypatches the real `binding_check` to always accept,
re-runs, confirms `binding_precision` **drops**, then reverts. This proves the
metric is actually driven by the binder's decision (not by corpus construction)
and would catch any regression toward the pre-A4 always-accept behavior. The
result is printed and stored under `sanity_metric_moves` in the artifact.

### Run

From the worktree root:

```
PYTHONPATH="packages/propab-core;." python bench/bench_binding.py
```

Writes `artifacts/bench/binding_baseline.json`:
`{"metrics": {...}, "n_seeds": N, "config": {...}, "sanity_metric_moves": {...},
"git_sha": "<HEAD>", "timestamp": "<iso>"}`.

The seed only shuffles corpus order; the binder is order-independent, so every
seed yields identical metrics — that invariance is a cheap self-check on the
harness. `--seeds N` and `--out PATH` are available.

### Baseline reading (honest)

At the measured baseline the layer has **recall 1.0 in every domain** (all 5
genuine supporters bind, including all four non-biology domains — A4's
domain-generality holds; the pre-A4 ~0 non-biology recall is gone) and
**precision 0.5**. The precision gap is entirely **cross-domain false-accepts**:
irrelevant and fabricated citations are correctly rejected (0/5 each), but all 5
cross-domain genuine-for-a-different-subject findings bind, because they share ≥2
relationship/methodology terms (e.g. `elasticity`+`reduce`, `scales`+`spacing`,
`below`+`density`) or, in biology, tag identically under the LOFO/within-family
regexes. That is the concrete gap this harness hands to future work: cross-domain
subject discrimination is where the citation-integrity layer can still improve.
