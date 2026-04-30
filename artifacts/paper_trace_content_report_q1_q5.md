# Paper-vs-Trace Content Report (Q1-Q5)

## Scope
- Source: `artifacts/questions_audit_report.json`
- Sessions analyzed:
  - Q1 `d287eeb3-e479-4772-ac23-d29294b64840`
  - Q2 `f597f83d-2482-48c0-953f-d8cfc9bac012`
  - Q3 `03b369e7-0628-40b4-b553-f69fe7c190fa`
  - Q4 `49049ddf-e0f1-4c1c-b82f-ffb55d6bf4e4`
  - Q5 `8564b79b-3b5f-4af7-8d88-4dbf1e78ba4b`

## Overall observations
- Every question completed and produced TeX/PDF (`HAS_TEX True HAS_PDF True`).
- Papers are now internally consistent with verdicts: all hypotheses shown as `inconclusive`, and abstracts avoid claiming breakthroughs.
- Results sections include structured evidence fragments (`metric_value`, `n_metric_steps`, `relevance_score`) rather than opaque legacy text.
- Main quality gap is scientific signal, not plumbing: traces are rich, but evidence rarely includes `p_value`/`effect_size` needed for confirmation.
- A secondary quality gap is hypothesis prose specificity: occasional templated/fallback statement styles still appear.

---

## Q1: Activation functions and transformer stability
- **Trace behavior**
  - 50 steps, tool-heavy execution (`activation_statistics`, `build_transformer`, `compare_attention_variants`, `train_model`, `lr_range_test`, etc.).
  - Evidence samples contain numeric metrics (`49.6094`, `298.0`) and moderate relevance scores.
- **Paper content**
  - Abstract explicitly states: `confirmed=0, refuted=0, inconclusive=5`.
  - Methods/Results include concrete tool call narratives with parameters and execution durations.
  - No overclaim language; verdict remains inconclusive with confidence around `0.5-0.6`.
- **What is actually happening**
  - The system executes substantial experiments and records metrics.
  - The confirmation gate rejects them due to insufficient significance/effect evidence.

## Q2: Pre-norm vs post-norm in noisy MLPs
- **Trace behavior**
  - 45 steps with broad coverage including `hessian_analysis`, `gradient_noise_scale`, `regularization_effect`, and `hyperparameter_sweep`.
  - One evidence sample shows an error signature: `(34, 'Numerical result out of range')`.
- **Paper content**
  - Abstract remains honest/inconclusive and references trace-backed synthesis only.
  - Results section attaches the structured evidence block to each hypothesis row.
- **What is actually happening**
  - Mixed-quality tool outputs: many usable metrics plus at least one numeric instability/error path.
  - Paper reflects this as inconclusive rather than fabricating certainty.

## Q3: Learning-rate warmup and final quality
- **Trace behavior**
  - 50 steps with representative warmup-related tooling (`lr_range_test`, `train_model`, `compare_attention_variants`, `ablation_study`).
  - Confidence spread (`0.4-0.5`) aligns with partial evidence richness.
- **Paper content**
  - Abstract/inference language is restrained and consistent with inconclusive ledger.
  - Results include hypothesis-specific evidence summaries and trace IDs.
- **What is actually happening**
  - The system is collecting many proxy metrics but still not reaching significance-grade confirmation.
  - Paper generation preserves uncertainty instead of overstating.

## Q4: Optimizer ranking across loss geometries
- **Trace behavior**
  - 50 steps, but tool mix in this run is narrower than earlier forensic runs (mostly deep-learning stack vs broader algorithm-optimization set).
  - Evidence has small numeric deltas (for example around `0.00054844`) with no decisive significance fields.
- **Paper content**
  - Abstract and results correctly stay inconclusive for all five hypotheses.
  - Tool narrative in Methods/Results is present and coherent with trace list.
- **What is actually happening**
  - Platform execution is stable (previous crash path fixed), but optimizer-ranking question still lacks strong statistical closure.

## Q5: Width vs depth under fixed compute
- **Trace behavior**
  - 50 steps; includes core training/analysis tools and some architecture comparison coverage.
  - Results snippet still shows fallback-like hypothesis phrasing (`Hypothesis N: ...`) in parts.
- **Paper content**
  - Abstract remains conservative and tied to inconclusive ledger.
  - Results contain structured evidence plus trace IDs.
- **What is actually happening**
  - Experiment loop runs end-to-end reliably.
  - Remaining issue is hypothesis wording quality and weak significance evidence, not execution failure.

---

## Cross-question diagnosis
1. **Execution reliability is now strong**: no session failed in this batch; paper pipeline produced artifacts for all.
2. **Scientific strictness is active**: evidence contract prevents false confirmations.
3. **Evidence depth is insufficient for confirmations**: metrics are present, but significance/effect-size fields are usually missing or non-decisive.
4. **Paper honesty improved**: abstracts and verdicts reflect inconclusive state and avoid unsupported claims.
5. **Prompt/planning quality still variable**: some hypotheses remain generic, reducing domain precision.

## Recommended next actions
1. Enforce at least one significance-capable tool call per hypothesis (`statistical_significance`, `bootstrap_confidence`, or baseline comparator).
2. Bias tool planning to question-specific tools (especially for ranking/comparison questions).
3. Add a hypothesis-quality validator to reject generic fallback text before experiment execution.
4. Add paper-level warning when all hypotheses are inconclusive despite high trace step counts.
