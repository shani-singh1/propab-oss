# Propab latent-bug audit — maintained ledger

The living record of hidden, semantically-corrupting defects (the `val_accuracy` class:
silent, invisible to tests, corrupts results or honesty). We loop: audit → log here →
assign fix agents → verify (read diff + re-run suite) → mark fixed → re-audit.

Origin: after the `val_accuracy` bug (a math campaign scored against an ML baseline so no
result could register) survived a full codebase map, we ran 4 parallel deep audits
(campaign/verdict/baseline · domain plugins · honesty gates/worker · API/config/generation).
They found the defects below. Status legend: 🔴 open · 🟡 fixing (agent assigned) · 🟢 fixed+verified.

## Iteration status
- **All items below (C1–C2, H1–H5, M1–M6, L1) are 🟢 FIXED + verified + merged** to `main`
  via PR #9 (each with a real regression test that fails-before/passes-after; suite green).
- **New findings from this iteration:**
  - **N1 🔴 (medium, reproducibility):** the time-budgeted cap search (added for the timeout
    fix — `CAP_GREEDY_TIME_BUDGET`/`CAP_BB_TIME_BUDGET`, best-so-far on budget) makes a
    "deterministic under fixed seed" cap result NONDETERMINISTIC under CPU load (observed 20
    vs 18 in a loaded worktree). A `best_so_far` cap cut by wall-clock is inherently
    run-dependent. Fix: make the determinism guarantee wall-clock-independent (iteration/
    node-count budget, or a generous budget in the determinism path) so a claimed cap size is
    reproducible. Matters for the honesty/reproducibility story.
  - **N2 🟢 (frontend crash, FIXED):** the merged right-panel virtualizer infinite-looped
    (`Maximum update depth exceeded`, crashed WorkersPanel) — a new ref callback every render
    + an unconditional ResizeObserver state update. A GREEN BUILD MISSED IT; only live browser
    verification caught it. Fixed in PR #11. Lesson: frontend changes need live-render
    verification, not just `tsc`/build.

## CRITICAL

### C1 🟡 Biology domains can never confirm — permutation p-value is broken
`genomics/verifier.py:71`, `enzyme_kinetics/verifier.py:38`. `np.mean([1 for n in nulls if
n>=obs])` = the mean of an all-ones list = **always 1.0 (or nan)**, never a fraction. So
`verified_true_steps` is always 0, the "refuted" branch fires (`p>0.5`) on a REAL signal, and
the only escape (observed beats all nulls → p=nan) lands in inconclusive — verdicts inverted
across the whole signal range. Every genomics/enzyme finding is corrupted; a real discovery is
reported as a refutation. Invisible to tests (tests only assert the key exists). Fix:
`np.mean(np.asarray(nulls) >= observed)`. → agent FIX-BIO.

### C2 🟡 Plugin-path false-confirm hole — gate downgrade lost on emit/gate exception
`services/worker/sub_agent_loop.py:~1310-1331`. The artifact-gate downgrade is EMITTED before
it's APPLIED, inside `try/except: pass`; the emit (redis+DB) fires only when rejecting a
confirm. If it raises, the downgrade assignment never runs → a rejected result is published as
`confirmed`. Also fails open if the gate itself raises. The generic path has no such try/except
(fails closed). Fix: apply the downgrade before the emit; fail CLOSED (→ inconclusive) on any
exception. → agent FIX-HONESTY.

## HIGH

### H1 🟡 `min_confirmed_findings` breakthrough path is dead — math can't declare a discovery
`campaign.py:80-83` reads `confirmed_nodes`; the producer (`campaign_loop.py:~2320`) never
injects it → always 0. A baseline=0 (math/verification) campaign that confirms 5+ real
discoveries still can't declare BREAKTHROUGH (the other sub-criterion also fails: reps=1 for
root/seed nodes). Fix: inject `"confirmed_nodes": campaign.total_confirmed`. → agent FIX-CAMPAIGN.

### H2 🟡 Failed ML baseline → false breakthrough frame
`measure_baseline` returns 0.0 on ANY failure; `is_breakthrough` treats baseline≈0 as
"verification campaign" and switches to confidence+replications, never comparing accuracy. An
ML run with zero real gain can be declared a breakthrough. Fix: distinguish measured-0 from
failed/absent (None); refuse the verification frame for is_ml campaigns whose baseline failed.
→ agent FIX-CAMPAIGN.

### H3 🟡 coding_theory mis-scored as ML (val_accuracy replicated)
`coding_theory/plugin.py` has no `objective_spec` → keeps `val_accuracy` → `_is_ml_campaign`
matches "accuracy" → trains a meaningless MLP baseline; a code beating the best-known bound can
never register. Fix: add `objective_spec(is_ml=False, metric=code_minimum_distance,
baseline_kind=best_known)`. → agent FIX-OBJSPEC.

### H4 🟡 materials mis-scored as ML
`materials/plugin.py` (a leave-one-crystal-system-out lofo_r2 holdout domain) has no
`objective_spec` → same ML mis-score. Fix: add `objective_spec(is_ml=False, metric=lofo_r2,
baseline_kind=measured)`. → agent FIX-OBJSPEC.

### H5 🟡 Campaign resume silently no-ops
`research.py:~554-559`. An empty-body `POST /campaigns/{id}/resume` never rebases
`started_at`, so the wall-clock budget is already exhausted → `should_stop()` True on the first
iteration → zero new hypotheses run, presented as a normal `budget_exhausted` completion. Fix:
always rebase the budget clock on resume. → agent FIX-API.

## MEDIUM

### M1 🟡 `_is_ml_campaign` still keyword-fallback (structural root of the val_accuracy class)
`campaign_loop.py:~839-852` only short-circuits when a plugin asserts `is_ml is False`; a
None-returning domain plugin falls to `_ML_QUESTION_TOKENS` ("network"/"error"/"regression"/
"classification"). The per-domain objective_spec additions are patches. Fix: INVERT the default
— a domain-owned campaign is non-ML unless the plugin explicitly asserts `is_ml is True`. →
agent FIX-CAMPAIGN.

### M2 🟡 lower_is_better treats a real 0.0 optimum as "unset"; `x or y` drops 0.0 metrics
`campaign.py:~89,290,305-306`. A lower-is-better campaign that reaches metric 0.0 regresses to a
worse "best"; a real `metric_value` of 0.0 is discarded as falsy. Fix: None sentinel + explicit
`is None` checks. → agent FIX-CAMPAIGN.

### M3 🟡 OOD + scope-integrity gates skipped on the entire plugin path
`sub_agent_loop.py::_plugin_verification_path` runs only the artifact gate — not OOD/
scope-integrity (which the generic/materials/mandrake paths run). A scope-inflated confirm on a
lofo/statistical plugin domain isn't caught here. Fix: for lofo/statistical (non-deterministic)
plugin evidence, also chain OOD + scope-integrity. → agent FIX-HONESTY.

### M4 🟡 Bare `deterministic:True` flag bypasses the artifact/null gate
`verdict_pipeline.py:~56-61`. `classify_evidence_type` trusts a plugin-set `deterministic:True`
verbatim; today guarded by `verified_true_steps`, but any future plugin stamping it on a
statistical result gets an automatic bypass. Fix: only honor the flag with a recognized proof
`verification_method` in `_PROOF_METHODS`. → agent FIX-HONESTY.

### M5 🟡 `config.llm_model` accepted but silently ignored
`research.py:~89` — echoed into an event only; the run hardcodes `settings.llm_model`. Fix:
thread it through, or remove the field. → agent FIX-API.

### M6 🟡 Injected `literature_prior` without `evidence_status` stamped READY
`campaign_loop.py:~1950-1960`. An empty injected prior is labeled READY and fed to generation as
real evidence. Fix: derive evidence_status from content. → agent FIX-API.

## LOW

### L1 🟡 Domain metric override inherits ML default direction
`research.py:~337-339`. A domain objective_spec with a metric but no `direction` inherits
"higher_is_better"; a minimization metric is scored backwards. Fix: require/validate direction
when adopting a domain metric. → agent FIX-API.

## Verified NOT bugs (do not re-audit)
- `classify_evidence_type` deterministic gate-bypass hole is CLOSED (requires explicit
  `deterministic:True` + whitelisted proof method, or falls to "unknown" and is gated).
- `campaign_db` save/load round-trips criteria/stop_reason without silent reset (`_db_float`
  avoids the zero-drop trap).
- `llm.py` fails loud on misconfig; no parse-failure-as-real-result. `hypotheses.py` never
  fabricates templates on parse failure (returns []).
- `literature_client.py` fallbacks are honest + labeled; empty corpus → INSUFFICIENT_EVIDENCE.
- `_survives_permutation`/`_survives_label_shuffle_lofo` fail closed; scope/OOD checks fail
  closed on missing data. graph_invariants/network_diffusion/mandrake objective_spec + p-values
  are correct. registry routing is score-based with logged near-ties.
- math_combinatorics objective_spec metric label `extremal_witness_ratio` not being emitted is
  INTENDED (display label; per-object best-known lives in the verifier).

## Loop notes
- Every fix agent must add a REAL test that fails before / passes after (these bugs were all
  invisible to existing tests — that's the defining property).
- Next audit passes to run after this wave lands: (a) the worker execution/sandbox path
  (code_timeout, partial results, resource limits); (b) literature service internals; (c) the
  event-emission contract vs the frontend's assumptions; (d) cross-domain metric normalization.
