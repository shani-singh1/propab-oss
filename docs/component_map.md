# Propab Component Map

> Checklist 5 of `fixes.md`. Built **before** any code removal so that the impact
> of every change is predictable. Every entry lists: what it does, what calls it,
> what it calls, the test that covers it, and a status
> (`active` / `legacy` / `not-yet-wired` / `partially-wired` / `dead`).
>
> Status legend:
> - **active** — reachable from the production campaign path
>   (`POST /campaigns` → `run_campaign_loop` → Celery `run_sub_agent_async`).
> - **legacy** — still wired to an API route but superseded by the campaign path.
> - **partially-wired** — used in a narrow live hook but mostly standalone.
> - **not-yet-wired** — exists, registered/tested, but no live campaign call site.
> - **dead** — no callers, or only reachable through other dead code.

Baseline at time of writing: `pytest tests` → **444 passed**.

> **Post-CL1/CL2/CL3 update (this session): `pytest tests` → 459 passed.**
> This map was written before code changes. The statuses below are updated to
> reflect what CL1 (dead-code removal), CL2 (domain decoupling), and CL3
> (evidence binding at write time) actually changed. Items still carrying a
> "Checklist 2 concern" note are the **deferred** domain-string relocations; the
> lifetime-LWW concurrency work and service-boundary items are **deferred to
> Checklist 4** (they can only be validated on a running stack). Deferred items
> are called out explicitly rather than omitted.

---

## Production campaign path (the spine)

```
POST /campaigns  (services/api/app/routes/research.py:190)
  → ResearchCampaign built in memory
  → db_save_campaign  (Postgres research_campaigns)
  → BackgroundTasks.add_task(run_campaign_loop, ...)   [runs INSIDE the API process]
       → build_prior / measure_baseline
       → while not campaign.should_stop():
            frontier empty? → generate_seed_hypotheses OR run_campaign_synthesis_pass (Tier-2)
            _iter_campaign_pipelined_results:
              _maybe_run_campaign_synthesis (Tier-2 mid-wave)
              run_sub_agent_task.delay(...)  → Celery worker
                 → run_sub_agent_async  → verdict (materials/mandrake fast path OR run_verdict_pipeline)
              hypothesis_tree.update_node + belief_state bookkeeping
            breakthrough? / hypothesis cap? / time budget?
       → ingest_campaign (lifetime learning write)
       → write_paper_minimal
       → CAMPAIGN_COMPLETED event → SSE /stream/{id}
```

Two structural facts that Checklists 2 and 4 target:
1. The campaign loop runs as a **FastAPI BackgroundTask inside the API process**, not in the orchestrator service. **(Still true — Checklist 4.)**
2. ~~The worker verdict path branches on hardcoded domain names (`is_materials_campaign`, `is_mandrake_campaign`).~~ **RESOLVED (CL2):** the worker now resolves the domain via `propab.domain_modules.registry.resolve_domain_plugin` and dispatches through a `{domain_id → worker path}` table (`sub_agent_loop.py:1190-1219`). Core no longer inspects question text for domain keywords; plugins own `matches`.

---

## Orchestration

### `run_campaign_loop`
- **File:** `services/orchestrator/campaign_loop.py:1659`
- **Does:** Drives a campaign from intake → baseline → experiment waves → finalize/paper.
- **Called by:** `services/api/app/routes/research.py:280` (create), `:458` (resume). No other call sites.
- **Calls:** `parse_question`, `build_prior`, `load_lifetime_state`, `enrich_prior_from_lifetime`, `measure_baseline`, `generate_seed_hypotheses`, `run_campaign_synthesis_pass`, `_iter_campaign_pipelined_results`, `db_save_campaign`/`db_load_campaign`, `write_campaign_snapshot`, `ingest_campaign`, `write_paper_minimal`.
- **Test:** `tests/test_campaign_redesign.py`, `tests/test_campaign_recount.py` (loop bookkeeping); no full end-to-end loop test (integration gap — flagged).
- **Status:** **active**.

### `ResearchCampaign`
- **File:** `packages/propab-core/propab/campaign.py:185`
- **Does:** In-memory campaign state (tree, belief_state, budget, stop_reason, metrics).
- **Called by:** API routes, campaign loop, resume, scripts.
- **Calls:** `HypothesisTree`, `CampaignBeliefState`, `BreakthroughCriteria`, `normalize_accuracy_metric`.
- **Test:** `tests/test_campaign_recount.py`, `tests/test_campaign_accuracy_plausibility.py`.
- **Status:** **active**.
- **Note:** `belief_state` is persisted **directly** inside `breakthrough_criteria_json._campaign_meta` (`campaign_loop.py:536-552`), restored via `_apply_campaign_meta_from_db` (`:555-570`). `campaign_resume.backfill_belief_state_if_empty` is a fallback that rebuilds from synthesis events only when meta is empty.

### `HypothesisTree`
- **File:** `packages/propab-core/propab/hypothesis_tree.py`
- **Does:** Directed hypothesis tree; frontier of dispatchable nodes; grows via seeds (Tier-1) and synthesis (Tier-2).
- **Called by:** `campaign_loop.py` (`add_seeds`, `update_node`, `next_dispatch_candidate`, `set_scoring_context`), `campaign_synthesis.py` (`apply_synthesis_to_frontier`).
- **Test:** `tests/test_hypothesis_tree_dispatch.py`, `tests/test_hypothesis_tree_information_gain.py`, `tests/test_frontier_snapshot_histograms.py`.
- **Status:** **active**.
- **CL1/CL2 update:** dead `expand_tree_node`/`_expand_node_async` removed. `build_expand_prompt` + `_EXPAND_PROMPT_TEMPLATE` are `# KEPT:` (still used by scope-gate tests). `infer_domain_scope_template` **RESOLVED (CL2):** it now delegates to the `DomainPlugin` registry (`scope_template()` per plugin) instead of hardcoded domain templates.

### `campaign_synthesis.py`
- **File:** `packages/propab-core/propab/campaign_synthesis.py`
- **Does:** Tier-2 frontier refill. `run_campaign_synthesis_pass` (`:219`) composes a synthesis prompt, calls the LLM, updates beliefs, and adds deduped candidates to the frontier. `should_trigger_synthesis` (`:292`) gates mid-wave synthesis.
- **Called by:** `campaign_loop.py:1017` (mid-wave via `_maybe_run_campaign_synthesis`), `:1981` (frontier empty).
- **Calls:** `belief_state.apply_synthesis_beliefs`, `enrich_entry_with_scope`, `validate_scoped_claim`, `tree.add_seeds`/`add_to_frontier`.
- **Test:** `tests/test_campaign_synthesis_builder.py`, `tests/test_campaign_redesign.py`.
- **Status:** **active**.

### `BeliefState` / `CampaignBeliefState` / belief objects
- **File:** `packages/propab-core/propab/belief_state.py`
- **Does:** Tracks active/closed rival beliefs, synthesis bookkeeping (`results_since_last_synthesis`, `exhaustion_rounds`, `branch_exhausted`). `apply_synthesis_beliefs` (`:118`) admits beliefs after evidence-binding + falsifiability filtering.
- **Called by:** `campaign_synthesis.py`, `campaign_resume.py`, `campaign_loop.py`.
- **Calls:** `evidence_binding.filter_node_citations`, `evidence_binding.belief_falsifiable_in_dataset`.
- **Test:** covered indirectly via `tests/test_campaign_synthesis_builder.py`; **no dedicated `test_belief_state.py`** (flagged untested).
- **Status:** **active**.

---

## Hypothesis generation

### `generate_seed_hypotheses`
- **File:** `services/orchestrator/campaign_loop.py:1165`
- **Does:** Produces bootstrap/refill seeds; branches to anomaly seeds when `seed_source == "anomaly"`, else `generate_ranked_hypotheses`. Applies scope enrich/validate + dedup.
- **Called by:** `campaign_loop.py:1968` (empty tree), `:1999` (synthesis-disabled fallback).
- **Test:** `tests/test_seed_fallbacks.py`, `tests/test_seed_validation_suite.py` (via scripts).
- **Status:** **active**.

### `generate_ranked_hypotheses`
- **File:** `services/orchestrator/hypotheses.py:403`
- **Does:** LLM-ranked hypothesis generation with scope enrichment + relevance gate.
- **Called by:** `generate_seed_hypotheses` (`campaign_loop.py:1211`), `research_loop.py:430` (legacy), `seed_validation.py:247` (script).
- **Test:** `tests/test_hypothesis_ranking.py`, `tests/test_question_relevance_gate.py`.
- **Status:** **active**.
- **Note (Checklist 2 concern — DEFERRED):** `services/orchestrator/hypotheses.py:21-160` (`_domain_fallback_options`, `_scoped_contagion_text`) contains hardcoded domain-specific fallback seeds (contagion/network, number theory, resilience, community). Behavior-controlling → should move to per-domain plugin `seed_fallbacks()` methods. Not relocated this session (large seed block on the generation path; low correctness risk — affects fallback seed content only).

### Synthesis-based frontier refill (Tier 2)
- See `campaign_synthesis.py` above. This **replaces** per-node LLM expansion (Tier 1), which is dead (`campaign_loop.py:2133` comment).

---

## Verification

### `run_verdict_pipeline`
- **File:** `packages/propab-core/propab/verdict_pipeline.py:247`
- **Does:** Pure composed pipeline: `classify_verdict_stage` → `artifact_gate_stage` → `ood_gate_stage` → `scope_integrity_stage`.
- **Called by:** `services/worker/sub_agent_loop.py:2101` (generic think-act path only), `scripts/reaudit_campaign_verdicts.py`, `scripts/checkpoint_campaign_lofo.py`.
- **Test:** `tests/test_verdict_pipeline_composition.py` (exists per audit), `tests/test_sub_agent_verdict.py`.
- **Status:** **active** (generic path). Materials/mandrake fast paths still have their own classifiers, but domain dispatch to them now goes through the plugin registry (CL2), and the generic path's confirmation threshold base is read from `DomainPlugin.confirmation_criteria()` (`sub_agent_loop.py:2082-2094`, CL2) rather than a hardcoded constant.

### `classify_verdict` / `classify_verdict_stage`
- **File:** `packages/propab-core/propab/significance.py:126` (`classify_verdict`); `verdict_pipeline.py:102` (stage wrapper).
- **Does:** Maps evidence (`verified_true_steps`, p-value, effect size, n) to a verdict + confidence.
- **Called by:** `verdict_pipeline.classify_verdict_stage`. `sub_agent_loop.py:39` imports `classify_verdict` but **never calls it** (dead import — Checklist 1).
- **Test:** `tests/test_significance.py`, `tests/test_sub_agent_verdict.py`.
- **Status:** **active** (via pipeline).

### `artifact_gate_stage` / `run_artifact_gate`
- **File:** `verdict_pipeline.py:144` (stage); `artifact_verification.py:480` (`run_artifact_gate`).
- **Does:** Generates ranked artifact-explanation models and runs adversarial tests (label-shuffle LOFO, permutation) to decide whether a "confirmed" verdict survives.
- **Called by:** pipeline `:136`, materials path `sub_agent_loop.py:1024`, mandrake path via `apply_artifact_gate_override`, `domain_profiles/base.py:73`.
- **Test:** `tests/test_artifact_verification.py`.
- **Status:** **active**.
- **CL2 update:** the network/graph artifact vocabulary (`_NETWORK_MARKERS`) has been **relocated** onto `NetworkDiffusionPlugin.artifact_question_markers`; core reads it via the registry (`artifact_verification._network_markers()`), so core holds no per-domain artifact keywords.
- **Note (Checklist 2 concern — PARTIAL):** the adversarial *thresholds* (n≥80, p<0.01, lofo>p95, confidence 0.85/0.65/0.7/0.55) are still hardcoded in the gate. The per-domain `min_metric_steps_for_confirm` is now sourced from `confirmation_criteria()`; the remaining adversarial constants are not yet per-domain configurable (deferred).

### `ood_gate_stage` / `scope_integrity_stage`
- **File:** `verdict_pipeline.py:178` / `:227`; primitives in `scoped_claim.py` (`check_ood_evidence:415`, `apply_ood_gate_to_verdict:440`, `check_scope_executed_integrity:320`).
- **Does:** Downgrade confirmed → inconclusive when OOD evidence is missing or executed LOFO doesn't match the declared OOD scope.
- **Called by:** pipeline stages; also directly in materials/mandrake fast paths and post-pipeline in the generic path (`sub_agent_loop.py:2118-2127`).
- **Test:** `tests/test_scoped_claim.py`, `tests/test_scoped_tree_expansion.py`.
- **Status:** **active**.
- **RESOLVED (CL2):** `scoped_claim.infer_domain_scope_template` no longer hardcodes contagion/mandrake/materials templates — it iterates the `DomainPlugin` registry and uses each plugin's `scope_template()`/`matches_scope()`. Templates now live in the plugins (`MaterialsPlugin`, `MandrakePlugin`, `NetworkDiffusionPlugin`).

### Evidence Binding
- **File:** `packages/propab-core/propab/evidence_binding.py`
- **Does:** Checks that a citation's subject matches the claim's subject; filters unsupported citations/anomalies; falsifiability check for beliefs.
- **Called at WRITE time:**
  - Synthesis beliefs — `belief_state.py:156-181` (`filter_node_citations`, `belief_falsifiable_in_dataset`) via `apply_synthesis_beliefs`.
  - Mechanism induction — `anomaly_engine/mechanism_inducer.py:374` (`filter_mechanism_anomalies`).
  - Artifact gate — `artifact_verification.py:500` when `tree_nodes` passed (mostly audit/preflight).
- **Post-hoc only:** `scripts/audit_evidence_binding.py`, `integrations/astabench/audit.py`.
- **Test:** `tests/test_evidence_binding.py`.
- **Status (CL3 — DONE):** **wired at write time.** Confirmed the real synthesis pass filters citations *before* persistence: `campaign_synthesis.apply_synthesis_to_frontier` passes `tree_nodes` + `metrics` to `belief_state.apply_synthesis_beliefs`, which calls `filter_node_citations` **before** constructing/persisting the `BeliefObject` (`belief_state.py:179-186`). Proven by `test_synthesis_pass_rejects_fabricated_citation_before_persist` (drives the real synthesis pass, not the check in isolation). Also genericized `belief_falsifiable_in_dataset` (removed the `98`/`56`/`biophysical set` Mandrake-specific tokens; concept is domain-agnostic).
- **Documented boundary:** per-hypothesis finding DB writes in `sub_agent_loop` are the worker's raw verdict output, not citations of one node by another; Evidence Binding applies to *citing objects* (beliefs, mechanisms), which are all filtered at write time. There is no separate per-finding citation write that bypasses binding.

### Worker think-act loop
- **File:** `services/worker/sub_agent_loop.py:1125` (`run_sub_agent_async`), `services/worker/think_act.py`.
- **Does:** Runs the tool loop for one hypothesis, builds evidence, produces a verdict.
- **Called by:** `services/worker/runner.py` ← Celery `run_sub_agent_task` (`tasks.py:7`).
- **Test:** `tests/test_think_act.py`, `tests/test_sub_agent_verdict.py`, `tests/test_sub_agent_plan.py`, `tests/test_sub_agent_metric_normalize.py`.
- **Status:** **active**.
- **CL2 update:** domain **detection** (`:1186-1209`) now goes through `resolve_domain_plugin` + a `{domain_id → worker path}` table; the tool-cluster selection no longer hardcodes `"mandrake"`/`"materials"` cluster names (`get_cluster_with_significance(domain)`). The `_materials_verification_path`/`_mandrake_verification_path` methods still hold the domain-specific verification code, but they are reached only via the plugin dispatch table, not via inline keyword checks — they are the worker-side half of each `DomainPlugin`.

---

## Domain layer

### Domain adapters — `packages/propab-core/propab/domain_adapters/`
| File | Role | Status |
|---|---|---|
| `__init__.py` | Exports mandrake only | active |
| `mandrake_adapter.py` | RT-family LOFO on mandrake-data; `is_mandrake_campaign`, `classify_mandrake_verdict`, shared LOFO helpers | active |
| `materials_adapter.py` | Matbench dielectric LOFO; `is_materials_campaign`, `classify_materials_verdict`, `run_materials_lofo` | active |
| `materials_featurizer.py` | Structure → descriptors | active |
| `materials_element_data.py` | Element property table | active |
| `materials_crystal_system.py` | Space group → crystal system | active |
| `materials_mp_bandgap.py` | MP bandgap cache via matbench_mp_gap | active |
| `materials_frame_cache.py` | Dielectric frame disk cache | active |
| `perovskites_adapter.py` | Matbench perovskites A-site LOFO (preflight) | active (preflight only) |
| `perovskites_a_site.py` | A-site chemistry grouping | active (preflight only) |

- **Tests:** `tests/test_mandrake_verification.py`, `tests/test_materials_verification.py`.
- **Note (git):** the materials/perovskites files are currently untracked in git but present on disk and imported by live code.

### Domain profiles — `packages/propab-core/propab/domain_profiles/`
| File | Provides | Wired into `run_artifact_gate`? | Live sub-agent path? |
|---|---|---|---|
| `base.py` | `DomainProfile` dataclass + `run_artifact_gate` delegate | — | — |
| `registry.py` | `resolve_domain_profile` (tag → payload → bucket → question markers) | called at `artifact_verification.py:489` | — |
| `enzyme_kinetics.py` | Enzyme family LOFO profile | yes (via tag) | yes (`EnzymeKineticsPlugin` LOFO verifier) |
| `materials.py` | Crystal-system profile + materials artifact models | yes | yes (via `_materials_verification_path`) |
| `graph_invariants.py` | SNAP graph-family profile | yes (via tag) | yes (`GraphInvariantsPlugin` cross-family verifier) |
| `econometrics.py` | Panel FE / within-group R² profile (DiscoveryBench) | yes (via tag) | yes (artifact gate for panel FE evidence) |
- **Test:** `tests/test_domain_profiles.py`, `tests/test_econometrics_profile.py`, `tests/test_enzyme_kinetics_plugin.py`, `tests/test_graph_invariants_plugin.py`.
- **Status:** materials **active**; enzyme/graph/econometrics **active** (full `DomainPlugin` + preflight + routing corpus as of `614d258`).
- **Relationship (CL2 update):** `domain_profiles` configure the **artifact gate** (grouping + artifact models); `domain_adapters` run the **experiment**. Both are fronted by the `DomainPlugin` layer (`propab/domain_modules/`): `DomainPlugin.domain_profile()` links to the profile, `confirmation_criteria()` reads thresholds from it, and `artifact_models()` delegates to `profile.generate_artifact_models`. Plugins: `MaterialsPlugin`, `MandrakePlugin`, `EnzymeKineticsPlugin`, `GraphInvariantsPlugin`, `GenomicsPlugin`, `NetworkDiffusionPlugin` (scope-only), `MathCombinatoricsPlugin`.

### `run_artifact_gate` dispatch
- **File:** `artifact_verification.py:480-516`
- **Does:** Resolves a domain profile from the evidence context/question; if found, delegates to `profile.run_artifact_gate`; else uses generic artifact models.
- **Status:** **active**.

### Tools registry
- **File:** `packages/propab-core/propab/tools/registry.py`
- **Does:** Auto-discovers tool modules with a `TOOL_SPEC` dict; clusters by each tool's `domain` field; `get_cluster(domain)` / `get_cluster_with_significance(domain)`.
- **Called by:** `sub_agent_loop.py` tool selection. **RESOLVED (CL2):** the previously hardcoded `"mandrake"`/`"materials"` cluster-name branches were removed; the generic path uses `get_cluster_with_significance(domain)` with the resolved domain.
- **Test:** `tests/test_tool_registry.py`, `tests/test_tool_selection.py`.
- **Status:** **active** (registry is discovery-based).

---

## Lifetime learning

Default backend: JSON files under `{propab_data_dir}/lifetime_knowledge/` (last-writer-wins).
When `lifetime_store_backend=postgres`, `propab/lifetime_postgres.py` upserts per entity
(T1-001 migration `20260704130000_lifetime_knowledge_postgres`).

> **T1-001 (in progress):** Postgres upsert path wired for `KnowledgeGraph`,
> `MetaScienceLedger`, and `PolicyFitnessLedger`. JSON remains default for tests;
> set `LIFETIME_STORE_BACKEND=postgres` and run migration on the stack to enable.

### `lifetime_postgres.py`
- **File:** `packages/propab-core/propab/lifetime_postgres.py`
- **Does:** Per-claim/theory/seed upserts; replaces JSON LWW when backend is postgres.
- **Test:** `tests/test_lifetime_postgres_concurrent.py` (skips without Postgres).

### `KnowledgeGraph`
- **File:** `packages/propab-core/propab/knowledge_graph.py`
- **Write:** `ingest_campaign` → `graph.save()` (`lifetime_knowledge.py:303`).
- **Read:** `load_lifetime_state` → `KnowledgeGraph.load()` (`lifetime_knowledge.py:53`).
- **Test:** `tests/test_lifetime_knowledge.py`.
- **Status:** **active** (LWW concurrency risk — Checklist 3 note).

### `MetaScienceLedger`
- **File:** `packages/propab-core/propab/meta_science.py`
- **Write/Read:** `ingest_campaign` save (`:305`) / `load_lifetime_state` (`:54`).
- **Test:** `tests/test_lifetime_knowledge.py`.
- **Status:** **active** (LWW risk).

### `PolicyFitnessLedger`
- **File:** `packages/propab-core/propab/policy_fitness_ledger.py`
- **Write:** `ingest_campaign` (`:306`). **Read:** at campaign end for the policy analyst (`campaign_loop.py:2242`).
- **Test:** covered via `tests/test_policy_governance.py`.
- **Status:** **active** (LWW risk).

### `lifetime_context_for_seeds`
- **File:** `services/orchestrator/lifetime_knowledge.py:125`
- **Does:** Builds lifetime context string injected into seed prompts.
- **Called by:** `campaign_loop.py:1839`, output reaches `generate_seed_hypotheses` prompt (`:1968-1970`, `:1999-2001`).
- **Test:** `tests/test_lifetime_knowledge.py`.
- **Status:** **active**.

### `ingest_campaign`
- **File:** `services/orchestrator/lifetime_knowledge.py:140`
- **Does:** Writes claims/mechanisms/failures/theories to the graph, an observation to the meta ledger, fitness records, and a candidate policy proposal.
- **Called by:** `run_campaign_loop:2272` (campaign end). Also `scripts/ingest_campaign_lifetime.py` (backfill).
- **Test:** `tests/test_lifetime_knowledge.py`.
- **Status:** **active**.

---

## Infrastructure

### API routes — `services/api/app/routes/`
| Route | File | Status |
|---|---|---|
| `POST /campaigns`, `POST /campaigns/{id}/resume`, `GET /campaigns/{id}` | `research.py` | **active** (production) |
| `POST /research`, session endpoints | `research.py`, `sessions.py` | **legacy** (old multi-round session loop) |
| `GET /stream/{session_id}` (SSE) | `stream.py` | **active** |
| `GET /health` | `health.py` | active |
| `/tools/*` | `tools.py` | active (introspection) |

### Celery tasks
- **File:** `services/worker/celery_app.py`, `services/worker/tasks.py`.
- **Task:** `run_sub_agent_task` (`tasks.py:7`), dispatched via `.delay()` from `campaign_loop.py:962` (baseline), `:1336` (hypothesis), `research_loop.py:199` (legacy).
- **Status:** **active**. Celery runs **individual hypotheses only** — the campaign loop itself is not a Celery task.

### `db_save_campaign` / `db_load_campaign`
- **File:** `services/orchestrator/campaign_loop.py:575` / `:751` (NOT in `propab/db.py`).
- **Does:** UPSERT/SELECT `research_campaigns`; belief_state stored in `breakthrough_criteria_json._campaign_meta`.
- **Called by:** API routes (`research.py`), campaign loop, `sub_agent_loop.py:743` (lazy cross-service import — Checklist 4 concern).
- **Test:** covered indirectly; no dedicated persistence round-trip test (flagged).
- **Status:** **active**. Location in a service module (imported by API + worker) is a Checklist 4 target — should move to `propab-core`.

### `propab/db.py`
- **File:** `packages/propab-core/propab/db.py`
- **Does:** Engine/session/redis factories and `insert_event`.
- **Status:** **active**.

### Redis pub/sub
- **Publisher:** `EventEmitter.emit` (`events.py:38-42`) → channel `propab:{session_id}` + `insert_event`.
- **Subscriber:** `stream.py:14-31` (SSE).
- **Also:** peer findings use Redis lists `propab:peer:{hypothesis_id}` (`services/worker/peer_findings.py`).
- **Test:** `tests/test_peer_findings.py`.
- **Status:** **active**.

### Postgres schema (Alembic: 20260424 → 20260425 → 20260501 → 20260530 → 20260531)
| Table | Purpose |
|---|---|
| `research_sessions` | Session/campaign top-level row (question, status, stage, prior_json) |
| `research_campaigns` | Campaign state: breakthrough criteria (+ `_campaign_meta`), hypothesis_tree_json, metrics, budget |
| `hypotheses` | Per-session hypotheses (text, scores, verdict, lineage, tool_trace_id) |
| `experiment_steps` | Think-act step trace (+ significance_json) |
| `tool_calls` | Tool invocations (+ significance metadata) |
| `llm_calls` | LLM prompt/response audit |
| `events` | Append-only event log for SSE replay |
| `research_rounds` | Legacy multi-round session tracking |
| `session_checkpoints`, `session_budgets`, `agent_memory`, `kb_findings` | Legacy long-running-agent tables |
| `literature_query_cache` | Query-hash → paper_ids cache |

---

## Paper writing

### `write_paper_minimal` / `paper_compiler.py`
- **File:** `services/orchestrator/paper.py:88` (`write_paper_minimal`); `packages/propab-core/propab/paper_compiler.py` (compile library).
- **Does:** Structures campaign findings into a paper at campaign end (or salvage on error).
- **Called by:** `campaign_loop.py:2289` (end), `:1641` (salvage), `research_loop.py:366/615` (legacy).
- **Test:** `tests/test_paper_compiler_tables.py`, `tests/test_paper_gate.py`, `tests/test_paper_sections.py`, `tests/test_paper_guard.py`, `tests/test_paper_effective_verdict_verified.py`.
- **Status:** **active**.

---

## Standalone / research subsystems (not on the live campaign path)

| Subsystem | Files | Live hook | Status |
|---|---|---|---|
| `anomaly_engine/` | 12 files | Only via `anomaly_seeds.py` when `seed_source == "anomaly"` (opt-in) | **partially-wired** (opt-in) |
| `layer05/` | 38 files | `SimulationFitnessLedger` read at campaign end for policy analyst (`campaign_loop.py:2243`) | **partially-wired** (2 symbols) |
| `operator_credit/` | 20 files | none | **not-yet-wired** (script/test only) |
| `benchmarks/graph_contagion_benchmark.py` | 1 | none | script/test only |

**Domain-string decoupling status (CL2 — updated 2026-07-05, T3-003 complete in `614d258`):**
- `finding_audit.py` — **relocated** to `domain_modules/mandrake/finding_audit.py`; `propab/finding_audit.py` is a re-export shim. Used by `scripts/inspect_confirmed_findings.py` + `tests/test_finding_audit.py`.
- `campaign_resume.py` `CONTRARIAN_*` — **relocated** to `MandrakePlugin.apply_contrarian_belief_reset()`; `campaign_resume.py` re-exports constants for API compatibility.
- `research_quality._THEME_RULES` — **relocated** to per-domain `theme_rules` on `DomainPlugin`; core merges via `all_theme_rules()` / `all_theme_fallbacks()`.
- `anomaly_engine/` (`competing_mechanisms.py`, `mechanism_inducer.py`) — opt-in Mandrake seed path; domain strings not relocated (single-domain, opt-in).
- `theory_objects.py` (`_contagion_theory` naming) — offline lifetime aggregation only; deferred.
- `services/orchestrator/hypotheses.py` `_domain_fallback_options` — hardcoded fallback seeds; deferred (low risk).
- `policy_buckets.py` / `services/orchestrator/question_domain.py` — infrastructure taxonomies by design.

---

## Dead code inventory (Checklist 1 — DONE)

| Item | File:line | Evidence | CL1 disposition |
|---|---|---|---|
| `expand_tree_node` | `campaign_loop.py:1032` | only caller is dead `_expand_node_async` | **removed** |
| `_expand_node_async` | `campaign_loop.py:1100` | no callers anywhere | **removed** |
| `build_expand_prompt` + `_EXPAND_PROMPT_TEMPLATE` | `hypothesis_tree.py:45,454` | reachable only via dead `expand_tree_node`; else tests/scripts | **KEPT** (`# KEPT:` — still used by scope-gate tests) |
| `campaign_expand_on_confirmed/refuted/inconclusive` | `config.py:109-111` | never read in code | **removed** |
| `campaign_expand_use_interpreter` | — | phantom; only in stale script/artifact notes | n/a (no code) |
| `build_failure_interpret_prompt` | — | removed; broken ref in `scripts/interpreter_bias_bench.py:334` | n/a (already gone) |
| `classify_verdict` import | `sub_agent_loop.py:39` | imported, never called | **removed** |
| `validate_resume_readiness` import | `campaign_loop.py:56` | imported, unused in loop | **removed** |
| `STOP_REASON_FRONTIER_EXHAUSTED` | `campaign.py:173` | defined, never assigned (kept for enum completeness) | kept (enum completeness) |

---

## Untested / integration gaps (feeds "What production-ready means")

- No end-to-end `run_campaign_loop` test (loop is only exercised piecemeal).
- No dedicated `test_belief_state.py`.
- No `db_save_campaign`/`db_load_campaign` round-trip persistence test.
- `domain_profiles` covered by plugin tests with synthetic adapters (enzyme/graph use BRENDA/SNAP-style synthetic subsets).
- `stop_reason` values are meaningful enums already (`campaign.py:169-180`) but `budget_exhausted` remains the `status` string — Observability item.
