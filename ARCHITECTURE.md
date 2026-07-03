# Propab — Architecture

> **Current-state architecture.** This document describes the system as it is
> actually wired today, using `docs/component_map.md` as the source of truth. It
> is deliberately not aspirational: every component listed here has a live call
> site on the campaign path, or is explicitly marked as partially-wired /
> opt-in. When code and this document disagree, the code (and the component map)
> win — update this document, not the other way around.

Companion documents:
- `propab_ownership_contracts.md` — what each component owns, never owns, its
  input/output contract, and its **health metric** (the number that tells you
  whether it is doing its job).
- `docs/component_map.md` — per-symbol map with file:line, callers, callees,
  covering test, and wiring status.
- `TOOLS.md` — the sub-agent tool surface.

---

## 1. What Propab is

Propab is an autonomous, **literature-grounded research campaign engine**. You
give it a scientific question and a domain; it runs a long-lived campaign that:

1. Builds a **structured literature prior** (established facts, open gaps,
   contradictions, dead ends) to seed the search.
2. Generates candidate **hypotheses** with explicit scope (population, claimed
   generalization, out-of-distribution test, expected failure mode).
3. Dispatches each hypothesis to a **worker** that runs a domain-appropriate
   experiment and returns raw evidence.
4. Runs a **verification pipeline** that decides confirmed / refuted /
   inconclusive, with an **artifact gate** (label-shuffle / permutation nulls)
   and an **out-of-distribution / scope-integrity gate** so a "confirmed"
   result actually generalizes.
5. **Synthesizes** completed results into a small set of rival beliefs, binds
   evidence to those beliefs at write time (no fabricated citations), and
   refills the frontier with the next most discriminating experiments.
6. Persists resumable state to Postgres after every round, records a non-null
   **stop reason** when it ends, and compiles a paper from the actual trace.

The design constraint that shapes everything: **nothing happens silently, and
every component has one measurable health metric.** Debugging starts with "which
component's number is out of range," not "Propab failed."

---

## 2. Service topology

Propab runs as four application services plus infrastructure, wired by
`docker-compose.yml`:

```
                    ┌─────────────┐      HTTP POST /internal/campaign
  browser ────────► │   api       │ ───────────────────────────────┐
   (SSE)            │  (FastAPI)  │                                 │
        ▲           └─────┬───────┘                                 ▼
        │                 │ db_save_campaign             ┌──────────────────┐
        │                 ▼                              │  orchestrator    │
        │           ┌───────────┐                        │  (FastAPI)       │
        └───────────│  redis    │◄───── events ──────────│  run_campaign_   │
        events/SSE  │ pub/sub   │                        │  loop            │
                    └───────────┘                        └───────┬──────────┘
                          ▲                                      │ Celery .delay()
                          │ events                               ▼
                    ┌───────────┐                        ┌──────────────────┐
                    │ postgres  │◄──── state / trace ─────│  worker (Celery) │
                    │           │                         │  run_sub_agent   │
                    └───────────┘                         └──────────────────┘

  infra: qdrant (vector search) · minio (paper/figure objects) · migrate (alembic)
```

- **`api`** — HTTP entrypoint. Creates/loads campaigns, persists the initial
  row, and **delegates execution** to the orchestrator via
  `POST /internal/campaign` (authenticated with `ORCHESTRATOR_INTERNAL_TOKEN`).
  If `ORCHESTRATOR_URL` is unset, it falls back to running the loop as an
  in-process `BackgroundTask` (single-node dev mode).
- **`orchestrator`** — owns the campaign lifecycle. `run_campaign_loop` drives
  intake → literature prior → baseline → experiment waves → synthesis →
  finalize → paper. This is where a campaign lives; **restarting the API does
  not kill an in-flight campaign** (the reason for the API↔orchestrator split).
- **`worker`** — a Celery worker pool. Each `run_sub_agent_task` executes **one
  hypothesis**: a think-act tool loop that produces raw evidence and a verdict.
  Configured for worker-loss resilience (`task_acks_late`,
  `task_reject_on_worker_lost`, broker `visibility_timeout`) so a crashed worker
  requeues its hypothesis rather than losing it.
- **`frontend`** — Vite/React campaign dashboard (event stream, hypothesis tree,
  findings, health overview).
- **Infrastructure** — Postgres (state + append-only event log), Redis (pub/sub
  for SSE + peer-finding lists + Celery broker), Qdrant (literature chunk
  vectors), MinIO (paper/figure objects), and a one-shot `migrate` service that
  runs `alembic upgrade head` before app services start.

---

## 3. The campaign path (the spine)

```
POST /campaigns                     (services/api/app/routes/research.py)
  → ResearchCampaign built in memory
  → db_save_campaign                (propab.campaign_db → Postgres research_campaigns)
  → _dispatch_campaign
       → POST orchestrator /internal/campaign   (or in-process BackgroundTask fallback)

run_campaign_loop                   (services/orchestrator/campaign_loop.py)
  → emit CAMPAIGN_STARTED
  → _enforce_domain_preflight       ← NEW gate: refuse launch if domain is underpowered
  → build_prior / measure_baseline
  → while not campaign.should_stop():
       frontier empty? → generate_seed_hypotheses OR run_campaign_synthesis_pass (Tier-2)
       _iter_campaign_pipelined_results:
         _maybe_run_campaign_synthesis           (Tier-2 mid-wave refill + health logging)
         run_sub_agent_task.delay(...)           → Celery worker
            → run_sub_agent_async → verdict (domain fast path OR run_verdict_pipeline)
         hypothesis_tree.update_node + belief_state bookkeeping
       breakthrough? / hypothesis cap? / time budget?
  → finalize_stop(<enum stop reason>)            ← always non-null
  → log_campaign_end_health / log_campaign_audit ← NEW per-campaign metrics
  → ingest_campaign (lifetime learning write)
  → write_paper_minimal
  → CAMPAIGN_COMPLETED / CAMPAIGN_BUDGET_EXHAUSTED → SSE /stream/{id}
```

Campaign persistence (`db_save_campaign` / `db_load_campaign` /
`db_load_session_events_tail`) lives in **`propab.campaign_db`** (core), so the
API, orchestrator, worker, and scripts all import it from one place instead of
reaching across service boundaries.

---

## 4. Component contracts and health metrics

Every major component has a one-line contract and exactly one health metric
(full text in `propab_ownership_contracts.md`). Propab logs these to Postgres so
they are queryable per campaign:

| Component | Health metric | Table |
|---|---|---|
| Literature Layer | citation verification rate | `campaign_literature_priors` |
| Hypothesis Generator | duplicate rate per round | `campaign_synthesis_events` |
| Campaign Synthesis | belief citation integrity, belief stability | `campaign_synthesis_events` |
| Worker / Executor | experiment success rate | `research_campaigns` |
| Verification Pipeline | artifact-gate precision | `campaign_audit_results` |
| Campaign Manager | worker utilization, stop-reason accuracy | `research_campaigns` |
| Evidence Binding | binding rejection rate | `campaign_synthesis_events` |

See §11 for how and when each is computed and logged.

---

## 5. Orchestration

### `run_campaign_loop` — `services/orchestrator/campaign_loop.py`
Drives a campaign end to end. Reachable only from the API routes (create /
resume) and the orchestrator's `/internal/campaign` endpoint. Checkpoints to
Postgres after every synthesis round; resumable after any restart.

### `ResearchCampaign` — `packages/propab-core/propab/campaign.py`
In-memory campaign state: hypothesis tree, belief state, budget, metrics, and a
non-null `stop_reason`. `belief_state` is persisted inside
`breakthrough_criteria_json._campaign_meta` and restored on load.
`finalize_stop(reason)` records the terminal stop reason (enum). Stop reasons
include `BREAKTHROUGH`, `HYPOTHESIS_CAP_REACHED`, `TIME_BUDGET_EXHAUSTED`,
`ALL_BRANCHES_EXHAUSTED`, `NO_DISPATCHABLE_NODES`, `FRONTIER_REFILL_FAILED`,
`SALVAGED_AFTER_ERROR`, `FATAL_ERROR`, and **`DOMAIN_PREFLIGHT_FAILED`**.

### `HypothesisTree` — `packages/propab-core/propab/hypothesis_tree.py`
Directed tree of hypotheses with a frontier of dispatchable nodes. Grows from
seeds (Tier-1) and synthesis candidates (Tier-2). `update_node` records verdicts
and maintains the confirmed list.

### `campaign_synthesis.py` — `packages/propab-core/propab/campaign_synthesis.py`
Tier-2 frontier refill. `run_campaign_synthesis_pass` composes a synthesis
prompt, updates beliefs (evidence-binding + falsifiability + rival-cap filtered
at write time), and adds deduplicated candidates. It also **logs per-round
health metrics** (`log_synthesis_health`) when a `session_factory` is provided.

### `CampaignBeliefState` — `packages/propab-core/propab/belief_state.py`
Tracks ≤3 active rival beliefs plus closed beliefs. `apply_synthesis_beliefs`
admits beliefs only after `evidence_binding.filter_node_citations` and
`belief_falsifiable_in_dataset` — this is Evidence Binding at write time.

---

## 6. Hypothesis generation

`generate_seed_hypotheses` (`campaign_loop.py`) produces bootstrap/refill seeds
— from the mechanism inducer when `seed_source == "anomaly"`, otherwise
`generate_ranked_hypotheses` (`services/orchestrator/hypotheses.py`), which does
LLM-ranked generation with scope enrichment and a question-relevance gate. Every
seed passes scope validation and duplicate rejection before it enters the tree.

---

## 7. Verification

The worker produces raw evidence; verification decides what it means. Core never
hardcodes a domain — it asks the resolved `DomainPlugin`.

### `run_verdict_pipeline` — `packages/propab-core/propab/verdict_pipeline.py`
A pure composed pipeline:
`classify_verdict_stage → artifact_gate_stage → ood_gate_stage → scope_integrity_stage`.
Materials/mandrake have dedicated fast-path classifiers, but domain dispatch to
them goes through the plugin registry, and the generic path reads its
confirmation threshold from `DomainPlugin.confirmation_criteria()`.

### Artifact gate — `artifact_verification.py` (`run_artifact_gate`)
Generates ranked artifact-explanation models and runs adversarial tests
(label-shuffle LOFO, permutation nulls) to decide whether a "confirmed" verdict
survives. Domain artifact vocabulary lives on the plugins
(`artifact_question_markers`), not in core.

### OOD / scope-integrity gates — `scoped_claim.py`
Downgrade confirmed → inconclusive when out-of-distribution evidence is missing,
or when the executed LOFO does not match the declared OOD scope. Scope templates
live in the plugins (`scope_template()`), resolved via the registry.

### Evidence Binding — `packages/propab-core/propab/evidence_binding.py`
Ensures every citation (a node listed as supporting/contradicting a belief, any
evidence attributed to a mechanism) actually bears on the specific claim. It runs
**at write time**: `apply_synthesis_to_frontier` passes `tree_nodes` + metrics to
`belief_state.apply_synthesis_beliefs`, which filters citations *before* the
belief is persisted. A fabricated citation is rejected before it can be written
(covered by `test_synthesis_pass_rejects_fabricated_citation_before_persist`).

---

## 8. Domain layer

The domain plugin is the single seam between domain-agnostic core and
domain-specific science. Core imports **no** dataset name, feature name, or
threshold directly.

### `DomainPlugin` — `packages/propab-core/propab/domain_modules/base.py`
Contract (each has a safe default so a plugin overrides only what it needs):
`matches` / `matches_scope`, `available_features`, `run_verification`,
`classify_verdict`, `artifact_models`, `confirmation_criteria`, **`preflight`**,
`literature_prior`, `scope_template`, `artifact_question_markers`,
`domain_profile`, **`belief_promotion_threshold`**, **`implementable_methodologies`**,
**`extract_numerical_seeds`** (math compounding: trend promotion + numerical seeds).

### Registry — `packages/propab-core/propab/domain_modules/registry.py`
`resolve_domain_plugin` resolves the owning plugin: explicit
`domain`/`domain_profile` on the payload → `[domain_profile:<id>]` tag in the
question → each plugin's own `matches`. Core calls only the registry.

Built-in plugins: `MaterialsPlugin`, `MandrakePlugin`, `EnzymeKineticsPlugin`,
`GraphInvariantsPlugin`, `NetworkDiffusionPlugin`. Domain **profiles**
(`domain_profiles/`) configure the artifact gate; domain **adapters**
(`domain_adapters/`) run the experiment; the plugin fronts both.

### Preflight gate (fail-fast power check)
Before a **fresh** campaign starts, `_enforce_domain_preflight` resolves the
owning plugin and calls `plugin.preflight()`. If it returns `passed=False`, the
campaign is finalized immediately with `DOMAIN_PREFLIGHT_FAILED` and never burns
compute. The materials plugin, for example, loads its frame and checks row
count; a domain with too few samples/groups fails here in seconds instead of
after hours of inconclusive experiments. The gate is fail-open on a plugin
*exception* (a buggy preflight must not block launch) and fail-closed on an
explicit `passed=False`. Resumed campaigns skip the gate (already passed).

---

## 9. Worker

`run_sub_agent_async` (`services/worker/sub_agent_loop.py`) runs the think-act
loop for one hypothesis: it resolves the domain via the plugin registry and a
`{domain_id → worker path}` table, selects the tool cluster
(`get_cluster_with_significance(domain)`), runs the experiment (domain fast path
or generic tool loop), builds evidence, and returns a verdict within the compute
budget. A timed-out worker returns `inconclusive` with reason `timeout` rather
than hanging. Dispatched via Celery `run_sub_agent_task.delay(...)`.

---

## 10. Lifetime learning

At campaign end, `ingest_campaign`
(`services/orchestrator/lifetime_knowledge.py`) writes claims, mechanisms,
failures, and theories to the `KnowledgeGraph`, an observation to the
`MetaScienceLedger`, fitness records to the `PolicyFitnessLedger`, and a
candidate policy proposal. These stores are JSON files under
`{PROPAB_DATA_DIR}/lifetime_knowledge/` with last-writer-wins semantics; because
`ingest_campaign` runs once per campaign (single writer per campaign), there is
no concurrent-write path today. `lifetime_context_for_seeds` injects prior
lifetime knowledge into seed prompts.

---

## 11. Health metrics and observability

`packages/propab-core/propab/health_metrics.py` computes the eight
ownership-contract metrics from data the pipeline already produces and persists
them. It never raises into the caller — metric logging must not break a campaign.

- **Per synthesis round** (`log_synthesis_health` → `campaign_synthesis_events`):
  hypothesis duplicate rate, evidence-binding rejection rate, belief citation
  integrity, and belief stability (computed against the previous round's
  persisted belief statements — correct across restarts). Emits warnings when a
  rate leaves its target range, and warns if Evidence Binding is called 50+
  times with zero rejections (a sign the check is not running).
- **Per literature prior build** (`log_literature_prior_health` →
  `campaign_literature_priors`): citation verification rate (fraction of
  established facts carrying a retrievable citation).
- **Per campaign end** (`log_campaign_end_health` → `research_campaigns`):
  worker experiment success rate (definitive verdicts / tested hypotheses) and
  worker utilization (summed sub-agent seconds / (elapsed × max concurrency)).
- **Per campaign audit** (`log_campaign_audit` → `campaign_audit_results`):
  artifact-gate precision — confirmed findings backed by an independent
  null/permutation test. The in-pipeline signal is logged automatically; the
  offline permutation-audit scripts can overwrite it with a full re-run.

Every campaign stop event carries a non-null enum `stop_reason` in
`research_campaigns` (the stop-reason-accuracy companion metric).

---

## 12. Events, streaming, and persistence

- **Events.** `EventEmitter.emit` (`services/orchestrator/events.py`) publishes
  each named event to Redis channel `propab:{session_id}` and appends it to the
  Postgres `events` table. The SSE endpoint `GET /stream/{session_id}`
  (`services/api/app/routes/stream.py`) replays and tails it. Peer findings use
  Redis lists `propab:peer:{hypothesis_id}`.
- **Persistence.** `propab/db.py` provides the async engine/session/redis
  factories and `insert_event`. Campaign state round-trips through
  `propab.campaign_db`.
- **Schema (Alembic).** `research_sessions`, `research_campaigns` (+ belief-state
  meta, `stop_reason`, `worker_experiment_success_rate`, `worker_utilization`),
  `hypotheses`, `experiment_steps`, `tool_calls`, `llm_calls`, `events`,
  `literature_query_cache`, and the health-metric tables
  `campaign_synthesis_events`, `campaign_literature_priors`,
  `campaign_audit_results`. Alembic is the single source of truth; the
  `migrate` service applies `alembic upgrade head` before app services start.

---

## 13. Failure handling and resumability

- **API restart** does not kill a running campaign — the loop lives in the
  orchestrator.
- **Worker crash** requeues the in-flight hypothesis (`task_acks_late` +
  `task_reject_on_worker_lost`); it goes back to pending rather than
  disappearing.
- **Campaign resume** reloads full state (tree + belief state) from Postgres and
  continues; the preflight gate is skipped for a warm resume.
- **Fatal error** in the loop finalizes with `FATAL_ERROR` (or
  `SALVAGED_AFTER_ERROR` when a partial paper is salvaged) — never a null stop
  reason.

---

## 14. Repository structure

```
packages/propab-core/propab/   # domain-agnostic core (importable everywhere)
  campaign.py                  # ResearchCampaign, stop reasons
  campaign_db.py               # campaign persistence (shared by all services)
  campaign_synthesis.py        # Tier-2 synthesis + per-round metric logging
  belief_state.py              # rival beliefs, evidence binding at write time
  hypothesis_tree.py           # hypothesis tree + frontier
  verdict_pipeline.py          # composed verification pipeline
  artifact_verification.py     # artifact gate (null/permutation tests)
  scoped_claim.py              # OOD / scope-integrity gates + templates
  evidence_binding.py          # citation relevance at write time
  health_metrics.py            # eight ownership-contract metrics
  domain_modules/              # DomainPlugin interface + registry + plugins
  domain_profiles/             # artifact-gate profiles
  domain_adapters/             # per-domain experiment runners
  tools/                       # discovery-based STEM tool registry

services/
  api/                         # FastAPI HTTP entrypoint + routes + SSE
  orchestrator/                # run_campaign_loop, literature, events, paper
  worker/                      # Celery app + run_sub_agent_async + tool loop

alembic/                       # migrations (single schema source of truth)
frontend/                      # Vite/React campaign dashboard
docs/                          # component_map.md and design notes
scripts/                       # preflight, audit, power-analysis, ops scripts
tests/                         # 467-test suite
```

---

## 15. Testing

The project suite is `pytest tests` (**467 passing**). Run it from the repo root
after `pip install -e ".[dev]"`. The vendored `asta-bench/` directory has its own
dependencies and is **not** part of the project suite — scope pytest to `tests/`.

Notable coverage: verdict pipeline composition, artifact verification, scoped
claims, evidence binding (write-time rejection), campaign redesign/recount,
synthesis builder, domain plugins, and the health-metric + preflight-enforcement
tests (`tests/test_campaign_health_and_preflight.py`).

---

## 16. Known gaps

- No single end-to-end `run_campaign_loop` unit test (the loop is exercised
  piecewise + validated live on a running stack).
- Enzyme/graph domains have artifact-gate profiles but no dedicated worker
  verification path yet (they use the generic path with per-domain
  `min_metric_steps`).
- Artifact-gate precision is logged from the in-pipeline null-test signal at
  campaign end; a full independent permutation re-audit is run by the offline
  audit scripts.
- Lifetime-learning stores are JSON with last-writer-wins; moving them to
  Postgres is a future item once campaigns run concurrently.
