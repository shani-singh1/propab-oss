# Propab Architecture Decision Register (ADR)

> **Living document.** Every architectural decision baked into the codebase is
> catalogued here with a rationale, a correctness judgment + tradeoff, and a
> status. Nothing is assumed obvious — small and large decisions are both
> questioned. The refactor loop consumes this doc and updates statuses.
>
> Maintainer note: this is a **first deep pass** (2026-07-09). HLD is covered
> broadly; central LLD subsystems are covered; areas not yet deep-read are marked
> `INVESTIGATE` rather than rubber-stamped. Extend, don't fake completeness.

## Status enum
- **KEEP** — sound HLD/LLD, correct for Propab; rationale recorded.
- **KEEP-WATCH** — acceptable, but a real tradeoff/risk to monitor.
- **FIX** — localized architectural defect; correct in place.
- **REPLACE** — fundamentally wrong for the goal; redesign required.
- **REMOVE** — dead / redundant / cruft; delete.
- **INVESTIGATE** — not yet deep-audited; decision pending (do NOT act yet).

Each entry interrogates the decision like an elite reviewer would:
**What** · **Why (rationale)** · **How (mechanism)** · **Needed?** · **Better way?** ·
**Scalable / Deployable / Maintainable / Testable** · **Assessment + tradeoff** ·
**Status** · **Action**. (Early sections A–J use a condensed form; grounded
code-read sections K+ use the full form.)

## Coverage tracker (drives the audit loop until 100%)
Legend: ✅ code-read + grounded · 🟡 partially read · ⬜ not yet read.

| Subsystem | LOC | Status |
|---|---|---|
| Backbone: db/events/llm/celery/api-entry/config | ~1.5k | ✅ (§K) |
| services/api routes (research/sessions/stream/tools) | 1.1k | 🟡 (research.py read; rest ⬜) |
| services/orchestrator/campaign_loop.py | 2.9k | 🟡 (verdict/dispatch/apply read; full ⬜) |
| services/orchestrator (hypotheses/prior_builder/answer_gate/question_domain/lifetime/policy_analyst/ranking/seed_validation/research_loop/diagnostics/budget/ledger) | ~4k | 🟡 (design surface mapped → §M; literature.py/literature_client/retrieval/literature_cache/literature_quality full ⬜) |
| services/worker/sub_agent_loop.py | 3.0k | 🟡 (verdict/confidence/routing/return read; full ⬜) |
| services/worker (think_act/permutation_null/sandbox/domain_router/failure_classify/sandbox_code_rewrite/peer_findings) | ~1.5k | 🟡 (think_act/sandbox/domain_router/peer_findings/significance read → §L; permutation_null/failure_classify/sandbox_code_rewrite ⬜) |
| services/literature (65 files) | 9.2k | ✅ design surface → §P (KEEP; own Gemini egress §P2) |
| core: hypothesis_tree | 0.9k | 🟡 |
| core: verdict_pipeline/significance | 0.4k | ✅ (§E, §K) |
| core: artifact_verification | 0.8k | ✅ (§O1) |
| core: campaign_synthesis | 1.0k | 🟡 |
| core: campaign / campaign_db / campaign_snapshot | ~1k | ✅ (§Q1) |
| core: research_quality / evidence_binding / scoped_claim / claim_grounding | ~2k | ✅ (§O) |
| core: paper_compiler / paper_narrative / paper_sections / paper_gate | ~1.9k | ✅ (§O6 + §Q2) |
| core: telemetry (§Q3 LIVE moat) / telemetry_db / health_metrics / knowledge_graph / numerical_seeds | ~1.5k | 🟡 (telemetry ✅; db/health/kg/seeds ⬜) |
| domain_modules (12 domains) | 14.7k | 🟡 (genomics ✅; pattern known; rest ⬜) |
| domain_adapters / domain_profiles | ~2.1k | ⬜ |
| anomaly_engine | 1.9k | ✅ (§Q4) |
| tools (registry ✅; tool impls ⬜) | 4.3k | 🟡 |
| skills | 0.3k | ✅ |
| operator_credit / layer05 | ~8.4k | ✅ (traced → SHELVED §I) |
| frontend/src | — | 🟡 (model/events/panels known from rebuild) |
| Security & data governance (auth/egress/tenancy/at-rest) | — | ✅ posture grounded → §N (build deferred) |

---

## A. HLD — System topology & service boundaries

### A1. Split into api / orchestrator / worker / literature services — KEEP-WATCH
- **What:** 4 Python services + postgres/redis/qdrant/minio, Celery between orchestrator→worker.
- **Why:** separate the always-on API, the long-running campaign brain, burstable experiment execution, and a heavy literature/RAG subsystem; scale workers independently.
- **Assessment/tradeoff:** correct for parallel experiment execution and independent worker scaling. Cost: 4 deploy units, cross-service imports, and stale-per-service deploys (we hit this: api ran old verdict code while workers ran new). Boundaries are currently *leaky* (see A2, E1). Sound topology, but the boundaries must be made honest.
- **Action:** keep the split; enforce the boundary (A2). Add a "rebuild-all or none" deploy rule to kill stale-per-service drift.

### A2. Layering: services depend on `propab-core`; core is domain- & service-agnostic — FIX
- **What:** intended dependency direction api/orchestrator/worker → propab-core → (nothing upward).
- **Why:** core holds reusable, testable, domain-general logic; services compose it.
- **Assessment/tradeoff:** **violated** — `campaign_synthesis.py`, `replay_support.py`, `cli.py` in core lazily import `services.*` (deferred imports to dodge a cycle). This is dependency inversion; it means "core" isn't actually a lower layer. Tradeoff of fixing: must move the imported helpers (e.g. `hypothesis_ranking`) down into core or invert via injection.
- **Action:** move shared helpers into core or pass them in; forbid `from services` inside `propab-core` (add a lint/test guard).

### A3. Separate literature microservice (9.2k LOC, own FastAPI + tests) — KEEP (earns separateness; see §P)
- **What:** standalone service for papers/embeddings/gaps/contradictions over HTTP.
- **Why:** literature/RAG is heavy and benefits from isolation + independent scaling.
- **Assessment/tradeoff (RESOLVED — read 2026-07-09, §P):** genuinely substantial and cleanly isolated (does NOT import propab-core): 7 sources, dual-store (PG+qdrant) RAG, extractors, novelty/gap mappers, thin main.py over a `LiteraturePipeline`. The separateness is justified. Caveat: it has its OWN Gemini LLM client (§P2) → a second egress point.
- **Action:** KEEP the service; still expose it to the orchestrator as a **tool** (not pre-fetch, M3); unify/govern the second LLM egress for security (N2).

### A4. Celery (redis broker) for orchestrator→worker dispatch — KEEP
- **What:** `run_sub_agent_task.delay(payload)`, pipelined pool up to N.
- **Why:** durable, parallel, retryable task execution decoupled from the brain.
- **Assessment/tradeoff:** appropriate for fan-out experiment execution. Tradeoff: payload-as-dict contract is untyped (see G3). Keep Celery, type the contract.
- **Action:** keep; formalize the dispatch/result payload (redesign §3).

---

## B. HLD — Campaign control flow (the core loop)

### B1. Orchestrator is a procedural pipeline, not an agent — REPLACE  ⟵ primary redesign
- **What:** the orchestrator runs a fixed script (seed→dispatch→score→batch-synthesis→finalize) with discrete LLM calls; it has **no tool-use reasoning loop** and never reasons over the tree.
- **Why (as built):** simpler to implement deterministically; each step testable in isolation.
- **Assessment/tradeoff:** **wrong for Propab's goal.** The whole value is an orchestrator that reads results with full tree context and *reasons* about what's working / what to deepen / drop. Procedural = no strategy, no visible thinking, and it forced judgment down into workers (E1). Tradeoff of fixing: an LLM reasoning call per returned result (cost/latency) — acceptable, and the point.
- **Action:** the orchestrator-brain redesign (`orchestrator-brain-redesign.md`), phases C1–C5. **In progress.**

### B2. "What to test next" decided by a hand-tuned `frontier_score` formula — REPLACE
- **What:** `frontier_score = info_gain × closure` (weighted relevance/novelty/uncertainty/…) picks the next node; plus a threshold-triggered batch synthesis; plus a dead `build_expand_prompt` path. Three mechanisms.
- **Why:** deterministic, cheap, no LLM in the hot loop.
- **Assessment/tradeoff:** a formula cannot know "this line is promising, narrow it; that one's a dead end." Three competing mechanisms is incoherent. Keep the score only as a cheap **dispatch-order prior**; the *decision* must be reasoned.
- **Action:** fold into the orchestrator reasoning step (C3); delete the dead expand path.

### B3. RETUNE (same hypothesis, new params/data) has no representation — FIX
- **What:** no node "attempt" concept; re-running a hypothesis differently isn't modeled.
- **Why:** n/a (missing).
- **Assessment/tradeoff:** the intended loop needs "try again differently" distinct from "deepen (child)". Add `attempts[]` on the node, bounded by config.
- **Action:** add in C3 (`max_retune_rounds_per_hypothesis` already added in config).

### B4. Three parallel seed sources (LLM seeds, anomaly seeds, synthesis) — INVESTIGATE
- **What:** LLM seed-gen + `generate_anomaly_seed_hypotheses` + Tier-2 synthesis all create nodes.
- **Why:** breadth of idea generation.
- **Assessment/tradeoff:** may be redundant/competing once the orchestrator reasons about generation. Decide which survive under the agent model.
- **Action:** re-evaluate during C3.

---

## C. HLD — Persistence & state

### C1. Dual state: serialized HypothesisTree **and** per-row `hypotheses` table — DONE (split-brain fixed)
- **What:** campaign state lives both as a serialized tree and as DB rows; the worker wrote the row verdict, the orchestrator mutated the tree, and they **diverged** (split-brain).
- **Why:** tree for in-memory campaign logic; rows for API/paper queries.
- **Assessment/tradeoff:** two sources of truth with no reconciliation was a data-integrity bug.
- **Action:** ✅ **C1 landed (commit 094dfb5, 39 tests):** `_apply_result_diagnostics` returns a `DiagnosticsOutcome`; `_persist_effective_verdict` mirrors the tree's effective verdict/confidence to the DB row only on divergence. DB now mirrors the tree (single source of truth). Remaining: still two *stores* (acceptable) — full unification deferred; consistency is now guaranteed.

### C2. Two migration systems: `alembic/` (11 versions) + `migrations/*.sql` (3 files) — REMOVE(one)
- **What:** alembic is the live system (redeploy asserts alembic-at-head); raw `migrations/*.sql` also present.
- **Why:** raw SQL predates alembic adoption (legacy).
- **Assessment/tradeoff:** two systems = confusion about the source of truth for schema. Confirm alembic subsumes the raw SQL, then delete `migrations/`.
- **Action:** verify coverage; delete `migrations/` (or document it as archival) — REMOVE candidate.

### C3. Persistence appears to be raw SQL / asyncpg without ORM model classes — INVESTIGATE
- **What:** no `declarative_base`/`__tablename__` model classes found on a first grep despite sqlalchemy in deps.
- **Why:** unknown (raw queries may be a deliberate perf/control choice).
- **Assessment/tradeoff:** raw SQL is fine but scatters schema knowledge; needs confirmation of where table access is centralized.
- **Action:** locate the DB-access layer; decide ORM-models vs a typed query module.

---

## D. HLD — Domain plugin model

### D1. Domain-independent core + per-domain plugins (12 domains) — KEEP
- **What:** `DomainPlugin` registry; each domain owns adapter/verifier/plugin/routing; core stays general.
- **Why:** multi-domain discovery without core edits; the north-star design constraint.
- **Assessment/tradeoff:** correct and load-bearing. Tradeoff: contract discipline needed so core never learns domain specifics. Mostly honored; one violation is worker-side domain routing (E2).
- **Action:** keep; enforce (see D2, E2).

### D2. Domain routing — WHY it exists, and how to do it right — REPLACE (partly done)
- **The user's question, answered — why is domain routing even needed?** To pick the correct, honesty-**audited verifier + objective frame** for a hypothesis. Verification is domain-specific (genomics = leave-one-tissue-out R²+null; math = exact `is_B3` check; materials = permutation test) and its honesty framing (`objective_spec`: `is_ml`, metric, baseline_kind) differs per domain. You cannot verify a genomics claim with a math verifier, and you must not let the LLM improvise a flawed null each time (it already produced buggy nulls when hand-written). So *some* routing is genuinely required. What is NOT required: guessing the domain from hardcoded keywords, or taking it as user input.
- **What was wrong:** three hardcoded/guessing mechanisms — `question_domain.infer_session_domain` (session-level keyword taxonomy), `domain_router.route_domain` + `_keyword_fallback_domain` (worker per-hypothesis keyword/LLM router), and the `[domain_profile:X]` user tag (`domain_from_profile_tag`).
- **The right design:** the registry ALREADY supports plugin **self-detection** — `resolve_domain_plugin(question, payload)` calls each `DomainPlugin.matches()` (registry.py:135). The worker already uses it (sub_agent_loop:1803). So central keyword tables + user tags are redundant; the orchestrator (reasoning brain) confirms/overrides routing per hypothesis.
- **Status/Action:** ✅ `question_domain.py` + `infer_session_domain` **deleted** (commit 59a2c03); intake no longer guesses a domain. ⏳ The worker still ALSO calls `route_domain` (domain_router, line 1954) alongside `resolve_domain_plugin` (line 1803) — two routers. `domain_router` is entangled in the worker verification flow that C2/C3 rewrites; **delete it during C2/C3** when routing moves to the orchestrator + plugin self-detection. Do NOT rip it out standalone (would break verification).

---

## E. HLD — Honesty / verdict architecture (Propab's credibility core)

### E1. Verdict decided in the worker, no tree context — REPLACE
- **What:** `run_verdict_pipeline` + `classify_verdict` + significance gate run in the worker.
- **Why (as built):** verdict computed where the experiment ran.
- **Assessment/tradeoff:** the judge has no parent/sibling/tree context and is duplicated across N workers → bugs hide in many places (the genomics false-1.0 lived here). Judgment must be central.
- **Action:** move to orchestrator, one central gate (C2/C3).

### E2. ≥4 competing verdict implementations + duplicated confidence — FIX→REPLACE
- **What:** `verdict_pipeline.run_verdict_pipeline`, `significance.classify_verdict`, each `DomainPlugin.classify_verdict`, mandrake/materials adapters; `_compute_confidence` vs `_compute_pipeline_confidence`.
- **Why:** grew organically per subsystem.
- **Assessment/tradeoff:** multiple honesty gates = no single audit point; the most dangerous kind of duplication for a credibility engine.
- **Action:** collapse to one core verdict+confidence impl the orchestrator invokes. ✅ **Confidence consolidated (C1, 094dfb5):** one canonical `compute_confidence` in core; worker `_compute_confidence` is a thin adapter (verified no behavior delta at the live call site). Remaining: the ≥4 verdict *classifiers* collapse in C2/C3.

### E3. `objective_spec(is_ml=False)` deterministic-frame + artifact/ood/scope honesty gates — KEEP
- **What:** evidence-type classification → artifact_gate → ood_gate → scope_integrity; deterministic domains run against baseline≈0.
- **Why:** prevents the "val_accuracy" class of false-breakthrough; matches evidence shape to gate.
- **Assessment/tradeoff:** this is genuinely good design and Propab's honesty backbone. Keep — just relocate its *invocation* to the orchestrator (one place).
- **Action:** keep the pipeline; centralize where it runs.

### E4. `classify_verification_method` substring-matches (mislabels `symbolic_identity`) — FIX
- **What:** telemetry classifier returns `symbolic_identity` on any `"verified_true"` substring — matches `"verified_true_steps": 0`.
- **Why:** cheap string heuristic.
- **Assessment/tradeoff:** cosmetic but wrong; pollutes telemetry/UX. Parse structured evidence, not substrings.
- **Action:** FIX (low priority) — parse the field, not the string.

---

## F. HLD — Deployment / ops

### F1. docker-compose as the deploy unit; `redeploy.sh` health-asserts — KEEP-WATCH
- **What:** compose up + alembic-at-head + health checks; multiple compose variants (prod/dev/campaign-run/astabench).
- **Why:** reproducible local+prod stack.
- **Assessment/tradeoff:** reasonable. Watch: many compose variants drift; partial rebuilds cause stale-service bugs (proven). Add a canonical "rebuild all changed" path.
- **Action:** keep; document the compose-variant matrix; enforce full-rebuild on core changes.

### F2. Healthchecks assume tools present in images (qdrant had none) — FIX(done)
- **What:** qdrant healthcheck used wget absent from the image.
- **Assessment/tradeoff:** already fixed (dropped the unworkable probe). Generalize: healthchecks must use tools that exist in each image.
- **Action:** done; note as a standing rule.

---

## G. LLD — Core data structures & contracts

### G1. `HypothesisTree` (926 LOC) as the campaign state object — KEEP-WATCH
- **What:** nodes dict + frontier + confirmed set + scoring + serialization.
- **Why:** one structure for the search tree.
- **Assessment/tradeoff:** central and reasonable, but it also *encodes policy* (frontier_score, downgrade logic in `update_node`) that should move to the orchestrator's reasoning. Keep the structure; extract the policy.
- **Action:** thin it to state + queries; move decisions out (C3).

### G2. `EventType` — 87 events, none for orchestrator reasoning — FIX
- **What:** rich lifecycle/worker event vocabulary; zero orchestrator-reasoning events.
- **Why:** events grew around what existed (worker think-act, lifecycle).
- **Assessment/tradeoff:** the missing category is exactly what the UI must show. Add reasoning/decision events (plain-language labels in UI, not `ORCH_*`).
- **Action:** add in C3; FRONTEND renders (C5).

### G3. Worker dispatch/result contract is an untyped dict with an embedded verdict — REPLACE
- **What:** `run_sub_agent_task.delay({...})` in, `{... verdict ...}` out.
- **Why:** quick to evolve.
- **Assessment/tradeoff:** untyped + carries a verdict the worker shouldn't own. Replace with a typed contract: dispatch `{hypothesis, instructions, tools_allowed, skills_available}`, result `{experiment_design, what_was_tested, raw_evidence, artifacts}` (no verdict).
- **Action:** C2.

---

## H. LLD — Tools & skills

### H1. `ToolRegistry` auto-scan of `TOOL_SPEC` modules — KEEP
- **What:** walk `propab.tools`, register modules exposing `TOOL_SPEC`.
- **Why:** drop-in tools without central registration.
- **Assessment/tradeoff:** good extensibility. Now has audience scoping (added). Keep.
- **Action:** keep; use audience scoping in the agent loop.

### H2. Skills as markdown with agentic on-demand read — KEEP
- **What:** 18 core skills; catalog shown, agent reads chosen bodies.
- **Why:** cheap awareness, expensive body only when needed; matches "orchestrator/worker read skills."
- **Assessment/tradeoff:** genuinely good pattern; underused (only at seed-gen today). Generalize to the orchestrator agent + workers.
- **Action:** keep; wire into both agents with audience scoping.

### H3. Orchestrator node tools (get_node/mark_node/list_frontier/write_hypothesis) — MISSING→BUILD
- **What:** don't exist; verdict/expansion done by code, not tools.
- **Why:** n/a.
- **Assessment/tradeoff:** the agent needs tools to inspect/mark the tree deterministically (vs prose-parsing).
- **Action:** build in C3.

---

## I. LLD — Big subsystems to audit next (honest INVESTIGATE)

### I1. `layer05` — offline replay / simulation / policy-eval (4.9k LOC) — SHELVED (revisit post-C3)
- **What:** search/hybrid/ensemble simulators, offline policy eval, calibration, fitness ledgers.
- **Why:** offline learning of the search/dispatch policy across campaigns (the "moat").
- **Assessment/tradeoff (TRACED 2026-07-09):** live footprint is 3 imports — a `SimulationFitnessLedger` load + `policy_analyst` (which is *decorative*: "the LLM never edits") + a small `_policy_score_multiplier` nudge on `frontier_score`. The **simulator bulk** (`simulate_search`, hybrid/ensemble, `replay_campaign_snapshots`, offline eval) is called **only by `operator_credit`** (I2), which has no consumers → dead subgraph. So ~4k of the 4.9k LOC does not affect a live campaign.
- **Action:** DECISION NEEDED (strategic, user's call — this is Track B "moat"): (a) **wire it** into the live loop so learned policy actually steers dispatch/expansion, or (b) **shelve** it behind a clearly-labelled flag/branch until the reasoning loop needs it. Do NOT silently carry it as if it's active. The tiny live policy-multiplier hook is KEEP-WATCH meanwhile.

### I2. `operator_credit` (3.5k LOC) — telemetry/operator-statistics moat — SHELVED (revisit post-C3)
- **What:** per-operator credit, running stats, difference-rewards, counterfactual replay.
- **Why:** the "telemetry moat" track (Track B).
- **Assessment/tradeoff (TRACED 2026-07-09):** **zero** non-self, non-test references anywhere in `packages/` or `services/`. It is a fully disconnected island — recorded/computable but feeding **no** decision. A moat that feeds nothing is not yet a moat; it's unintegrated code carrying maintenance + honesty risk (it can drift like the genomics verifier did, invisibly).
- **Action:** DECISION NEEDED (user's call): **wire operator-credit into the orchestrator reasoning loop** (so per-operator/per-mechanism credit actually informs what the brain tries next — this is where it becomes a real moat), or **shelve** it explicitly. Confirm there is also no CLI/script/cron entrypoint before either. NOT an auto-delete: it is strategic-by-intent, just unwired.

### I3. Literature service internals (9.2k LOC) — INVESTIGATE (see A3).

### I4. `paper_compiler`/`paper_narrative`/`paper_sections` (~1.7k LOC) — INVESTIGATE
- **What:** turns a finalized campaign into a paper.
- **Assessment/tradeoff:** valuable output, but writes a paper even on zero confirmed findings (an honest-signal guard exists). Confirm it never overclaims.
- **Action:** audit the honest-reporting path.

### I5. `anomaly_engine` (1.9k LOC) + `numerical_seeds`/`evidence_binding`/`scoped_claim` — INVESTIGATE.

---

## J. Repo hygiene (not code, but architecture debt)

### J1. Top-level scratch docs (`agent1.md`, `agent2.md`, `agent3.md`, `fixes.md`, `test_que.md`, `propab_ownership_contracts.md`) — REMOVE
- **Why they exist:** working notes from prior sessions.
- **Assessment:** clutter the root; not source of truth. Move real content into `docs/`, delete the rest (matches the standing "delete temp/scratch" rule).
- **Action:** triage → delete/relocate.

### J2. `artifacts/`, `logs/`, `bench/`, tracked into repo — REMOVE/gitignore
- **Assessment:** build/run outputs shouldn't be tracked (standing rule: artifacts/logs never pushed).
- **Action:** gitignore + purge from tracking.

---

## K. Backbone (grounded — code read 2026-07-09)

### K1. Persistence = raw SQL (`sqlalchemy.text`) + asyncpg + `jsonb`, no ORM models — KEEP-WATCH
- **What:** `db.py` exposes `create_engine`/`create_session_factory`; all table access is hand-written SQL strings (`events`, `llm_calls`, `hypotheses`, `campaigns`, …); JSON columns stored as `jsonb`.
- **Why:** full control over queries; avoids ORM overhead/magic; async-native via asyncpg.
- **How:** each module writes its own `text("INSERT/SELECT …")`; schema defined only in `alembic/` + `migrations/*.sql`.
- **Needed?** A data layer, yes. Raw SQL specifically — defensible but not required.
- **Better way?** A thin typed query module (or lightweight table-metadata) would centralize the ~dozen scattered SQL strings and give one place to see the schema. Full ORM is probably overkill for this workload.
- **Scalable:** fine (asyncpg + pool_pre_ping). **Deployable:** fine. **Maintainable:** ⚠ weak — schema knowledge is smeared across many `text()` literals + two migration systems (C2); a column rename is a global grep. **Testable:** needs a live PG (no model-level unit tests).
- **Assessment/tradeoff:** works and is fast, but the schema has no single source of truth in code. Risk: silent drift between SQL literals and actual schema (exactly the `inconclusive_reason`-has-no-column gotcha C1 hit).
- **Action:** FIX-later — introduce a single `schema.py`/typed-row module enumerating tables+columns; keep raw SQL but import column names from it. Low priority, high maintainability payoff.

### K2. Event log: `events` table append-only + per-event commit, then redis publish — KEEP-WATCH
- **What:** `EventEmitter.emit` writes one row to `events` (with its own commit) then `redis.publish` to `propab:{session_id}`; SSE clients replay via `load_events_after`.
- **Why:** durable audit/replay + live streaming from one call; reconnecting UI can catch up.
- **How:** `insert_event` commits per event; redis is the live bus; DB is the replay store.
- **Needed?** Yes — the campaign transcript IS the product surface (frontend) and the audit trail.
- **Better way?** Batch inserts / a single transaction per step would cut commit overhead; or an append-only stream (redis stream) with periodic DB flush.
- **Scalable:** ⚠ one INSERT+COMMIT per event; a busy campaign emits thousands (llm.prompt/response, tool.*, progress). At scale this is a write-amplification + fsync hotspot. **Maintainable:** simple, good. **Testable:** good.
- **Assessment/tradeoff:** simplicity now vs write throughput later. Fine at current volume; will bite under many parallel campaigns.
- **Action:** KEEP-WATCH; revisit batching if event volume becomes a bottleneck. No orchestrator-reasoning events exist yet (G2 — add in C3).

### K3. `LLMClient` — multi-provider, fail-loud, retrying; single model per instance — FIX(wire role-split)
- **What:** one client for openai/gemini/ollama; validates provider+key at construction (raises `LLMConfigError`, never fabricates a placeholder); bounded exponential backoff on transient (timeout/429/5xx); emits `llm.prompt`/`llm.response` with a `call_id`, duration, tokens; persists to `llm_calls`.
- **Why:** provider-agnostic; honesty (a misconfig must fail loud, not silently "research" a canned answer across domains); resilience (one timeout used to kill a campaign).
- **How:** `_call_provider_once` dispatches per provider; `usage_out` dict threads tokens without racing concurrent calls.
- **Needed?** Yes — central, correct LLM boundary.
- **Better way?** Minor: streaming responses; a shared httpx client (currently one per call — connection churn). Not architectural.
- **Scalable:** ok (per-call client is a tiny inefficiency). **Deployable:** ok. **Maintainable:** good, single boundary. **Testable:** good (provider methods monkeypatchable).
- **Assessment/tradeoff:** genuinely well-built and honesty-aligned — one of the better modules. Its one redesign gap: it takes a single `model`, so the **orchestrator/worker model split** (config added in Wave 1) is **not wired** — every `LLMClient(...)` construction currently passes `settings.llm_model`.
- **Action:** FIX in C3 — construct the orchestrator's client with `effective_orchestrator_model` and workers' with `effective_worker_model`. Consider a shared httpx client (minor).

### K4. Celery config — `acks_late` + `reject_on_worker_lost` + `visibility_timeout` — KEEP
- **What:** JSON serializer; `task_acks_late=True`, `task_reject_on_worker_lost=True`, `visibility_timeout = hard_limit+60`; soft/hard time limits (env-tunable, default 3600/3900s).
- **Why:** a hypothesis task is only acked after completion → a killed worker's task is redelivered, not dropped.
- **Needed? / Better way?** Yes; correct pattern. Tradeoff: at-least-once delivery ⇒ a task may run twice (crash after side effects) — verify idempotency of `_update_hypothesis`/event writes.
- **Scalable/Deployable/Maintainable:** all good.
- **Assessment/tradeoff:** correct, thoughtful resilience. **Action:** KEEP; add an idempotency note/guard for redelivered tasks (INVESTIGATE: are event/DB writes idempotent on redelivery?).

### K5. API entry — FastAPI `lifespan` holds engine/session/redis/emitter; CORS `allow_origins=["*"]` — FIX(prod CORS)
- **What:** shared engine/session_factory/redis/emitter on `app.state`; 5 routers; CORS wide-open, `allow_credentials=False`.
- **Why:** local dev simplicity; single place for shared resources.
- **Needed?** Shared-resource lifespan — yes. Wildcard CORS — only for local.
- **Better way?** Env-driven allowed origins for prod.
- **Scalable/Deployable:** fine. **Maintainable:** fine. **Security:** ⚠ `*` CORS is acceptable only because there are no credentials/cookies; still tighten for a real deployment.
- **Assessment/tradeoff:** fine for now; a prod checklist item.
- **Action:** FIX-later — origins from config in prod.

### K6. Worker plumbing: `tasks → runner → asyncio.run(run_sub_agent_async)`; `worker/significance.py` re-exports `propab.significance` — KEEP
- **What:** the Celery task is a 3-line shim to a sync runner that drives the async loop; `worker/significance.py` is `from propab.significance import *`.
- **Why:** keep Celery boundary thin; single significance impl in core.
- **Assessment/tradeoff:** clean. Clears part of the first audit's E2 worry: significance is NOT duplicated in the worker — it's a re-export. (Open: does `propab.significance.classify_verdict` differ from `verdict_pipeline.classify_verdict`? — verify during the verdict-consolidation of C2.)
- **Action:** KEEP; resolve the significance-vs-verdict_pipeline classifier question in C2.

---

## L. Worker execution layer (grounded — code read 2026-07-09)

### L1. Worker think-act loop (`think_act.decide_next_action`) — KEEP (worker's core job)
- **What:** the worker's LLM observes accumulated results + extracted numeric values + peer findings + tool failures, and chooses ONE action: `tool` | `code` | `stop`. Bounded by max_steps + a monotonic wall deadline.
- **Why:** per-hypothesis autonomous experimentation — design + run + iterate.
- **How:** a single big think prompt → JSON action → execute → repeat; value-extraction feeds real measurements back into the next prompt (good — prevents placeholder drift).
- **Needed?** Yes — this IS the experimenter. Survives the redesign (worker keeps its design LLM; it just stops *judging*).
- **Better way?** Split the mega-prompt by task shape; see L3.
- **Scalable:** ok (per-hypothesis, parallel). **Maintainable:** ⚠ one 300-line prompt template mixing concerns. **Testable:** decision logic is unit-testable; prompt quality isn't.
- **Assessment/tradeoff:** correct role, but the loop currently also owns the significance gate (L5) which blurs into the verdict.
- **Action:** KEEP; in the redesign the worker returns raw evidence, orchestrator judges (E1).

### L2. Anti-cheat: `_is_spec_example_params` — KEEP (genuinely good honesty design)
- **What:** rejects significance-tool calls whose numeric params equal (or are a trivial scale/offset/reorder of) ANY tool's spec-example values; value-based + cross-tool; legacy hardcoded floor as backstop.
- **Why:** an agent copying `[0.9,0.88,0.91]` from a tool's doc into a real significance test would fabricate a result.
- **Assessment/tradeoff:** exactly the right kind of adversarial guard for a credibility engine; generalizes to new tools without edits. **Action:** KEEP; move alongside the central honesty gate so it applies wherever significance is judged.

### L3. Think prompt is ML-hardcoded — FIX (domain-independence violation)
- **What:** the "generic" think prompt is saturated with ML specifics: `val_losses`, `build_mlp`, `train_model`, `run_experiment_grid`, MNIST, `n_steps`, classification defaults.
- **Why:** the first domains were ML; the prompt grew around them.
- **Needed?/Better way?** The worker must be domain-general (core rule D1). The prompt should be assembled from the resolved domain plugin's vocabulary/tools, not hardcode ML. For a combinatorics hypothesis ~half the prompt is misleading noise.
- **Maintainable:** ⚠ every new non-ML domain fights the ML framing.
- **Assessment/tradeoff:** a latent domain-independence violation in the hottest worker path.
- **Action:** FIX — parameterize the think prompt by domain (inject the plugin's tool cluster + guidance; drop ML specifics into an ML-domain fragment). Fold into C2/C3.

### L4. Docker sandbox (`run_sandboxed_python`, `network_mode="none"`) — KEEP-WATCH (+ consolidate)
- **What:** model-written code runs in an isolated no-network Docker container, mem-capped, wall-clock enforced via `container.wait(timeout)+kill`; one Docker client reused per worker; base64+exec; JSON-last-line output contract.
- **Why:** untrusted codegen must be sandboxed; no-network prevents data exfiltration / cheating by download.
- **Needed?** Yes — non-negotiable for running LLM-written code.
- **Deployable:** ⚠ requires the worker to reach a Docker daemon (socket mount / DinD) — a real infra coupling and attack surface; document it.
- **Maintainable:** ⚠ a **second** sandbox exists in `math_combinatorics/discovery/sandbox_exec.py` (AST screen + restricted builtins + subprocess). Two security models = two things to keep correct.
- **Assessment/tradeoff:** the Docker approach is the stronger boundary; the AST/subprocess one is lighter but weaker.
- **Action:** KEEP the Docker sandbox; INVESTIGATE consolidating the math_combinatorics subprocess sandbox onto it (one security model).

### L5. Significance gate lives in the worker think-loop — FIX(clarify split)
- **What:** the worker forbids `stop` until a significance tool ran (correction prompts + a forced fallback significance call).
- **Why:** ensure evidence exists before a verdict.
- **Assessment/tradeoff:** conflates two things — (a) "did the worker GATHER significance evidence" (an experiment-completeness concern → legitimately the worker's) and (b) "is that evidence CONFIRMATORY" (the verdict → orchestrator's, E1). Keeping (a) in the worker is fine; (b) must move.
- **Action:** FIX — worker keeps "must produce significance evidence before returning"; the *judgment* of that evidence moves to the orchestrator (C2/C3).

---

## M. Orchestrator modules (grounded — code read 2026-07-09)

### M1. TWO execution engines: `research_loop` (rounds) + `campaign_loop` (tree) — ✅ DONE (legacy removed)
> **Resolved (commit 59a2c03):** `research_loop.py` deleted; `POST /research` +
> `ResearchConfig/Request/Response` removed from the API; `/internal/research`
> (+`InternalResearchBody`) removed from orchestrator/main.py. One engine now:
> `campaign_loop`. Legacy-specific tests removed; affected suites green.
- **What:** `research_loop.run_research_loop` (session → fixed rounds → hypothesis rows/checkpoints) and `campaign_loop.run_campaign_loop` (campaign → tree → frontier) are two independent orchestration engines. Both are exposed from `research.py` (POST @122 → research_loop; create_campaign @311 → campaign_loop) and `orchestrator/main.py` still delegates to `run_research_loop`.
- **Why:** `research_loop` is the older rounds-based session engine; `campaign_loop` is the current tree-based campaign engine.
- **How/evidence:** the frontend creates `/campaigns` (campaign_loop) and only *reads* `/sessions/*` (shared session_id); nothing in the UI drives the rounds engine.
- **Needed?** One engine. Two is legacy debt: double the surface, double the bugs, and the redesign would otherwise have to be applied twice.
- **Scalable/Maintainable:** ⚠ maintaining two loops is the classic "which one is real?" trap.
- **Assessment/tradeoff:** near-certain that `research_loop` is legacy; must confirm no live caller (orchestrator/main.py entrypoint, seed_validation, tests) before deletion.
- **Action:** INVESTIGATE → REMOVE — confirm `research_loop` has no live path, then delete it (and its `main.py` entrypoint) so the redesign targets one engine. High priority: do this BEFORE C2/C3 so we don't refactor the wrong/both engines.

### M2. Seed generation (`hypotheses.generate_ranked_hypotheses`) — KEEP (folds into the agent)
- **What:** builds a seed prompt (`_build_hypothesis_prompt`), agentically selects+reads skills (`_select_and_read_skills`), parses/repairs JSON, injects a null hypothesis (`_ensure_null_hypothesis`), guards ML-template hypotheses (`_is_ml_template_hypothesis`), ranks.
- **Why:** turn a question + prior into ranked falsifiable hypotheses.
- **Assessment/tradeoff:** solid and already uses the skill-catalog pattern (H2). The auto-injected "Null hypothesis: no falsifiable pattern…" boilerplate is where the constant-evidence nodes we saw originate — fine, but the orchestrator agent should own generation holistically in C3.
- **Action:** KEEP; subsume into the orchestrator agent loop (generation becomes a reasoned tool-using step, not a one-shot prompt).

### M3. Literature pipeline: `prior_builder.synthesize_prior_from_papers` + `answer_gate.evaluate_literature_short_circuit` — REPLACE(to tool) / KEEP(short-circuit)
- **What:** papers → LLM-synthesized `Prior` (facts/gaps), injected into seed-gen; a short-circuit that skips the campaign when literature already answers the question (cosine sim).
- **Why:** ground hypotheses in prior work; avoid re-deriving a known answer.
- **Assessment/tradeoff:** the *pre-fetch + inject* is the A3/E-adjacent anti-pattern (user's point 3: literature should be a tool the orchestrator calls, not an injected prefetch). The **short-circuit is a genuinely good idea** — but should also be a tool/skill the reasoning orchestrator invokes, not an automatic gate.
- **Action:** REPLACE the inject with a `literature_search` tool (C3); KEEP the short-circuit logic, re-expose as an orchestrator tool/decision.

### M4. `question_domain.infer_session_domain` — hardcoded ML-first keyword taxonomy — ✅ DONE (deleted)
> **Resolved (commit 59a2c03):** `question_domain.py` deleted; `intake.parse_question`
> no longer guesses a domain (`domain=""`). Routing = plugin self-detection +
> campaign tag. See D2. (Original entry below for rationale.)
- **What:** fast keyword heuristic mapping a question to a domain; comment says "v1 focus: DL/ALGO/ML first."
- **Assessment/tradeoff:** same domain-independence violation as `domain_router` (D2) and the ML think-prompt (L3): domain detection belongs in the plugins (`DomainPlugin.matches`), not a central hardcoded table biased to ML.
- **Action:** FIX — delegate detection to plugins; delete the central taxonomy. Fold into the D2 routing move.

### M5. `lifetime_knowledge` (cross-campaign policy) — KEEP-WATCH (this part IS live)
- **What:** load ACCEPTED policy for a bucket at campaign start; enrich prior + seed context from lifetime; at end propose a CANDIDATE policy (never auto-promote). Feeds the `_policy_score_multiplier` frontier nudge.
- **Why:** learn across campaigns without unsafe auto-promotion.
- **Assessment/tradeoff:** unlike the SHELVED layer05 simulators + operator_credit (§I — dead), this small policy layer *is* wired into the live loop. The "candidate, never auto-promote" safety is good. Its value is bounded by how much the frontier nudge actually helps (small).
- **Action:** KEEP-WATCH; re-evaluate once the reasoning orchestrator (C3) exists — the learned policy should feed the *reasoning*, not just a score multiplier.

### M6. `policy_analyst` — LLM rationale that "never edits" decisions — INVESTIGATE(value)
- **What:** an LLM produces narrative/predictions about policy; a deterministic engine does the actual mutation; the LLM output changes nothing.
- **Assessment/tradeoff:** decorative — cost + surface with no decision effect. Either make it real (let it inform mutation, with guards) or drop it.
- **Action:** INVESTIGATE → likely REMOVE or promote-to-real during C3.

### M7. `hypothesis_ranking` — the dependency-inversion source — FIX (A2)
- **What:** novelty (embeddings) / testability / impact / scope_fit scoring; also `strip_question_suffix` + `compute_question_relevance_score_lexical`, which **core** `campaign_synthesis.py` imports *upward*.
- **Assessment/tradeoff:** the ranking itself is fine; the upward import is the A2 layering violation.
- **Action:** FIX — move the two lexical helpers into core (they're domain-general text utilities); forbid core→services imports.

### M8. `seed_validation` — offline seed-quality eval suite — KEEP (dev tooling)
- **What:** `run_seed_pipeline_for_question` + `evaluate_suite` — a no-sandbox validation harness for seed generation (fixes.md Phase 1), with a `_NullEmitter`.
- **Assessment/tradeoff:** appears to be an offline eval/CI harness, not the live path.
- **Action:** confirm it's not on the live path; if so KEEP as dev tooling (or move under `tests/`/`scripts/`).

---

## N. Security & data governance (grounded — code read 2026-07-09)

> Framing: **building** this is a deliberately-deferred separate track (user's
> call). This section only *captures* the current posture as first-class HLD, so
> the gap is visible and not rediscovered late. Lab onboarding (clinical/PII data,
> GDPR/HIPAA) makes several of these **hard blockers**, not nice-to-haves.

### N1. No authN/authZ on the public API — BLOCKER (build before any external user)
- **What:** `deps.py` injects only infra handles; no auth dependency on create/get/list/resume campaigns or `/sessions/*` reads. Only the internal orchestrator endpoint has `orchestrator_internal_token`.
- **Why (as built):** single-user local dev.
- **Needed?** Absolutely, before any shared/hosted use. **Better way?** Per-tenant API keys or OAuth/OIDC + per-request authz scoped to the caller's org.
- **Secure:** ❌ open read/write to all data incl. stored prompts. **Assessment:** hard blocker for onboarding.
- **Action:** BUILD (deferred) — authn (keys/OIDC) + authz + rate limits.

### N2. Data egress to third-party LLMs (OpenAI/Gemini) — BLOCKER for clinical/PII data
- **What:** `LLMClient` sends prompts (hypothesis text, extracted numeric values, result summaries, possibly raw data snippets) to external provider APIs; `ollama` is the only in-perimeter option and nothing enforces it.
- **Why:** access to frontier models.
- **Needed?/Better way?** For clinical/PII data, egress must be controllable: a **local-only mode** (ollama / self-hosted), per-tenant model policy, PII detection/redaction before egress, and an explicit data-classification → allowed-provider mapping.
- **Secure:** ❌ the core GDPR/HIPAA violation risk. **Assessment:** hard blocker for regulated data; the single biggest thing labs will ask about.
- **Action:** BUILD (deferred) — enforce a per-tenant "no data leaves perimeter" mode (local models), egress guard, redaction. Highest-priority security item.

### N3. No tenant isolation — BLOCKER (multi-lab)
- **What:** one Postgres DB + one flat schema (campaigns/sessions/events/llm_calls); no org/tenant column or boundary; minio/qdrant single-namespace.
- **Needed?** Yes for multi-lab. **Better way?** Tenant id on every row + row-level security (or DB-per-tenant); per-tenant minio buckets / qdrant collections.
- **Secure/Scalable:** ❌ cross-tenant data mixing risk. **Action:** BUILD (deferred) — tenancy model end-to-end.

### N4. Data stores published to host; no encryption at rest — FIX (prod hardening)
- **What:** compose publishes postgres:5432, redis:6379, qdrant:6333, minio:9000/9001 to the host; no at-rest encryption configured.
- **Assessment:** fine for local dev, unacceptable for prod/hosted. **Deployable:** ⚠.
- **Action:** FIX (deferred) — don't publish data-store ports in prod compose/k8s; enable encryption at rest (PG TDE/volume encryption, minio SSE); network-segment the stores.

### N5. Sensitive data at rest: `llm_calls` stores full prompt+response text — FIX
- **What:** every LLM call persists `prompt_text`/`response_text` (may contain sensitive inputs) with no access control, retention limit, or encryption.
- **Assessment:** combined with N1 this is sensitive-data-at-rest wide open.
- **Action:** FIX (deferred) — retention/TTL, field-level encryption or redaction, access-scoped reads; GDPR right-to-erasure hook.

### N6. Secrets via plain env vars — KEEP-WATCH
- **What:** `openai_api_key`/`google_api_key`/`minio_secret_key`/`orchestrator_internal_token` from env.
- **Assessment:** standard-ish; upgrade to a secrets manager for prod (rotation, no plaintext in compose).
- **Action:** KEEP-WATCH → secrets manager at prod.

### N7. Sandbox is no-network — KEEP (the one strong control)
- **What:** LLM-written code runs `network_mode="none"` (§L4) → cannot exfiltrate data or phone home.
- **Assessment:** exactly right; keep and never regress. **Action:** KEEP; add a test that asserts the sandbox has no network.

### N8. No audit/retention/erasure (GDPR Art. 17/30) controls — BUILD
- **What:** the `events` table audits *actions*, but there's no data-access audit, retention policy, or subject-erasure path.
- **Action:** BUILD (deferred) — data-access audit log, retention policy, erasure API.

---

## O. Core honesty / quality backbone (grounded — code read 2026-07-09)

> **Overall verdict: the strongest part of the codebase.** A thoughtful, layered,
> domain-agnostic honesty architecture. The redesign should **relocate its
> invocation** (worker → orchestrator) and **unify entry points** — NOT rewrite
> the logic. Do not "improve" these gates casually; they encode hard-won
> anti-false-confirm lessons.

### O1. Artifact-adversarial verification (`artifact_verification`) — KEEP (crown jewel)
- **What:** confirmation must survive adversarial tests against plausible artifact models — `generate_artifact_models` (confounds/leakage/network markers), `run_adversarial_test`, `_survives_label_shuffle_lofo` / `_survives_permutation` / `_survives_panel_within_fe`, `apply_two_stage_gate` (a real claim must beat the top-k artifact explanations, and a second trivial artifact must NOT also "confirm").
- **Why:** the anti-"val_accuracy"/anti-false-breakthrough core.
- **Needed?/Better way?** Essential; well-factored. **Maintainable/Testable:** good (pure functions on evidence dicts).
- **Action:** KEEP; centralize *invocation* in the orchestrator (E1/E3). Never regress.

### O2. Scoped-claim + OOD integrity (`scoped_claim`) — KEEP (check one smell)
- **What:** every hypothesis declares where it applies + where it should transfer (OOD); `validate_scoped_claim`, `ood_similarity`, `is_boilerplate_scope`, `parse_scope_from_*`.
- **Assessment:** strong scope-discipline gate. One to check: `infer_domain_scope_template(question)` may carry ML/domain assumptions — verify it's domain-general (INVESTIGATE-lite).
- **Action:** KEEP; verify `infer_domain_scope_template` is domain-agnostic.

### O3. Evidence binding (`evidence_binding`) — KEEP
- **What:** mechanical/deterministic check that any field claiming a relationship to evidence actually references the tested subject/population/features; `extract_subject_from_{node,statement,mechanism,anomaly}`, `BindingMetrics`.
- **Assessment:** prevents ungrounded claims; deterministic and testable. **Action:** KEEP.

### O4. `research_quality` shared controls — KEEP (domain-agnostic, load-bearing)
- **What:** `compute_evidence_hash` / `compute_verification_hash` / `compute_claim_dedup_key` (the dedup + split-brain hashing), `classify_inconclusive_reason`, `is_control_hypothesis`/`infer_node_role`, `extract_theme_vector`, `compute_replication_level`, `retry_policy_for_signature`.
- **Assessment:** exactly the domain-general layer core should own; used by both the tree and the orchestrator apply-path. **Action:** KEEP.

### O5. Verdict classifiers — refine E2 (base + composition + domain overrides)
- **What:** `significance.classify_verdict` is the **base** classifier (p/effect/CI + verification scan); `verdict_pipeline` **composes** it with the artifact/ood/scope stages; `DomainPlugin.classify_verdict` are **domain overrides**; mandrake/materials adapters add their own.
- **Assessment (refined):** it's less "4 rival implementations" than "a base + a composition + domain overrides" whose *invocation path* isn't single — the worker sometimes calls a plugin override, sometimes the composed pipeline. The risk is inconsistent gating, not four copies of the same math.
- **Action:** unify so ALL verdicts flow through the composed `verdict_pipeline` (which internally may call the base + domain override) at ONE orchestrator call site (C2/C3). Confirm `significance.classify_verdict` vs the pipeline don't disagree.

### O6. Honest-output controls (`claim_grounding`, `paper_gate`) — KEEP
- **What:** `claim_grounding` matches paper prose sentences to actual trace evidence (flags ungrounded prose); `paper_gate.session_merits_paper` decides full-paper vs trace-only so a nothing-campaign doesn't get a fabricated paper.
- **Assessment:** directly addresses the "writes a paper even with zero findings" worry — a merit gate exists. **Action:** KEEP; audit that the gate + the zero-findings honest-signal (campaign_loop O3-log) are consistent.

---

## P. Literature service (grounded — design surface read 2026-07-09)

### P1. Standalone, isolated RAG service — KEEP
- **What:** `config.py` explicitly does not import propab-core; `sources/` (arxiv, pubmed, semantic_scholar, crossref, europepmc, oeis, mathoverflow), `indexer/` (postgres_store + qdrant_store + embeddings — dual store), `retriever/` (query, chunk_rag, novelty_check, gap_mapper), `extractors/` (claims, llm_claim_locator, tables), `pipeline.py` orchestration, thin `main.py` HTTP wrappers.
- **Why:** heavy multi-source RAG deserves isolation + independent scaling/caching.
- **Scalable:** good (separately scalable; qdrant for vectors). **Maintainable:** good (clean internal layering: main→pipeline→sources/indexer/retriever). **Deployable:** own Dockerfile/service.
- **Assessment/tradeoff:** the cleanest-bounded service in the system. **Action:** KEEP.

### P2. Its own Gemini LLM client (separate from core `LLMClient`) — FIX-later
- **What:** `services/literature/app/llm_client.py` is a "shared Gemini text-generation client," independent of propab-core's multi-provider `LLMClient`.
- **Assessment/tradeoff:** two LLM clients = two egress points, two retry/observability behaviours, two places to enforce a "no data leaves perimeter" policy (security N2). Justifiable for strict service isolation, but the **security egress policy must cover both**.
- **Action:** FIX-later — at minimum, govern both egress points under the same data-egress policy (N2); consider a shared client interface. Not urgent for the brain redesign.

### P3. `evaluator/` (litqa2_live, astabench, metrics) — KEEP (offline eval tooling)
- **What:** LitQA2 / AstaBench benchmark harnesses + metrics.
- **Assessment:** offline evaluation, not the live campaign path (like `seed_validation`, M8). Valuable for measuring literature quality. **Action:** KEEP as eval tooling; confirm it's not imported by the live request path.

### P4. Public surface = thin `main.py` over `LiteraturePipeline` — KEEP
- **Assessment:** no business logic in routes; clean. **Action:** KEEP; this is the surface the orchestrator's `literature_search` tool (M3) wraps.

---

## Q. Remaining core modules (grounded — code read 2026-07-09)

### Q1. `campaign.py` — `ResearchCampaign` state primitive — KEEP
- **What:** the persistent long-running research object = HypothesisTree + `BreakthroughCriteria` + measured baseline + checkpoint/resume via `research_campaigns`; `should_stop()` (budget/breakthrough).
- **Why:** one durable primitive for a campaign's lifecycle.
- **Assessment:** central, well-defined, resume-safe (baseline "never assumed"). **Maintainable/Testable:** good. **Action:** KEEP. (Its `should_stop`/breakthrough scoring is where the `direction` validation matters — already regex-guarded.)

### Q2. Paper chain (`paper_compiler` + `paper_narrative` + `paper_sections`) — KEEP
- **What:** compile evidence → LaTeX paper (`parse_evidence`, `latex_tabular_from_jsonish`, `format_stats`, `_effective_verdict`).
- **Why:** the campaign's publishable output.
- **Assessment:** uses `_effective_verdict` (consistent with the C1 single-verdict-authority) and is gated by `paper_gate.session_merits_paper` (O6) + `claim_grounding` (O) so a nothing-campaign doesn't get a fabricated paper. Honest-output path is sound. **Action:** KEEP; audit that `_effective_verdict` here matches the orchestrator's post-C1 effective verdict (avoid a 3rd verdict notion).

### Q3. `telemetry.py` — `HypothesisTrajectory` (the LIVE moat data) — KEEP
- **What:** persists one structured trajectory per hypothesis; `build_trajectories` DERIVES records from tree + event stream; **pure instrumentation** (never mutates verdict/honesty). Wired into campaign finalize.
- **Why:** the "dataset nobody else has" — cross-campaign meta-learning corpus.
- **Assessment/tradeoff (important nuance):** this is the LIVE, valuable half of the "moat" and is cleanly non-invasive. It is **distinct from the SHELVED `operator_credit`/`layer05` (§I)**, which is the *analysis/credit layer on top* and is dead. So: **data collection = alive + good; the analytics on top = shelved**. **Action:** KEEP telemetry; when C3 lands, wire the (revived) analytics on top of this existing dataset.

### Q4. `anomaly_engine` — sweep→detect→induce→seed — KEEP-WATCH
- **What:** an anomaly-detection pipeline (sweep_engine, anomaly_detector, mechanism_inducer, competing_mechanisms) feeding `anomaly_seeds` (orchestrator) — the 3rd seed source (B4).
- **Why:** surface surprising effects as hypothesis seeds.
- **Assessment/tradeoff:** wired (via `anomaly_seeds` in campaign_loop). Real machinery (1.9k LOC) whose value depends on how often anomaly seeds beat LLM seeds — measure it. Overlaps the generation concern the orchestrator agent will own (B4).
- **Action:** KEEP-WATCH; under the C3 agent, decide whether anomaly-seeding is a tool the brain calls vs an always-on parallel source.

---

## Loop protocol
1. Pick the highest-severity non-`INVESTIGATE` entry.
2. For `INVESTIGATE`, do the trace first → assign a real status here (with rationale/tradeoff).
3. Refactor (self or scoped subagent, worktree-isolated), verify (tests + live campaign), update the entry's status to KEEP/removed.
4. Repeat. The orchestrator-brain redesign (B1/C1/E1/E2/D2/G2/G3/B2/B3/H3) is the current focus.
