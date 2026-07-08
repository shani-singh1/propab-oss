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
| services/orchestrator (literature/research_loop/hypotheses/paper/retrieval/lifetime/policy_analyst/seed_validation/ranking/anomaly/ledger/prior/quality/diagnostics/budget/answer_gate/question_domain) | ~6k | ⬜ |
| services/worker/sub_agent_loop.py | 3.0k | 🟡 (verdict/confidence/routing/return read; full ⬜) |
| services/worker (think_act/permutation_null/sandbox/domain_router/failure_classify/sandbox_code_rewrite/peer_findings) | ~1.5k | 🟡 (think_act/sandbox/domain_router/peer_findings/significance read → §L; permutation_null/failure_classify/sandbox_code_rewrite ⬜) |
| services/literature (65 files) | 9.2k | ⬜ |
| core: hypothesis_tree | 0.9k | 🟡 |
| core: verdict_pipeline/significance | 0.4k | ✅ (§E, §K) |
| core: artifact_verification | 0.8k | ⬜ |
| core: campaign_synthesis | 1.0k | 🟡 |
| core: campaign / campaign_db / campaign_snapshot | ~1k | ⬜ |
| core: research_quality / evidence_binding / scoped_claim / claim_grounding | ~2k | ⬜ |
| core: paper_compiler / paper_narrative / paper_sections / paper_gate | ~1.9k | ⬜ |
| core: telemetry / telemetry_db / health_metrics / knowledge_graph / numerical_seeds | ~1.5k | ⬜ |
| domain_modules (12 domains) | 14.7k | 🟡 (genomics ✅; pattern known; rest ⬜) |
| domain_adapters / domain_profiles | ~2.1k | ⬜ |
| anomaly_engine | 1.9k | ⬜ |
| tools (registry ✅; tool impls ⬜) | 4.3k | 🟡 |
| skills | 0.3k | ✅ |
| operator_credit / layer05 | ~8.4k | ✅ (traced → SHELVED §I) |
| frontend/src | — | 🟡 (model/events/panels known from rebuild) |

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

### A3. Separate literature microservice (9.2k LOC, own FastAPI + tests) — INVESTIGATE
- **What:** a full standalone service for papers/embeddings/gaps/contradictions, called over HTTP.
- **Why:** literature/RAG is heavy (embeddings, external APIs, caching) and benefits from isolation + independent scaling/caching.
- **Assessment/tradeoff:** plausibly justified by weight, but 9.2k LOC + 65 files is large; needs a scan for dead endpoints and whether the orchestrator/worker actually use more than a thin slice. Two prior-builders already compete (E-adjacent). In the redesign it must be exposed as a **tool**, not a pre-fetch.
- **Action:** audit its public surface vs actual callers; confirm it earns its separateness; wire as a tool.

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

### D2. Domain routing decided in the worker via its own LLM + hardcoded taxonomy — REPLACE
- **What:** `domain_router.py` picks verifier/tool-cluster in the worker with a baked-in domain list.
- **Why:** worker had the hypothesis in hand.
- **Assessment/tradeoff:** a routing/strategy decision that belongs to the orchestrator, and the hardcoded taxonomy violates domain-independence. Plugins should self-detect (`DomainPlugin.matches`); orchestrator decides.
- **Action:** move routing to orchestrator (C2/C3); detection stays in plugins.

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

## Loop protocol
1. Pick the highest-severity non-`INVESTIGATE` entry.
2. For `INVESTIGATE`, do the trace first → assign a real status here (with rationale/tradeoff).
3. Refactor (self or scoped subagent, worktree-isolated), verify (tests + live campaign), update the entry's status to KEEP/removed.
4. Repeat. The orchestrator-brain redesign (B1/C1/E1/E2/D2/G2/G3/B2/B3/H3) is the current focus.
