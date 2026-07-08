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

Each entry: **What** · **Why (rationale)** · **Assessment + tradeoff** · **Action**.

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

## Loop protocol
1. Pick the highest-severity non-`INVESTIGATE` entry.
2. For `INVESTIGATE`, do the trace first → assign a real status here (with rationale/tradeoff).
3. Refactor (self or scoped subagent, worktree-isolated), verify (tests + live campaign), update the entry's status to KEEP/removed.
4. Repeat. The orchestrator-brain redesign (B1/C1/E1/E2/D2/G2/G3/B2/B3/H3) is the current focus.
