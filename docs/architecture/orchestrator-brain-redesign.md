# Orchestrator-as-brain redesign

> Status: **APPROVED FOR BUILD** (2026-07-09). Current architecture is to be
> **replaced and deleted** — no legacy/compat shims.

## 0. Root cause (the one architectural bug behind all the symptoms)

**The orchestrator was built as a procedural pipeline, not an agent.** It makes
discrete LLM calls (seed-gen, batch synthesis, post-hoc policy analysis) but has
**no tool-use reasoning loop** and never reasons over the tree. Every symptom we
hit is downstream of that:

- workers judge good/bad (verdict) because the orchestrator can't;
- "expand/narrow" is a hand-tuned `frontier_score` formula because nothing reasons;
- literature is pre-fetched + injected because there's no agent to *call* it;
- no tool audience scoping, one shared model, and a frontend with nothing to show
  from the orchestrator because it never thinks out loud.

The fix is singular: **make the orchestrator a real agent** (tools + skills +
reasoning loop with full tree context) and **demote workers to experimenters.**

## 1. Target design

```
question
  ▼
ORCHESTRATOR = agent (frontier model, holds the whole tree)
  · system prompt: knows its tools + skills (catalogs; grows over time)
  · instructed to call the LITERATURE tool at campaign start (papers/gaps/
    contradictions), and may call it again anytime — NOT pre-fetched/injected
  · reads skills on demand (brainstorm/think/critique/…)
  · reasons → writes seed hypotheses
  · dispatches up to `max_parallel_workers` (config), each with instructions
  ▼                                            ▲ raw result
WORKER = experimenter (cheaper model, one hypothesis)
  · own system prompt + orchestrator's per-hypothesis brief
  · has tools + skills (a subset; some are orchestrator-only)
  · designs experiment, writes & runs code, tests thoroughly
  · reports back RAW: {experiment_design, what_was_tested, raw_evidence, artifacts}
  · NEVER judges good/bad
  ▼
ORCHESTRATOR reasons over (raw result + FULL tree context)
  · runs the honesty/verdict pipeline centrally (all domains, one place)
  · decides + MARKS the node via a tool: DEEPEN | REFUTE | INCONCLUSIVE |
    RETUNE(params/data, same hypothesis) | DROP | SPAWN-RELATED
  · emits its reasoning as user-friendly events (transcript / chain-of-thought /
    tool calls / skills read)
  └──────────────── loop: worker executes next round ────────────────┘
```

## 2. Intended vs current (architecture audit)

| Axis | Intended | Current | Verdict |
|---|---|---|---|
| Orchestrator nature | **Agent**: tool-use + skills + reasoning loop | Procedural script; discrete LLM calls | ❌ rebuild |
| Verdict / good-bad | Orchestrator, central, tree context | Worker, local, no tree context | ❌ move |
| Expand / narrow / retune | Orchestrator reasons per result | `frontier_score` formula + batch synthesis | ❌ replace |
| Literature | **Tool** orchestrator calls (start + on demand) | Pre-fetched + injected at seed time | ❌ convert to tool |
| Tool access control | Audience-scoped (orch-only / worker-only / both) | Flat registry, `domain` only | ❌ add scoping |
| Skills | Both use; some orch-only; on-demand read | Only read during seed-gen | ⚠ generalize |
| Node marking | Orchestrator marks via **tool** | Worker sets `node.verdict` directly | ❌ move to tool |
| Models | Orchestrator frontier / workers cheap | Single `settings.llm_model` | ❌ split |
| Events / frontend | Simple user-facing; orchestrator reasoning stream | Lifecycle only; worker emits `llm.*` | ❌ add + simplify |

## 3. Contracts (what every builder codes against)

### 3.1 Worker → Orchestrator result (NO verdict)
```json
{
  "hypothesis_id": "...",
  "experiment_design": "what the worker chose to run and why",
  "what_was_tested": "precise operationalization",
  "raw_evidence": { "metrics": {...}, "nulls": {...}, "provenance": {...} },
  "artifacts": ["ref://..."],
  "worker_notes": "anomalies/caveats/failures — advisory, NOT a verdict"
}
```

### 3.2 Orchestrator → Worker dispatch
```json
{ "hypothesis_id": "...", "hypothesis": {...}, "instructions": "orchestrator brief",
  "tools_allowed": ["..."], "skills_available": ["..."], "round": N }
```

### 3.3 Tool/skill audience scoping
- Add `audience` to `TOOL_SPEC` and to skill front-matter:
  `"orchestrator" | "worker" | "both"` (default `"both"`).
- `ToolRegistry.get_for(audience)` / `skills_catalog(..., audience=...)` filter by it.
- Orchestrator-only examples: `literature_search`, `mark_node`, `get_node`,
  `list_frontier`, `write_hypothesis`. Worker keeps the experiment/stat/compute tools.

### 3.4 Orchestrator node tools (point 4 — tool, not parsing)
- `get_node(node_id)` → node + status + evidence + lineage.
- `list_tree()` / `list_frontier()` → tree context for reasoning.
- `mark_node(node_id, action, rationale)` where `action ∈ {deepen, refute,
  inconclusive, retune, drop, confirm}` — this is how the verdict/expansion
  decision is recorded (deterministic, not regex-parsed from prose).
- `write_hypothesis(parent_id|None, text, kind)` — seed / child / lateral.

### 3.5 Central honesty gate
`run_verdict_pipeline` + shape-aware artifact/significance gates + the
degenerate-metric guard run **once, in the orchestrator**, on `raw_evidence`, for
every domain. Domains keep their `classify_verdict`/verifiers (domain-independence
rule) — the orchestrator *invokes* them. `mark_node(confirm)` cannot succeed
unless the central gate passes.

### 3.6 RETUNE = node attempts, not new nodes
Re-running the *same* hypothesis with changed params/data is a new **attempt**
recorded on the node (`attempts: [...]`), distinct from `deepen` (which creates a
child). Bound by `max_retune_rounds_per_hypothesis`.

## 4. Frontend (points 5, 6, 8 — clean, no jargon)
- **User-facing event labels are plain language** — never `ORCH_*`. e.g.
  "Reviewing the literature", "Thinking through the result", "Testing a
  hypothesis", "Marked inconclusive — why". Internal enums may exist but the UI
  maps them to friendly copy.
- **Mid panel = orchestrator activity**: transcript / chain-of-thought / tool
  calls / skills read / decisions, grouped into **collapsible** event groups,
  color-coded by kind.
- **Right panel = workers**: a tab/card per dispatched worker (subagent); click a
  worker to view its full activity (its design, code, tools, result).
- Goal: full transparency, zero mental overhead — the mid panel reads like the
  orchestrator narrating the campaign; workers are drill-down detail.

## 5. Config (point 7)
- `max_parallel_workers` (surface existing concurrency).
- `orchestrator_model` (frontier, expensive) vs `worker_model` (cheaper) —
  replaces the single `settings.llm_model` for these two roles.
- `max_retune_rounds_per_hypothesis`.

## 6. Delete, don't deprecate
Remove — not shim — the worker-side verdict path (`run_verdict_pipeline`/
`classify_verdict`/significance-as-verdict in `sub_agent_loop.py`), the
`frontier_score`-as-decision and threshold batch-synthesis expansion, and the
literature pre-fetch/inject. No legacy code paths.

## 7. Build assignment (subagents, worktree-isolated, disjoint file sets)
- **CORE** — orchestrator agent loop + worker contract split; owns
  `services/orchestrator/campaign_loop.py`, new `orchestrator/agent_loop.py`,
  `services/worker/sub_agent_loop.py`, worker return contract; moves the honesty
  pipeline in; deletes dead worker verdict paths; wires literature-as-tool.
- **TOOLS/SKILLS** — audience scoping; owns `packages/propab-core/propab/tools/
  registry.py`, tool specs, `propab/skills/__init__.py` + skill front-matter; adds
  the orchestrator node tools (`get_node`/`mark_node`/`list_frontier`/
  `write_hypothesis`) as tool modules.
- **FRONTEND** — panel restructure + friendly event labels; owns `frontend/`.
- **CONFIG** — model split + `max_parallel_workers` + retune bound; owns
  `propab/config.py` + settings wiring.
- **AUDIT (read-only)** — independent deep pass for MORE architectural
  divergences beyond this matrix; produces a report, changes nothing.

Integration + verification (contracts in §3, event parity, live campaign) is
owned centrally after the agents land.

## 8. Verified audit findings (fold into CORE unless noted)

An independent read-only audit confirmed the following (all verified firsthand):

**Blockers → CORE**
- **Split-brain verdict.** Worker writes `hypotheses.verdict` to the DB
  (`sub_agent_loop.py:1487`); orchestrator downgrades the *tree* node
  (`campaign_loop.py:376-397`) but never writes back → DB (paper/API) and tree
  diverge. Fix: single source of truth — the orchestrator owns the verdict and
  writes tree **and** DB from one decision.
- **Domain routing in the worker.** Which verifier/tool-cluster to use is chosen
  by a worker LLM call over a hardcoded taxonomy (`domain_router.py:8-30,121`,
  called `sub_agent_loop.py:1960`). Move the routing decision to the orchestrator;
  keep detection in the plugins (domain-independence).
- **≥4 competing verdict implementations** + duplicated confidence
  (`_compute_confidence` sub_agent_loop.py:807 vs `_compute_pipeline_confidence`
  verdict_pipeline.py:92). Collapse to ONE core implementation the orchestrator
  invokes.

**Major → CORE (unless noted)**
- Orchestrator's only "judgment" is a hardcoded downgrade cascade
  (`_apply_result_diagnostics`, campaign_loop.py:318-439) duplicating
  `hypothesis_tree.update_node:383-394` — replace with the reasoning step.
- **API decides framing** — `research.py:349-382` resolves the domain plugin and
  overrides metric_name/direction/plausibility before the orchestrator runs. Move
  to the orchestrator.
- **Two literature-prior builders** (`literature.py:709 build_prior` vs
  `literature_client.py:356 build_prior_via_service`), env-selected, injected —
  replace with the single literature TOOL (§3). → CORE + TOOLS-SKILLS.
- **Lazy core→services imports** (`campaign_synthesis.py:26,749`,
  `replay_support.py:22,52`, `cli.py:98,214,299`) — smell to clean as logic moves;
  not a module-load inversion.
- **Three "what to test next" paths** (frontier `_information_gain_score`; dead
  `build_expand_prompt`/`parse_expanded_nodes`; Tier-2 synthesis) and synthesis
  itself is subordinated to ~15 deterministic reject filters
  (`campaign_synthesis.py:481-802`) — the filters decide, not the model. Collapse
  into the reasoning step; `attempts` (§3.6) doesn't exist yet.
- **No orchestrator-reasoning EventTypes** in `types.py`; frontend `events.ts`
  has no orchestrator phase. → CORE (emit) + FRONTEND (render).

**Minor**
- `policy_analyst.py:189` LLM call is decorative ("the LLM never edits"). → revisit.
- A third seed path (anomaly, campaign_loop.py:1206) parallels LLM seeds+synthesis.

Full report: `scratchpad/arch-audit-findings.md`.

## 9. CORE is sequenced, not one shot
The CORE stream is a coupled refactor; do it in verified phases, not one dispatch:
1. **C1 — one verdict + single source of truth:** collapse the competing verdict/
   confidence impls into core; orchestrator writes tree+DB from one decision (kills
   split-brain). Behaviour-preserving on a fixed evidence set.
2. **C2 — worker contract split:** worker returns raw evidence (no verdict); move
   verdict + domain routing to the orchestrator; delete dead worker verdict paths.
3. **C3 — orchestrator agent loop:** tool-use reasoning + node tools + literature
   tool; replace frontier/synthesis expansion with reasoned DEEPEN/REFUTE/
   INCONCLUSIVE/RETUNE/DROP; emit reasoning events.
4. **C4 — API cleanup:** remove framing decisions from `research.py`.
5. **C5 — FRONTEND:** panels + friendly labels consuming C3 events.
Verify (tests + live campaign) between phases.
