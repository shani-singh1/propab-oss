# Propab — Codebase Investigation & Failure-Pattern Map

> **Living document.** Maps the architecture as actually wired, questions the
> load-bearing assumptions, and ranks the flaws that plausibly explain the
> months-long "campaigns run but don't do real frontier science" pattern.
> Method: read `ARCHITECTURE.md` (intended design) then trace the actual code on
> the campaign spine. Each finding is tagged **[VERIFIED]** (I read the code and
> confirmed), **[LIKELY]** (strong code evidence, not yet runtime-confirmed), or
> **[AUDIT-PENDING]** (flagged, not yet deeply read).
>
> Started 2026-07-06 by Agent 3 after finishing the literature layer.

---

## 0. TL;DR — where the frontier-science failure actually lives

The system is unusually well-architected on paper (four services, a domain-
plugin seam, write-time evidence binding, a non-null stop-reason discipline,
eight health metrics). The failure is **not** "no structure exists." It is that
the **search does not converge**: the hypothesis tree grows in node count and,
now, in depth, but the parent→child edges and the frontier-selection policy do
not reliably turn a confirmed/refuted finding into a *narrower* next test.

Ranked, the things most likely to keep Propab from doing real science:

1. **[VERIFIED] Frontier selection deprioritizes deepening confirmed findings.**
   `_information_gain_score` scores a *confirmed* parent as low-uncertainty (0.45)
   vs *inconclusive* (0.85), and the depth/lineage reward is ≤0.015 — so the
   policy spends dispatches on inconclusive **breadth**, not confirmed **depth**.
   A direct, code-level cause of "generation increases, depth does not." (§3.3)
2. **[VERIFIED] Semantic lineage is inferred, not derived.** The flat-tree bug
   (all candidates `parent_id=None`) is now structurally fixed, but the fix
   assigns a parent by *text similarity + verdict preference*, not by the
   candidate actually being a refinement of that parent. Depth > 0 is restored;
   whether each edge means "this narrows that" is not guaranteed. (§3.2)
3. **[VERIFIED] "Belief binding rejects hundreds" is by design, not a bug** —
   the belief state is a ≤3-rival posterior *summary*, not the search state.
   Candidates still enter the tree. Do not "fix" this by widening the cap. (§3.4)
4. **[AUDIT-PENDING] Worker experiment quality** (`sub_agent_loop.py`, 2418 lines)
   — the actual experiments that produce evidence. If the evidence is weak/noisy,
   no amount of tree structure helps. Not yet read. (§4)
5. **[AUDIT-PENDING] Verdict + artifact gate calibration** — if "confirmed" is
   too easy or too hard, the tree either chases noise or never gets a parent to
   refine. Not yet read. (§4)

**Recommended next layer to improve (after literature): the orchestrator
campaign search loop — specifically convergence** (semantic lineage + frontier
selection). Rationale in §6.

---

## 1. Service topology (as wired)

Four app services + infra (`docker-compose.yml`), matching `ARCHITECTURE.md §2`:

- **api** (`services/api/`) — FastAPI entry; creates/loads campaigns, persists the
  row, delegates to orchestrator via `POST /internal/campaign` (or in-process
  BackgroundTask fallback if `ORCHESTRATOR_URL` unset).
- **orchestrator** (`services/orchestrator/`) — owns the campaign lifecycle.
  `run_campaign_loop` (campaign_loop.py, **2369 lines** — the biggest single
  file and the spine). Survives API restarts.
- **worker** (`services/worker/`) — Celery pool; `run_sub_agent_async`
  (sub_agent_loop.py, **2418 lines**) runs ONE hypothesis's think-act experiment.
- **frontend** (`frontend/`) — Vite/React dashboard.
- infra: postgres (state + append-only events), redis (pub/sub + broker), qdrant
  (lit vectors), minio (objects), alembic migrate one-shot.

The API↔orchestrator split is a genuine strength: an in-flight campaign is not
tied to the HTTP process.

## 2. Core package map (`packages/propab-core/propab/`, ~12.5k LoC)

The domain-agnostic engine. Load-bearing modules on the spine:

| Module | LoC | Role on the spine |
|---|---|---|
| `hypothesis_tree.py` | 797 | the tree + frontier; `add_seeds` (flat) vs child expansion (depth+1) |
| `campaign_synthesis.py` | 744 | Tier-2 refill: resolve parent → add candidates as children/roots |
| `campaign.py` | 463 | `ResearchCampaign`, belief_state, stop reasons |
| `belief_state.py` | 275 | ≤3 rival beliefs; evidence binding at write time |
| `verdict_pipeline.py` | 271 | classify → artifact gate → OOD → scope integrity |
| `artifact_verification.py` | 753 | null/permutation adversarial tests on "confirmed" |
| `evidence_binding.py` | 384 | citation relevance at write time |
| `numerical_seeds.py` | 365 | math compounding: crossings → lifetime context (just committed) |
| `scoped_claim.py` | 528 | OOD/scope gates + `validate_expansion_child` |
| `entropy_trajectory.py` | — | convergence signal (AUDIT-PENDING — is it *used* to steer?) |

Domain seam: `domain_modules/` (plugins + registry), `domain_profiles/` (gate
config), `domain_adapters/` (experiment runners). Core imports no dataset/feature
/threshold name directly — verified by spot-checks; this seam looks clean.

---

## 3. The campaign spine — deep dive

### 3.1 Intended loop (ARCHITECTURE.md §3)
`intake → literature prior → baseline → while not should_stop(): {frontier empty
→ seeds or Tier-2 synthesis; dispatch workers; update tree + beliefs} → finalize
(non-null stop reason) → lifetime ingest → paper`.

### 3.2 Lineage: the flat-tree bug is structurally fixed — but parenthood is *inferred* [VERIFIED]
The original failure (all synthesis candidates enter via `add_seeds` with
`parent_id=None, depth=0`, so the tree never gains depth) is addressed in
`campaign_synthesis.apply_synthesis_to_frontier`:
- `_resolve_synthesis_parent(item, raw_text, tree, eligible_parents)` returns a
  parent for **every** candidate when any completed/eligible parent exists:
  1. explicit id from the candidate (`_candidate_parent_ids` accepts many key
     aliases: `parent_id`, `refinement_of`, `discriminates_node_ids`, …),
  2. else **inferred** — rank eligible parents by (verdict preference for the
     candidate's `expansion_type`, generation, depth, −child_count) **+ text
     similarity** and take the top.
- resolved `parent_id` is written into the entry dict; candidates then split into
  `root_dicts` (no parent → `add_seeds`, flat) vs `child_dicts` (→ attached with
  `parent_id=parent.id, depth=parent.depth+1, lineage_length+1`).
- child scope gets `compute_scope_delta(parent_scope, child_scope)`.

**So depth > 0 is now created.** `tests/test_synthesis_diversity.py` +
`test_q1_steering.py` exercise the crossing/lineage path (18 tests pass).

**The load-bearing assumption this leaves unproven:** an *inferred* parent (top
text-similarity among completed nodes) is treated as the node the candidate
*refines*. Text similarity is not derivation. A candidate can be attached under a
parent it doesn't actually narrow, producing a tree that has structural depth but
whose edges don't encode "child reduces the uncertainty left open by parent."
Convergence (log-style narrowing) needs the latter. **This is the #1 thing to
verify/fix in the next layer.** Concretely: the synthesis prompt should make the
LLM name the `parent_id` it is refining and state the specific uncertainty the
child closes, and the code should prefer explicit-derivation over similarity-
inference (fall back to inference only when the LLM declines).

### 3.3 Frontier selection *structurally deprioritizes deepening confirmed findings* [VERIFIED]
`next_dispatch_candidate` / `next_batch` sort the frontier by
`_information_gain_score` (hypothesis_tree.py:663) = `info_gain × closure`, where
`info_gain = 0.30·relevance + 0.25·novelty + 0.25·parent_uncertainty +
0.10·coverage + 0.10·lineage_bonus`. Two components fight convergence:

- **parent_uncertainty inverts the exploit signal.** It is hard-coded by parent
  verdict: `inconclusive → 0.85`, `refuted → 0.70`, `confirmed → 0.45`. So a
  child of a **confirmed** node scores *lower* than a child of an inconclusive
  one. That is defensible for pure information-gain exploration, but it is the
  **opposite** of convergence: the "confirmed parent → targeted child → narrower
  region" deepening the user wants is exactly what this de-ranks. The frontier
  keeps preferring inconclusive *breadth* over confirmed *depth*.
- **lineage_bonus is too weak to compensate.** `min(0.15, lineage·0.03)` at 10%
  weight ⇒ at most +0.015 to the final score. Depth is barely rewarded.

**This is a concrete, code-level explanation for "generation increases, depth
does not."** Even with §3.2's structural lineage fixed, the *selection policy*
won't spend dispatches deepening a confirmed finding, so confirmed lineages stay
shallow. Fix direction: add an explicit **exploit/convergence term** — once a
node is confirmed, its children (which *narrow* its scope) should be *boosted*,
not penalised; make the confirmed-parent uncertainty contribution a function of
*residual* open uncertainty (scope not yet closed), not a flat 0.45; and raise
the effective weight on lineage/scope-narrowing. `entropy_trajectory.py` still
unverified as a *steering* (vs reporting-only) signal.

### 3.4 Belief state: ≤3 rivals, rejection is expected [VERIFIED]
`belief_state.apply_synthesis_beliefs` caps active beliefs at
`MAX_ACTIVE_BELIEFS=3` (2 in rival-exhaustion mode) and rejects candidates on
falsifiability failure, belief-cap, or ungrounded-citation (after
`evidence_binding.filter_node_citations`). "Accepts 0, rejects hundreds" is the
*designed* behaviour of a small posterior summary — **not** the tree bug and
**not** to be fixed by widening the cap. The tree, not the belief list, is the
search state. (This directly refutes one of the reported "wrong" bullets.)

### 3.5 Evidence binding at write time [VERIFIED design, AUDIT-PENDING calibration]
`apply_synthesis_beliefs` filters citations *before* persist (covered by
`test_synthesis_pass_rejects_fabricated_citation_before_persist`). Good
discipline. Open question: is `filter_node_citations` so strict that genuinely-
supporting nodes are dropped (→ beliefs never form → no rival tension to drive
discriminating experiments)? Needs a rejection-reason histogram from a real run.

---

## 4. Layers not yet deeply read (AUDIT-PENDING, ranked by leverage)

1. **Worker / executor** (`services/worker/sub_agent_loop.py` 2418, `think_act.py`
   585) — the experiments that produce evidence. Highest leverage after the tree:
   garbage evidence ⇒ garbage verdicts ⇒ nothing to refine. Check: tool-loop
   termination, evidence quality, timeout→inconclusive rate.
2. **Verdict pipeline + artifact gate** (`verdict_pipeline.py`,
   `artifact_verification.py` 753) — is "confirmed" calibrated? Too-easy ⇒ tree
   chases noise; too-hard ⇒ no parents to expand.
3. **Seed / hypothesis generation** (`services/orchestrator/hypotheses.py` 627,
   `generate_ranked_hypotheses`) — duplicate rate, question-relevance gate.
4. **Lifetime learning** (`lifetime_knowledge.py` 356) — JSON last-writer-wins;
   does prior knowledge actually improve later campaigns, or just accumulate?
5. **Paper compiler** (`paper_compiler.py` 644) — does the paper reflect the real
   trace (already claimed) — spot-check for fabrication.

## 5. Cross-cutting observations
- **Windows/OneDrive dev reality**: heavy artifacts churn; gitignore now covers
  ad-hoc scripts + `*.tsbuildinfo`; local artifacts pruned (logs/pollers/tails).
- **Health metrics are computed but are they *acted on*?** Eight metrics land in
  Postgres. Convergence failure would be visible as depth/lineage stagnation —
  is any metric wired to *stop or steer* a stuck campaign, or only logged?
  (AUDIT-PENDING — this is the "debug by which number is out of range" promise;
  verify a number actually gates behaviour.)

## 6. Recommendation — the next layer to improve

**Orchestrator campaign search loop → make the tree converge, not just grow.**

Why this over the worker: the reported months-long symptom is precisely
"generation increases, depth does not; more roots, not narrower tests." The
structural half is fixed; the *semantic* half (§3.2/§3.3) is the missing piece,
and it sits in code I own the context for now (synthesis + tree). It is also
measurable: a good fix shows up as rising mean depth-of-confirmed-lineage and a
falling open-uncertainty metric across generations, on a real campaign.

Concrete plan for that layer (to be validated before/while implementing):
1. Make synthesis **derive** parenthood: prompt requires each candidate to name
   the `parent_id` it refines + the specific open uncertainty it closes; code
   prefers explicit derivation, uses similarity-inference only as fallback, and
   *logs* the explicit-vs-inferred ratio as a health metric.
2. Make frontier selection **exploit**: bias `next_dispatch_candidate` toward the
   child that most narrows a confirmed parent's residual uncertainty.
3. Add a **convergence metric** (mean confirmed-lineage depth; open-uncertainty
   trend) and surface it so a stuck campaign is visible and steerable.
4. Validate on a real math_combinatorics campaign (the numerical-crossing path
   just committed is the cleanest place to measure depth growth).

> Next step in this doc: line-by-line read of `next_dispatch_candidate` and the
> synthesis candidate-generation prompt to confirm §3.3 before writing code.
