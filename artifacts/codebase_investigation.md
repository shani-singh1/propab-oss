# Propab — Codebase Investigation & Issue Registry (source of truth)

> **Purpose.** The single source of truth for what is broken, dishonest, or
> research-blocking across Propab, per layer. We do NOT run campaigns until the
> infrastructure each layer depends on is verified — every past cycle of "run a
> long campaign, watch it fail, guess a fix" failed because the map didn't exist.
> This file IS the map. If it misses something, we spiral again.
>
> **How to use / maintain.** Every issue has an ID, a severity, and a status.
> When you (any agent) touch an issue, update its status line — do not delete it.
> Add new issues with the next ID in that layer. Keep the *reasoning* (design
> intent vs. actual behaviour) so a future reader can re-derive the call.
>
> **Status:** `OPEN` · `FIXING` · `FIXED` · `CLEARED` (investigated, not a bug) ·
> `WONTFIX`. **Severity:** `CRITICAL` (blocks real research / silent wrong
> science) · `HIGH` (wrong for a class of domains/campaigns) · `MED` (degrades
> quality) · `LOW` (hygiene). **Confidence:** `VERIFIED` (read the code, traced
> it) · `LIKELY` · `SUSPECTED`.
>
> **Method.** Read each component; compare its docstring/name (what it was
> designed to do) to what the code actually does; find where it silently degrades,
> hardcodes a domain assumption, swallows a failure, or blocks a whole class of
> domains. Reason line by line. Loop until every file is covered.

## Coverage tracker (honest audit depth)

| Area | Files | Depth |
|---|---|---|
| Campaign spine: hypothesis_tree, campaign_synthesis, belief_state, frontier | core | **deep** (convergence layer — fixed) |
| Verdict: verdict_pipeline, significance | core | **deep** |
| Worker: sub_agent_loop, think_act | services/worker | partial (1 flaw found) |
| Evidence binding, artifact_verification, scoped_claim | core | pending |
| Numerical seeds, synthesis_diversity, belief_promotion | core | partial |
| Orchestrator loop: campaign_loop.py (2369) | services/orchestrator | partial |
| Generation: hypotheses, seed_validation, hypothesis_ranking, anomaly_seeds | orch | survey |
| Domain layer: base, registry, plugins, adapters, profiles | core | spot-check |
| Lifetime learning: lifetime_knowledge, knowledge_graph, meta_science, policy_* | mixed | pending |
| Paper: paper_compiler, paper_sections, paper_gate, research_quality | core | pending |
| API + events + persistence: routes, stream, events, db, campaign_db | services | pending |
| Tools registry | core/tools | pending |
| Literature service | services/literature | **owner-known** (done, 0.77) |

Legend for the loop: pending = not yet read; survey = grepped/skimmed; partial =
key paths read; deep = traced line by line.

---

## LAYER 1 — Verdict pipeline (the "confirmed" decision) [deep]

Files: `verdict_pipeline.py`, `significance.py`. This decides confirmed / refuted
/ inconclusive; everything downstream (belief state, convergence, paper) trusts
it. Composition `classify_verdict_stage → artifact_gate_stage → ood_gate_stage →
scope_integrity_stage` is clean and pure. But:

**V1 · CRITICAL · OPEN · VERIFIED — statistical-only evidence can NEVER be
confirmed, silently blocking every non-LOFO/non-deterministic domain.**
`classify_verdict` (significance.py) will return "confirmed" for a good
statistical result (gate passed + metric direction supports + replicated). But
`artifact_gate_stage` (verdict_pipeline.py:168) then classifies that evidence as
`"statistical"` (has p_value+metric, no `lofo_r2`/`label_shuffle_null_p95`) and
**forces it to "inconclusive"** — "no cross-group holdout available to rule out
artifact." So the ONLY evidence shapes that can survive to "confirmed" are:
(a) `deterministic` (symbolic proof / exact check / `verified_true_steps>=2`) or
(b) `lofo` (a leave-one-family-out holdout with a passing null test). A domain
whose experiments produce ordinary statistics — a regression coefficient, a
correlation, an effect size, an A/B delta — with no LOFO grouping gets **nothing
confirmed, ever**. Consequences cascade: no confirmed nodes ⇒ belief state can't
form supported beliefs ⇒ the convergence layer (deepens *confirmed* parents) has
nothing to deepen ⇒ the campaign runs, spends its whole budget on inconclusive
breadth, and finalizes with zero findings. This is very likely a core reason
campaigns "run but produce nothing" outside math/mandrake/materials. *Design
intent:* rule out artifacts before believing a result — legitimate. *Actual
effect:* only two hand-picked evidence shapes can ever pass, so the gate is a
domain filter masquerading as a rigor filter. **Fix direction:** a statistical
result with a real independent null/permutation test (not necessarily LOFO)
should be confirmable; the gate should demand *an* adversarial control
appropriate to the evidence type, not specifically LOFO. Needs a
domain-plugin-provided "confirmation evidence contract."

**V2 · HIGH · OPEN · VERIFIED — the "deterministic" class bypasses the artifact
gate on a loose, agent-influenceable trigger.** `classify_evidence_type`
(verdict_pipeline.py:43) tags evidence "deterministic" when
`verified_true_steps>0 AND verified_false_steps==0 AND (method not in
{None,'','significance'} OR verified_true_steps>=2)`, and `artifact_gate_stage`
returns a deterministic verdict **unchanged — no null test at all** (line 158).
So any evidence carrying `verified_true_steps>=2` confirms with zero adversarial
control. If the worker can set `verified_true_steps` from tool outputs the agent
influences (see W1 — the anti-fabrication guard is a 3-item denylist), a
fabricated "two checks passed" confirms a claim with no artifact gate. *Intent:* a
real proof needs no statistical null. *Risk:* "deterministic" is inferred from a
counter the agent can drive, and it is a full gate bypass. **Fix:** require the
`verification_method` to be an actual proof method (symbolic/exact/counterexample)
for the bypass — drop the bare `verified_true_steps>=2` path.

**V3 · MED · OPEN · VERIFIED — `min_metric_steps` default 2 is a silent
replication bar that many single-shot experiments can't meet.** `classify_verdict`
downgrades an otherwise-confirmable result to "needs replication" when
`n_metric_steps < 2` (or `verified_true_steps < 2`). A worker that runs one
decisive experiment (common) yields `n_metric_steps=1` → inconclusive. Whether
this is right depends on whether the worker is built to produce ≥2 metric steps
per hypothesis; if not, this silently caps everything at inconclusive. Cross-check
against worker evidence construction (LAYER 3). *Not necessarily a bug — flag to
verify the worker actually produces ≥2 steps.*

**V4 · LOW · CLEARED — OOD/scope stages are correctly no-ops when unpopulated.**
`ood_gate_stage`/`scope_integrity_stage` only act when scope/methodology or
`scope_gate_result` is present, else pass through. Reasonable; not a bug, but note
that a confirmed verdict with no scope info skips the OOD downgrade (upstream
should always supply scope — verify in the worker/generation layer).

---

## LAYER 2 — Campaign spine / convergence (search structure) [deep — FIXED]

Files: `hypothesis_tree.py`, `campaign_synthesis.py`, `belief_state.py`. This was
the first layer taken to "done" (branch `campaign-convergence`). Full history in
the git log; the issues and their fixes:

**C1 · CRITICAL · FIXED — frontier deprioritized deepening confirmed findings.**
`_information_gain_score` scored a confirmed parent as low-uncertainty (0.45) vs
inconclusive (0.85); depth reward ≤0.015. Fixed: exploit bias for scope-narrowing
children of confirmed parents + `confirmed_lineage_depth()` metric.

**C2 · CRITICAL · FIXED — dedup deleted the narrowing move.** `text_similarity`
(first-line-title dominated) rejected two narrowing steps (different regions) as
duplicates (0.96 similar), starving the frontier → campaign halted at depth 1.
Fixed: parameter-aware dedup (numeric signature + scope-line signature); rephrasings
still deduped.

**C3 · CRITICAL · FIXED — relevance gate dropped narrowing children.** Lexical
question-relevance falls as a child narrows (0.43→0.33), so the 0.35 gate rejected
every refinement. Fixed: synthesis children (already on-topic + scope-validated)
bypass the lexical threshold, inherit parent relevance for ranking.

**C4 · CRITICAL · FIXED — anti-monoculture diversity force rejected deepening.**
`forced_type` rejected 8/8 narrowing children (deepening one finding is single-
type). Fixed: deepening refinements of a confirmed parent bypass the type-diversity
force.

**C5 · HIGH · FIXED — silent candidate drops.** `validate_scoped_claim` /
relevance `continue`s incremented no metric; candidates vanished uncounted. Fixed:
`n_rejected_invalid_scope` / `n_rejected_low_relevance` counted + emitted.

**C6 · MED · FIXED — O(n²) dedup crawled long campaigns** (131-node round 16.5s).
Fixed: `real_quick_ratio` pre-filter → 6.3s.

**C7 · LOW · OPEN · VERIFIED — `expansion_passes_merit_gate` is dead code.** No
caller in the synthesis/frontier path; a would-be quality gate that never runs.
Remove or wire it. Not currently harmful (it just doesn't exist in the live path).

**C8 · LOW · CLEARED — belief ≤3 cap and branch-exhaustion are by-design**, not
the tree bug; do not "fix" by widening. `entropy_trajectory` is report-only (fed
to snapshots, does not steer) — an enhancement opportunity, not a bug.

Convergence benchmark (real code): max confirmed-lineage depth 1.1→3.2 (capability
to 7), confirmed nodes 1.3→17.3, narrowing-dedup reject 0.32→0.001.
**Caveat: C-layer convergence is moot for any domain hit by V1** (nothing confirms
⇒ nothing to deepen). V1 is upstream of everything here.

---

## LAYER 3 — Worker / executor (evidence generation) [partial]

Files: `sub_agent_loop.py` (2418), `think_act.py` (585), `sandbox_code_rewrite.py`.
Produces the raw evidence every verdict trusts. Highest-leverage remaining layer.

**W1 · HIGH · OPEN · VERIFIED — the anti-fabrication guard is a 3-item denylist.**
`think_act._is_spec_example_params` (line 350) detects an agent using placeholder
numbers only by exact-matching three hardcoded tool-spec example arrays
(`[0.9,0.88,0.91]`, `[0.1,0.2,0.15,0.18]`, `[0.42,0.44,0.41]`). Any *other*
invented numbers pass. The significance gate checks that a stat tool *ran*, not
that its inputs came from a real sandbox execution. So an LLM can feed fabricated
inputs to a real stat tool → real-looking p-value → confirmed. Evidence integrity
is the load-bearing assumption of the whole system; this doesn't enforce it.
**Fix:** require metric values to carry provenance from a sandbox-executed run;
generalize the fabrication check.

**W2 · SUSPECTED · OPEN — description-only → deterministic no-op stub** runs
in-process (not Docker) when the agent supplies a code description but no source
(sub_agent_loop.py ~2084); `_run_inline_trusted_sandbox_code` execs "trusted"
code in-process with no isolation. Need to trace what evidence a no-op stub yields
and whether it can reach a confident verdict, and the trust boundary. PENDING deep
read.

**W3 · PENDING — does the worker produce ≥2 metric steps per hypothesis?** If not,
V3's `min_metric_steps=2` bar silently makes single-shot experiments inconclusive.
Cross-layer with V3. Trace `_build_evidence` / the tool loop's step accounting.

---

## LAYERS PENDING DEEP AUDIT (loop continues — do not treat absence as clean)

Each gets the same four-lens read (bugs / arch flaws / dishonest components /
domain-generality blockers). Nothing below is cleared yet.

- **L4 Evidence binding + artifact_verification + scoped_claim** — does
  write-time citation filtering reject genuine support (beliefs never form)? Is
  the artifact gate's null test real or a rubber stamp? Does LOFO assume grouped
  data that generic domains lack (ties to V1)?
- **L5 Orchestrator `campaign_loop.py` (2369 lines)** — frontier refill, stop
  reasons, preflight, health logging, the `lifetime_context_ref` crossings path
  (recently committed). Where does it swallow errors or finalize with a stop
  reason that hides a real failure?
- **L6 Generation** (`hypotheses.py`, `seed_validation.py`, `anomaly_seeds.py`,
  `hypothesis_ranking.py`) — duplicate rate, question-relevance gate calibration,
  scope enrichment honesty.
- **L7 Domain layer** (`domain_modules/*`, `domain_adapters/*`, `domain_profiles/*`)
  — do the "generic" defaults actually run science, or return safe empties that
  make an unsupported domain look supported? (dishonesty hot-spot.)
- **L8 Lifetime learning** (`lifetime_knowledge.py`, `knowledge_graph.py`,
  `meta_science.py`, `policy_*`) — JSON last-writer-wins; does prior knowledge
  improve later campaigns or just accumulate? Is any of it read back and used?
- **L9 Paper** (`paper_compiler.py`, `paper_sections.py`, `paper_gate.py`,
  `research_quality.py`) — does the paper reflect the real trace, or can it
  present inconclusive/absent findings as results?
- **L10 API + events + persistence** (`routes/*`, `stream.py`, `events.py`,
  `db.py`, `campaign_db.py`) — SSE/event integrity, resume correctness.
- **L11 Tools registry** (`tools/*`) — are the STEM tools real computations or
  spec stubs? (ties to W1.)
- **L12 config.py / llm.py** — ret/timeout defaults, provider fallbacks that
  silently degrade.

---

*Audit loop status: LAYERS 1–2 deep; 3 partial; 4–12 pending. Continue reading
files, add issues with IDs, keep this the source of truth.*
