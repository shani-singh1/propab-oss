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
| Verdict: verdict_pipeline, significance | core | **deep** (V1-V4) |
| Worker: sub_agent_loop, think_act | services/worker | partial (W1-W3; _build_evidence is LOFO-shaped) |
| Artifact gate + evidence binding, scoped_claim | core | partial (A1-A3) |
| Domain layer: base defaults, registry, plugins, adapters | core | partial (D1-D3) |
| Numerical seeds, synthesis_diversity, belief_promotion | core | partial |
| Orchestrator loop: campaign_loop.py (2369) | services/orchestrator | partial (O1-O3) |
| Generation: hypotheses, seed_validation, hypothesis_ranking, anomaly_seeds | orch | partial (G1-G2) |
| Domain layer: base, registry, plugins, adapters, profiles | core | partial (D1-D3); adapters pending |
| Lifetime learning: lifetime_knowledge, knowledge_graph, meta_science, policy_* | mixed | survey (wired; L8 quality pending) |
| Paper: paper_compiler, paper_sections, paper_gate, research_quality | core | survey (P1 honest); assembly pending |
| Tools registry | core/tools | survey (T1/T2 — verify significance tools compute) |
| API + events + persistence: routes, stream, events, db, campaign_db | services | pending |
| config.py / llm.py | core | pending |
| Literature service | services/literature | **owner-known** (done, 0.77) |

Legend for the loop: pending = not yet read; survey = grepped/skimmed; partial =
key paths read; deep = traced line by line.

---

## ★ CENTRAL THESIS (cross-layer) — Propab is domain-agnostic in *architecture* but domain-LOCKED in *fact*

The single most important finding of this audit, spanning V1 · A1 · A3 · W1 · L7:
**only ~3 domains can ever produce a "confirmed" finding; every other domain
launches a campaign and silently produces nothing.** The confirmation machinery
assumes evidence is either (a) a deterministic proof (math: `verified_true_steps`)
or (b) a leave-one-family-out grouped-regression with null stats (mandrake /
materials: `lofo_r2` + `label_shuffle_null_p95`). The chain:

1. Domain defaults fail-open: `preflight()` returns `passed=True` (L7/D1) → any
   domain **launches**, even underpowered/unsupported.
2. Generic verification is unimplemented: `DomainPlugin.run_verification` raises
   `NotImplementedError` (L7/D2); the worker's `_build_evidence` is built around
   `mean_r2`/`lofo_r2` (LOFO regression) — there is no generic "run an experiment,
   get statistics" path that yields confirmable evidence.
3. Statistical evidence auto-downgrades: even if stats are produced,
   `artifact_gate_stage` forces `statistical` → `inconclusive` (V1).
4. The rigor gate rubber-stamps or trusts self-reported stats (A1) and assumes
   grouped data (A3).

⇒ A new/statistical domain: launches, confirms nothing, spends its budget on
inconclusive breadth, finalizes empty — **and every layer reports "success"** (a
non-null stop reason, health metrics logged) so nothing surfaces the failure.
This is the infrastructure-level reason campaigns "run but do no real research,"
and it is *upstream of the convergence layer I already fixed* (no confirmed
parents ⇒ nothing to deepen). **This must be fixed before any multi-domain
campaign.** Fix shape: a `DomainPlugin.confirmation_contract()` that declares what
evidence + which adversarial control confirms in *that* domain, a real generic
statistical-with-permutation-null path, and a preflight that fails-closed when the
domain can't actually be verified.

---

## LAYER 1 — Verdict pipeline (the "confirmed" decision) [deep]

Files: `verdict_pipeline.py`, `significance.py`. This decides confirmed / refuted
/ inconclusive; everything downstream (belief state, convergence, paper) trusts
it. Composition `classify_verdict_stage → artifact_gate_stage → ood_gate_stage →
scope_integrity_stage` is clean and pure. But:

**V1 · CRITICAL · FIXED (fix/confirmation-layer, merged+verified) · VERIFIED — statistical-only evidence can NEVER be
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

**V2 · HIGH · FIXED (fix/confirmation-layer, merged+verified) · VERIFIED — the "deterministic" class bypasses the artifact
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

**W1 · HIGH · FIXING (agent: fix/worker-provenance) · VERIFIED — the anti-fabrication guard is a 3-item denylist.**
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

## LAYER 4 — Artifact gate & evidence binding (the "rigor" layer) [partial]

Files: `artifact_verification.py` (753), `evidence_binding.py` (384),
`scoped_claim.py`. This is what makes "confirmed" mean "survived an adversarial
control." Reading the survival tests:

**A1 · CRITICAL · FIXED (fix/confirmation-layer, merged+verified — rubber-stamps removed, fail-closed; input-provenance remains under W1) · VERIFIED — the artifact gate does not run the null test;
it reads the worker's self-reported null statistics, and its fallbacks
rubber-stamp.** `_survives_label_shuffle_lofo` / `_survives_permutation`
(artifact_verification.py:286, 330) consume `lofo_r2`, `label_shuffle_null_p95`,
`label_shuffle_permutation_p`, `permutation_p` **from the evidence dict the worker
produced** — no shuffle/permutation is computed here. Worse, the fallback branches
are near-trivial: `_survives_permutation` returns `survived = (lofo > 0.0)` when a
LOFO number is present (line 337) — i.e. *any positive effect "survives" the
permutation null*, which is not a null test at all; `_survives_label_shuffle_lofo`
has a `lofo > 0.05 and gap < 0.85` heuristic (line 308) that passes with no
label-shuffle null, and a `lofo > -0.05 and y_perm_p < 0.05` path (line 311) whose
LOFO condition is almost always true. *Design intent (ARCHITECTURE §7):* "runs
adversarial tests (label-shuffle LOFO, permutation nulls) to decide whether a
confirmed verdict survives." *Actual behaviour:* it threshold-checks
agent-supplied statistics and, when the strong stats are absent, waves the result
through on `lofo > 0`. With W1 (the worker can fabricate inputs) this means the
core rigor guarantee is **self-attested by the same agent whose claim is being
judged**. The offline permutation-audit scripts (ARCHITECTURE §11) can re-run it
independently, but the *in-pipeline* "confirmed" — the one the tree, beliefs, and
paper use — is not independently verified. **Fix direction:** the gate must
recompute the null from raw per-sample data (require the worker to return the raw
arrays / a reproducible artifact), or at minimum FAIL-closed when the strong null
stats are absent instead of falling back to `lofo>0`.

**A2 · HIGH · CLEARED (fix/evidence-binding, verified) + instrumentation FIXED — does evidence binding reject genuine support?**
`evidence_binding.filter_node_citations` runs at write time and is credited with
"belief citation integrity." If it is too strict, beliefs never accrue supporting
nodes → beliefs stay `unclear` → (with the exhaustion logic) the branch trends to
exhausted, and rival tension that drives discriminating experiments never forms.
Needs a rejection-reason histogram from real data; the health metric warns if it
runs 50+ times with zero rejections (too loose) but nothing warns on *too strict*.
Trace `filter_node_citations`.

**A4 · HIGH · OPEN · VERIFIED — evidence binding's acceptance criterion is hardcoded biology/mandrake vocab, so beliefs can't form for other domains.** (Found while VERIFYING the A2 audit — the subagent correctly cleared "binding wrongly over-rejects for its designed inputs" but missed this.) `infer_test_targets`/`infer_population_scope` (evidence_binding.py ~40-127) match only hardcoded regexes — `_LOFO_RE`, `_REDUND_RE` (sequence identity), `_CONFOUND_RE` (plate id), `_FAMILY_CAT_RE` (evolutionary family), and `_FEATURE_TOKEN_RE` which literally lists mandrake feature names (`triad_best_rmsd`, `sp_motif_found`, `D2_D3_dist`). For ANY non-biology domain a node's text matches none → it gets no tags → `binding_check` returns `cited_node_untyped_for_citing_claim` → EVERY citation is rejected → `apply_synthesis_beliefs` sends the belief to `proposed_ungrounded_beliefs`, it stays `unclear`, and the branch trends to exhausted. So belief formation is starved for every domain outside the demo biology vocab — a CENTRAL-THESIS domain-lock, now made VISIBLE by the symmetric over-strict health warning A2 added. **Fix:** binding acceptance must use STRUCTURED finding fields (verdict, metric name/direction, claim_scope, scope_delta) — which the code currently ignores — not hardcoded text regexes, so a genuine supporter in any domain can be matched to a belief while fabricated/irrelevant nodes are still rejected.

**A3 · HIGH · PENDING · SUSPECTED — LOFO assumes grouped/family data most domains
lack.** Label-shuffle-LOFO is a leave-one-*group*-out control; it presupposes the
dataset has meaningful groups (mandrake families, material systems). A domain with
i.i.d. samples and no grouping can't produce `lofo_r2`, so per V1 it can't confirm
statistically and per A1 can't take the LOFO path — it is structurally locked out
of "confirmed." This is the domain-generality root that V1 and A1 share. Confirm
by checking whether the generic domain path can ever emit `lofo_r2`.

## LAYER 7 — Domain layer (the domain-agnostic seam) [partial]

Files: `domain_modules/base.py`, `registry.py`, plugins, `domain_adapters/*`,
`domain_profiles/*`. Core imports no domain constant — architecturally clean — but
the *defaults* decide what happens for an unsupported domain:

**D1 · CRITICAL · FIXED (fix/domain-preflight, merged+verified) · VERIFIED — `preflight()` defaults to `passed=True`.**
(base.py:170) The "fail-fast power check" that is supposed to refuse an
underpowered domain in seconds **fails open**: any domain that doesn't override
preflight launches a full campaign. So the gate protects only the domains that
least need it (materials implements it) and waves through exactly the new/unknown
domains that most need a power check. **Fix:** default preflight should fail-closed
(or at least WARN loudly) when the domain provides no verification path.

**D2 · CRITICAL · OPEN · VERIFIED — `run_verification` raises NotImplementedError
by default**, and the worker's evidence builder is LOFO-shaped (`_build_evidence`
reads `mean_r2`/`lofo_r2`, sub_agent_loop.py:429). A domain that doesn't implement
a verification path (or a LOFO adapter) produces no confirmable evidence — its
experiments error to `inconclusive`. There is no generic "run experiment → get
statistics → confirm with a permutation null" path. See CENTRAL THESIS.

**D3 · MED · FIXED (fix/domain-preflight, merged+verified — has_scope_template/has_artifact_models added) — `artifact_models`, `extract_numerical_seeds`,
`scope_template` default to empty/None.** Individually safe, but together they mean
a generic domain gets: no artifact vocabulary (gate has nothing domain-specific),
no numerical-seed compounding, no scope templates → the scope/OOD gates degrade to
no-ops. Verify each isn't silently disabling a check that then reports "passed."

## LAYER 6 — Hypothesis generation (the starting frontier) [partial]

Files: `hypotheses.py` (627), `seed_validation.py`, `hypothesis_ranking.py`,
`anomaly_seeds.py`. The seeds are the campaign's initial frontier — bad seeds ⇒
bad campaign.

**G1 · HIGH · FIXED (fix/generation-layer, merged+verified) · VERIFIED — the "domain fallback" seed generator is a
hardcoded keyword→canned-hypotheses lookup for ~5 demo topics.**
`_domain_fallback_options` (hypotheses.py:32) matches the question against literal
keyword lists (`egyptian`/`unit fraction`, `collatz`, `prime gap`,
`contagion`/`sir`/`sis`, …) and returns 3-4 fully-written hypotheses per known
topic; **any question outside these returns `[]`.** So seed quality is a cliff:
excellent for the handful of demo questions this was tuned on, empty for anything
else (falling through to `_fallback_hypothesis_text`, a generic template). This is
another facet of the CENTRAL THESIS — the system looks domain-general but is
tuned to specific demo research questions. For a genuinely new question the
campaign starts from weak/empty seeds. **Fix:** generation must not depend on a
hardcoded topic table; the LLM path + literature prior should carry unknown
domains, and the fallback should be domain-shape-driven, not keyword-driven.

**G2 · FIXED (fix/generation-layer, merged+verified — fallbacks flagged is_fallback/scope_valid=0, non-fallback boilerplate rejected) — question-relevance gate + scope-fallback honesty.** `used_fallback`
/ `scope_fallback` paths substitute template text when the LLM output fails scope
validation; verify these don't quietly inject boilerplate that then passes as a
real hypothesis (the `is_boilerplate_scope` check at line 548 suggests this failure
mode is known — confirm it's fully closed).

## LAYER 5 — Orchestrator loop (`campaign_loop.py`, 2369) [partial]

**O1 · MED · OPEN · VERIFIED — breakthrough metric extraction hardcodes ML metric
names.** `_extract_metric_from_result` (line 508) regexes for
`val_accuracy|accuracy|test_accuracy` as a fallback. For a non-ML domain (math,
physics, econ) the metric isn't "accuracy"; only the generic `metric_value` JSON
path works, and if the worker doesn't emit that exact key, breakthrough detection
misses — another ML/demo-domain assumption baked in.

**O2 · MED · FIXED (fix/loop-honesty, merged+verified) · VERIFIED — several `except Exception: pass` swallow errors
in the loop** (lines ~373, 507, 515, 733, 1369, 1404, 1469-74). Salvage/preflight
ones are intentionally best-effort (fine), but the others need a read to confirm
none hide a real failure (e.g. a synthesis or dispatch error swallowed so the
round looks empty rather than errored). PENDING targeted read.

**O3 · MED · FIXED (fix/loop-honesty, merged+verified — finalized_without_findings signal) · VERIFIED — "success" stop reasons mask zero-finding
campaigns.** A campaign that confirms nothing still finalizes with a normal
`HYPOTHESIS_CAP_REACHED` / `TIME_BUDGET_EXHAUSTED` and writes a paper — presenting
as a completed run. This is the "every layer reports success" half of the CENTRAL
THESIS at the orchestrator level. Not a crash bug, but it's why broken campaigns
looked fine. A stop-reason (or health flag) for "finalized with 0 confirmed
findings" should exist and be surfaced loudly.

## LAYER 8 — Lifetime learning [survey — wired, quality unverified]
`ingest_campaign` writes; `lifetime_context_for_seeds` (lifetime_knowledge.py:125)
is read back into seed generation (campaign_loop.py:1696). So the read-back loop
EXISTS (not write-only). **L8 open question (PENDING):** whether the injected
context measurably improves later campaigns or is inert prose; JSON last-writer-
wins is fine at one-writer-per-campaign. No wiring bug found yet.

## LAYER 11 — Tools registry [survey — appears real]
Per-domain tool dirs exist (`mathematics/`, `statistics/`, `deep_learning/`,
`materials/`, `mandrake/`, `ml_research/`, `general_computation/`) — not obvious
stubs. `model_registry.py` and `registry.py` carry placeholder-detection (the LLM
copying `"x"`/`"dummy"` model ids; spec `example_params`) — same self-attestation
theme as W1. **T1 · PENDING:** read a representative tool (e.g.
`statistics/statistical_significance`) to confirm it computes rather than echoes,
and check `registry.py:154` where `example_params` are injected as defaults (could
be the very source of the spec-example values W1 tries to detect — circular).

## LAYER 9 — Paper compiler [survey — honest, positive]
**P1 · CLEARED (positive) — the paper layer does NOT over-claim.**
`paper_compiler._effective_verdict` (line 211) applies an evidence bar before a
finding reaches the paper: a DB "confirmed" with no metric is presented as
"inconclusive" ("cannot claim without evidence"); unknown verdicts default to
inconclusive. So the paper honestly reflects the evidence bar rather than
laundering weak findings into results. This is the one layer that is *more*
conservative than its inputs — good. (Full read of section assembly still pending
for completeness, but no dishonesty found.)

## LAYER 11 continued — the load-bearing significance tool is unlocated
**T2 · CLEARED (positive) — the significance tools compute real statistics.**
They live in `tools/ml_research/` (not `tools/statistics/`):
`statistical_significance.py` uses real scipy (`stats.mannwhitneyu`,
`stats.wilcoxon`); `literature_baseline_compare.py` uses `stats.t`/`stats.norm`.
Not stubs. **But this sharpens W1/A1 rather than closing it:** the math is real,
computed on the `results_a`/`results_b` arrays *the agent passes in*. So the
integrity gap is purely INPUT PROVENANCE — a real p-value computed on fabricated
data is still fabricated science, and the only guard is the 3-item denylist (W1).
The tools are fine; the input pipeline is the exposure.

## LAYERS STILL PENDING (next passes — enumerated, not assumed clean)

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
