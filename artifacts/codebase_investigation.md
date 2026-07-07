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

**W1 · HIGH · FIXED (fix/worker-provenance, merged+verified — guard generalized to all tool specs by value; stat_input_provenance now recorded) · VERIFIED — the anti-fabrication guard is a 3-item denylist.**
`think_act._is_spec_example_params` (line 350) detects an agent using placeholder
numbers only by exact-matching three hardcoded tool-spec example arrays
(`[0.9,0.88,0.91]`, `[0.1,0.2,0.15,0.18]`, `[0.42,0.44,0.41]`). Any *other*
invented numbers pass. The significance gate checks that a stat tool *ran*, not
that its inputs came from a real sandbox execution. So an LLM can feed fabricated
inputs to a real stat tool → real-looking p-value → confirmed. Evidence integrity
is the load-bearing assumption of the whole system; this doesn't enforce it.
**Fix:** require metric values to carry provenance from a sandbox-executed run;
generalize the fabrication check.

**W1b · CRITICAL · FIXED (fix/w1b-provenance-enforce, verified — pending merge) — provenance is now ENFORCED.** Fix (`verdict_pipeline.py:174`, inside the `evidence_type=="statistical"` branch, BEFORE the artifact gate): when `evidence["stat_input_provenance"]=="agent_literal"` the statistical confirm fails closed → inconclusive (`stat_inputs_agent_literal_untrusted`). Scoped strictly to the statistical branch — deterministic-proof and LOFO paths (which compute in-sandbox) are untouched. Policy: fail-closed ONLY on the KNOWN-untrusted `agent_literal`; `computed`/`unknown`/absent proceed (absence ≠ fabrication; blocking `unknown` would falsely downgrade legit legacy/third-party paths). **Verified by me:** guard confirmed imported from the worktree core (not main checkout); 56 verdict/artifact/provenance tests pass; my own independent adversarial run through the real `artifact_gate_stage` with a PASSING null (perm_p=0.001, n=160) → `agent_literal`=inconclusive, `computed`/`unknown`/absent=confirmed. Together with D2/A1 this closes the loop: a statistical result confirms only with a real null at n≥100 AND non-fabricated inputs. Was: recorded but read by nothing (repo-wide grep confirmed). The round-2 fix records `evidence["stat_input_provenance"]` = computed/agent_literal/unknown and `inputs_from_sandbox`, and generalizes the spec-example guard by VALUE (closes the 3-array denylist), but the signal is ADVISORY: the verdict pipeline does not yet refuse to confirm on `agent_literal` inputs. Also it's a value-fingerprint match (not true taint) and only covers the think-act/heuristic worker paths (the mandrake/materials/plugin fast paths return before it). **Fix (small, core):** in `verdict_pipeline` (or the artifact gate), treat `stat_input_provenance=="agent_literal"` as fail-closed for a statistical confirm, so a p-value computed on agent-typed numbers cannot confirm. Pairs with A1.

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

**A4 · HIGH · FIXED (fix/binding-general, merged+verified — structured overlap requires shared mechanism/feature id OR scope subject term OR ≥2 salient content terms, `_MIN_SHARED_SALIENT_TERMS=2`; biology tag path preserved; 17 tests) — evidence binding's acceptance criterion WAS hardcoded biology/mandrake vocab, so beliefs couldn't form for other domains.** (Found while VERIFYING the A2 audit — the subagent correctly cleared "binding wrongly over-rejects for its designed inputs" but missed this.) `infer_test_targets`/`infer_population_scope` (evidence_binding.py ~40-127) match only hardcoded regexes — `_LOFO_RE`, `_REDUND_RE` (sequence identity), `_CONFOUND_RE` (plate id), `_FAMILY_CAT_RE` (evolutionary family), and `_FEATURE_TOKEN_RE` which literally lists mandrake feature names (`triad_best_rmsd`, `sp_motif_found`, `D2_D3_dist`). For ANY non-biology domain a node's text matches none → it gets no tags → `binding_check` returns `cited_node_untyped_for_citing_claim` → EVERY citation is rejected → `apply_synthesis_beliefs` sends the belief to `proposed_ungrounded_beliefs`, it stays `unclear`, and the branch trends to exhausted. So belief formation is starved for every domain outside the demo biology vocab — a CENTRAL-THESIS domain-lock, now made VISIBLE by the symmetric over-strict health warning A2 added. **Fix:** binding acceptance must use STRUCTURED finding fields (verdict, metric name/direction, claim_scope, scope_delta) — which the code currently ignores — not hardcoded text regexes, so a genuine supporter in any domain can be matched to a belief while fabricated/irrelevant nodes are still rejected.

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

**D2 · CRITICAL · FIXED (fix/generic-verification, merged+verified) — the generic
worker path now produces a real, confirmable statistical null.** Was: `_build_evidence`
is LOFO-shaped (reads `mean_r2`/`lofo_r2`), so a domain without a LOFO adapter had
no "run experiment → get statistics → confirm with a permutation null" path and
errored to `inconclusive` forever. Fix (Agent G, `services/worker/permutation_null.py`
+ wiring in `sub_agent_loop.py`): when the agent runs a two-group significance
comparison on real outcome arrays (`results_a/results_b`, `treatment/baseline`),
a genuine label-permutation null (`|mean(a)-mean(b)|`, ≥1000 perms, unbiased
`(ge+1)/(n+1)` p, fixed seed) is computed from the SAME arrays and attached as
`permutation_p`+`n_samples`, which the merged A1 gate (`_survives_permutation`)
requires (`p<0.01` at `n≥100`). **Integrity (verified by reading + independent
end-to-end run against the real core gate):** null is only ever written by the
permutation fn from arrays captured via `extract_two_group_arrays(call_params)`
(both-arrays-or-`None`, bools/short/mixed rejected → fail-closed); group-mean
`metric_value` filled only when no real metric exists (never overwrites, flagged
`metric_from_permutation_groups`); `stat_input_provenance` left intact for W1b;
LOFO fast paths `return` before the attach (untouched). My independent check:
real effect n=160→p=0.0005→**confirms**; no effect→p=0.36→refuses; real effect
n=40→refuses (n<100); no null→refuses. Tests: 16 new + core integrity green.
Closes the statistical-evidence arm of the CENTRAL THESIS. **NB → bumps W1b:**
generic domains can now confirm on real *or* agent-typed arrays until W1b enforces
the provenance tag — W1b is now the last gate keeping fabricated inputs out.

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

**O1 · MED · FIXED (fix/o1-general-metric, verified — pending merge) — breakthrough
metric extraction WAS hardcoded to ML metric names.** Correction: the fn is
`_extract_primary_metric_from_worker_result` (campaign_loop.py; the registry's
`_extract_metric_from_result` name was wrong — it doesn't exist). It regexed
`val_accuracy|accuracy|test_accuracy`, so a non-ML domain (math/physics/econ) whose
worker didn't emit the exact `metric_value` key missed breakthrough detection.
Fix: new `_find_declared_metric` extracts the campaign's DECLARED `metric_name`
first (own key → `metric_value`+matching `metric_name` → common sub-dicts
`metrics/scores/results/evidence/output`); ML accuracy fallbacks fire ONLY when
`_is_accuracy_metric(metric_name)`; otherwise fail-closed `None` (never substitutes
a differently-named value). Core `is_breakthrough` was already declared-metric-aware
(`finding.get("metric_value") or finding.get(self.metric_name)`) — no core edit
needed. **Verified by me:** helpers confirmed imported from the worktree; 15 tests
pass (non-ML metric drives `is_breakthrough`; ML `val_accuracy` regression intact;
fail-closed on missing/wrong-name metric). One pre-existing unrelated failure
(`test_gold_corpus_enforcement`, empty gold-corpus data file) — the subagent
verified via `git stash` it fails identically on base; to be re-confirmed by the
post-merge full suite.

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

## LAYER 12 — Config, core-state, sandbox [survey, self-hunt pass]

**CFG1 · LOW · OPEN · VERIFIED — committed defaults don't match the intended deployment.** `config.py`: `llm_provider="openai"`/`llm_model="gpt-4o"`, `embed_provider="openai"`/`text-embedding-3-small`, and sandbox default image falls back to `python:3.11-alpine` — while the actual deployment runs Gemini + the `propab-oss-worker` image (env-overridden in `.env`). Harmless when env is set, but a fresh/mis-configured run silently uses gpt-4o (needs an OpenAI key) or a numpy-less alpine sandbox (experiments ImportError). Align defaults or fail-loud if the expected provider/image isn't configured.

**CAM1 · CLEARED — `campaign.recount_from_tree` is correct.** It counts `verdict=="confirmed" and node_role != CONTROL`; `node_role` is a dataclass field defaulting to `DISCOVERY` and is always set by `add_seeds`/synthesis, so the `getattr(default=CONTROL)` never fires for real nodes. `total_confirmed` (the paper/O3 count) is honest.

**BP1 · CLEARED — belief trend-promotion is evidence-gated.** `belief_promotion.try_trend_promotion` requires the domain threshold to allow it, `>= requires_supporting_nodes` (default 3) CONFIRMED metric nodes, and a consistent monotonic trend — it cannot promote a belief without confirmed evidence. (Confirmed now means a real null passed, post round 1.)

**SBX1 · CLEARED — the sandbox is real.** `sandbox.run_sandboxed_python` runs in an isolated Docker container (no network) and returns explicit `image_not_found`/`docker_api`/`docker_timeout` errors — no silent in-process fake. The in-process `_run_inline_trusted_sandbox_code` path is scoped to the deterministic JSON-printer 'trusted stubs' (W2 — verify that classifier's boundary can't be tricked into running arbitrary code un-isolated).

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
  `research_quality.py`) — **SURVEYED (read-only): honest.** `_effective_verdict`
  (paper_compiler.py:211) is the single source of truth for outcome counts and
  downgrades before the reader: 0 steps → `unexecuted` (excluded); `confirmed`
  with no metric AND no verified-true-steps → `inconclusive`; a `confirmed`
  CONTROL hypothesis → `inconclusive`; plus `_enrich_finding_row`→None /
  `paper_eligible_finding` filtering (P5.1). So absent/unsupported "confirmed"
  verdicts cannot be presented as results. Residual **PAPER1 · LOW** — the LLM
  writes prose over the (correctly-bucketed) findings and could soft-embellish
  within an already-gated confirmed finding; counts/verdicts themselves are
  deterministic. Not opened as an actionable issue (low, structural gate sound).
  **UPDATE 2026-07-07 — REWORKED (feat/paper-research-quality, merged+verified):** the
  paper is no longer an experiment-log. New `paper_narrative.py` + reworked
  `paper_compiler`/`paper_sections`/`paper.py` emit real sections (Abstract, Intro,
  Methods w/ verification protocol, Results w/ summary-counts + findings tables,
  matplotlib figures, a Research-Narrative chain-of-reasoning from the real trace,
  Discussion + threats-to-validity). Honesty PRESERVED + TESTED — `_effective_verdict`
  stays the single source of truth; figures/tables/narrative read the SAME gated
  `findings`; a regression test proves an inconclusive/control/absent finding appears in
  NO table, figure, or narrative. Verified by me: 37 paper tests pass on worktree core.
  Residual: no local `pdflatex` end-to-end PDF check (validated at LaTeX-string + PNG).
- **L10 API + events + persistence** (`routes/*`, `stream.py`, `events.py`,
  `db.py`, `campaign_db.py`) — SSE/event integrity, resume correctness.
  **PARTIALLY SURVEYED → L10-R1 opened (below).**

**L10-R1 · MED · OPEN · SUSPECTED — resume rebuilds the tree from a JSON snapshot
and never reconciles node verdicts against the authoritative `hypotheses` table.**
Verified facts: (1) `db_save_campaign` persists `campaign.hypothesis_tree.to_dict()`
as a `hypothesis_tree_json` snapshot (campaign_db.py:136); `db_load_campaign` rebuilds
the tree ONLY from that snapshot (`Campaign.from_dict`, :207). (2) The WORKER writes
`hypotheses.verdict` DIRECTLY (`UPDATE hypotheses SET … WHERE id=:id`,
sub_agent_loop.py:822); there is NO orchestrator-side verdict write. (3) `campaign_resume.py`
backfills belief state from events and validates readiness but does NOT read
`hypotheses.verdict` rows back into the resumed tree. **Consequence:** if the
orchestrator crashes AFTER a worker's DB verdict-write but BEFORE it collects+applies+
snapshots, the resumed tree has that node stale-pending while the DB row (and the
paper, which reads the `hypotheses` table via `_fetch_result_rows`) has the verdict —
so the campaign loop (convergence/synthesis/stop) transiently operates on a tree
missing completed results, and the tree vs paper can disagree. **VERIFIED self-healing → downgraded MED→LOW-MED.**
Traced the resume path fully: `resume_warm` (campaign_loop.py:1601-1637) loads the
snapshot tree + backfills belief state but does NOT reconcile node verdicts from the
DB. The in-flight set is derived from the IN-MEMORY `pending` list (`pending=[]` at
:1222, `inflight={p["nid"] for p in pending}` at :1252), which is EMPTY on a fresh
resumed process — so `next_dispatch_candidate(inflight={})` re-selects every
tree-pending node, INCLUDING one the pre-crash worker already verdicted in the DB, and
re-dispatches it. The worker's `UPDATE hypotheses SET … WHERE id=:id` (:822) is
unconditional, so the row is overwritten. **Net:** no stranded node, no silent wrong
answer (the paper reads the healed DB state); the divergence is transient and closes
on re-dispatch. **Residual real cost (LOW-MED):** (a) wasted recompute of the in-flight
wave that completed just before the crash; (b) possibly duplicate/orphaned
`experiment_steps`/evidence rows from the discarded pre-crash run; (c) a nondeterministic
re-run can yield a different verdict than the (uncommitted) pre-crash one — acceptable
since the pre-crash verdict never entered the tree. **Clean fix (if we invest):** on
`resume_warm`, reconcile the snapshot tree against the `hypotheses` table — for each
node whose DB row already carries a terminal verdict the tree lacks, apply it via
`update_node` before dispatching, so completed work isn't recomputed and step rows
aren't duplicated. Not a campaign-blocker; batch with other L10 items.
## LAYER 11 — Tools (audited 2026-07-07; the dishonesty hot-spot) [deep]

Auditor mapped all 30 tool modules; I verified the top findings by reading the code.
**The significance/statistics tools are REAL and correct** (scipy t-test/Wilcoxon/
Mann-Whitney, real bootstrap, correct Cohen's d — W1/W1b guards hold). The rot is in
the deep-learning / algorithm-optimization tools, several of which fabricate
measurement-shaped output.

**TOOL1 · CRITICAL · FIXED (fix/tool1-real-eval, merged+verified: read diff + 7 new tests + 54 tool-suite pass; train_model persists state_dict+held-out split, evaluate_model bootstraps REAL eval over held-out set or fails closed) — WAS: `evaluate_model` fabricated the exact
variance the significance gate consumes, with `computed` provenance that W1b trusts.**
`tools/deep_learning/evaluate_model.py:68-72`: `eval_losses` = one stored
`final_val_loss` + `torch.randn(n)*max(0.002, val*0.02)` (manufactured 2% jitter), and
the summary literally says "use with statistical_significance." Fed to
`statistical_significance` these pass the zero-variance guard and yield tight
"significant" CIs. Because the jitter is computed IN-SANDBOX, `stat_input_provenance`
= `"computed"` (trusted) → the W1b guard (which only blocks `agent_literal`) does NOT
catch it → **a DL hypothesis can confirm on fabricated variance.** This partially
defeats the V1/A1/W1b confirm-gate honesty work for the ML domain. Fallback path
(:98-121) is also fake: rebuilds the net from `dims` with RANDOM init (no `state_dict`
is ever persisted by `train_model`) and evaluates on RANDOM labels. **Fix:** run real
independent eval passes on persisted weights + held-out data, or stop emitting
`eval_losses` and don't advertise them for significance. **This is the top L11 fix.**

**TOOL2 · HIGH · VERIFIED — `inspect_gradients` reports gradients of a fresh RANDOM
net.** `evaluate_model.py:98-123` + `inspect_gradients.py:33-46`: both rebuild from
`dims` on random-init weights because `train_model` never writes a `state_dict` (the
only `state_dict` ref in the tools tree is the READ in inspect_gradients). Any
gradient-health/eval claim on a "trained" model is meaningless. **Fix:** persist
weights (or retrain deterministically) before inspection.

**TOOL3/TOOL4 · HIGH/MED · FIXED (fix/tool34-algo-honesty, merged+verified: benchmark_algorithm + compare_implementations now return success=False — core has no safe sandbox to execute agent code, so they refuse rather than emit name-hash timing / hardcoded all_correct / broken complexity; 5 tests). Original finding below.**
**TOOL3 · HIGH · (was LIKELY; auditor VERIFIED; I spot-read evaluate_model, trust the rest)
— `benchmark_algorithm` and `compare_implementations` IGNORE the agent's code/inputs
and emit confident fake results.** `compare_implementations.py:39-100`: ignores
`implementations`/`test_inputs`; times a fixed matmul seeded by the impl NAME hash;
`peak_memory_mb = 12 + hash(name)%100/10`; `correctness` hardcoded `all_correct:True`.
`benchmark_algorithm.py:30-58`: ignores `code`, benchmarks a fixed dot-product; two
branches both return `"O(n)"` so `O(log n)`/`O(n log n)` are unreachable; `r2`
hardcoded 0.85. A buggy O(n²) impl is certified "correct" and a false complexity is
"confirmed." **Fix:** sandbox-execute the provided code or return `success=False`.

**TOOL4 · MED · LIKELY — six DL/opt tools are seeded-RNG/formula synthetics whose
output ignores the model architecture** (`activation_statistics`, `hyperparameter_sweep`
[discards `n_steps`], `compare_attention_variants` [hardcodes "standard" +0.02],
`lr_range_test` [fixed parabola → constant `suggested_lr`], `gradient_noise_scale`
[monotonic SNR → always picks largest batch], `regularization_effect`). Labeled
"synthetic" in spec TEXT but emit measurement-shaped numeric fields that feed narrative
findings. None are `significance_capable` (can't drive the confirm gate directly).
**Fix:** flag `"synthetic": true` in output so downstream refuses them as evidence, or
make them real.

**TOOL5 · MED · LIKELY — `TOOLS.md` ↔ registry drift.** 7 documented-but-missing
(`effect_size_analysis`, `knowledge_distillation`, `pruning_analysis`,
`quantization_analysis`, `hessian_analysis`, `loss_landscape`, `ablation_study`);
7 implemented-but-undocumented; TOOLS.md's "all 40 discoverable via GET /tools" is
false. Agent tool-planning may target non-existent tools. **Fix:** regenerate TOOLS.md
from `registry.get_all_specs()`.

**TOOL6 · LOW · LIKELY — `statistical_significance` bootstrap one-sided `less` tail is
inverted** (`ml_research/statistical_significance.py:166-178`; uses `>= obs` for
`alternative="less"`). t-test/Wilcoxon paths correct. **Fix:** use `<= obs` for `less`.
Also TOOL7 · LOW — `compare_gradient_methods` `steps_to_1pct` hardcoded to `n_steps`.

## LAYER 8 — Lifetime learning (audited 2026-07-07) [deep]

Auditor produced a write-site/read-site map; I verified the top mechanisms.
**Partially honest:** numerical-seed accumulation → seed prompt, and meta-ledger
baseline → candidate accept/reject, ARE genuinely wired. The dishonesty is the NUMERIC
policy-learning story.

**LL1 · HIGH · FIXED (fix/ll1-ll3-lifetime, merged+verified) — `theme_success_rates`
is structurally always 1.0.** `knowledge_graph.py:184-192` computes fraction-confirmed
over `self.claims`, but `graph.claims` is written only from `extract_confirmed_claims`
(so it holds ONLY confirmed rows; refuted/inconclusive become `failures`). Every theme
rate = 1.0 → in `policy_mutation.py:37-54` `rate>=0.4` always true (uniform boosts) and
the `rate<0.15` penalty/saturation branch is DEAD. "Learn what fails" doesn't exist.
A masking test inserts a `refuted` Claim that production never creates. **Fix:** compute
denominators from confirmed + matching `graph.failures`. (TODO: confirm add_claim only
ever gets confirmed — I verified the rate math + established_fact filter, not the caller.)

**LL2 · HIGH · FIXED (fix/ll2-policy-dispatch, merged+verified: `_information_gain_score` applies a bounded per-theme multiplier from the loaded policy — theme_weight clamped to [0.1,2.0]→±15%, +0.75 for blocked signatures; campaign_loop threads `search_policy` into `set_scoring_context`; convergence bench UNCHANGED depth 3.2; I finished the agent's partial work + added a defensive clamp + 4 tests) — WAS: learned policy knobs never steer live dispatch.** Live dispatch
(`next_dispatch_candidate`→`_information_gain_score`) never references the `SearchPolicy`;
only `saturated_themes` COUNT nudges one global scalar (`campaign_loop.py:1725`).
`theme_boost`/per-theme `theme_penalty`/`blocked_failure_signatures`/
`prefer_replication_t2_plus`/`closure_target` are telemetry + LLM-prose only; the numeric
application (`layer05/policy_dispatch.py`) is called ONLY from offline replay. So
"search-policy learning" is inert at the numeric decision layer for every domain.
**Fix:** route live dispatch through `policy_adjusted_score`, or fold `theme_weight`
into `_information_gain_score`. (LL4 · MED — `blocked_failure_signatures` etc. reported
as enforced but are advisory only.)

**LL3 · MED-HIGH · FIXED (fix/ll1-ll3-lifetime, merged+verified: from_dict now field-filters every record; load() re-raises loudly on genuine corruption instead of returning empty-that-overwrites) — WAS: a single drifted record silently WIPES the
entire lifetime store.** `knowledge_graph.py:212-221` builds records with unfiltered
`Claim(**v)`/`MechanismRecord(**v)`/…; any persisted key not in the current dataclass
raises `TypeError`; `load()` (:252-254) catches `TypeError` and returns an EMPTY graph;
the next `ingest_campaign` `save()`s empty over the JSON → permanent, unlogged loss of
all cross-campaign knowledge. Triggered by ANY schema drift (fields were added over
time) or a mixed-version deploy. The Postgres/meta-ledger loaders DO field-filter — this
path was missed. **Fix:** field-filter in `from_dict`; on load error, log loudly and
REFUSE to overwrite (fail closed).

**LL5/LL6 · MED/LOW-MED · FIXED (fix/ll5-ll6-lifetime-honesty, merged+verified: LL5 theory names/assumptions domain-neutral — contagion framing only for genuine network-diffusion themes; LL6 `established_fact_texts` gates on T2+ replication OR confidence≥0.6 and carries campaign_id/claim provenance not []; 14 tests). Original finding below.**
**LL5 · MED · (was LIKELY) — theory objects hardcoded to the contagion/network domain.**
`theory_objects.py:24-33`: `name=f"{theme}_contagion_theory"`, assumption "Competing
diffusion models apply", injected into EVERY future campaign's prior via
`enrich_prior_from_lifetime`. A materials/number-theory campaign gets contagion-framed
"established" priors → cross-domain contamination. **Fix:** derive from the domain
plugin or make domain-neutral. **LL6 · LOW-MED — `established_fact_texts` promotes T1,
confidence-0 claims with `paper_ids:[]`** (no confidence/replication gate, no provenance)
→ future campaigns compound on under-evidenced self-generated "facts."

## LAYER 12 — config / llm (audited 2026-07-07) [deep]

**CFG2/CFG3 · HIGH · FIXED (fix/cfg2-llm-failclosed, merged+verified: SUPPORTED_LLM_PROVIDERS + LLMConfigError + _validate_llm_config, both placeholder returns removed, config errors non-transient; 58 llm/hypotheses tests pass) — WAS: LLM client returns a hardcoded PLACEHOLDER
hypothesis on unsupported-provider or missing-key, silently treated as real output.**
`llm.py:112-130`: after handling ollama/gemini/openai, `if prov != "openai" or not
self.api_key: return json.dumps([{...placeholder hypothesis...}])` (mirror for gemini
at :166-178). So `LLM_PROVIDER=anthropic`/`claude`/typo, OR openai/gemini with an empty
key, yields a canned hypothesis; downstream `_parse_hypothesis_json` sees 1 valid
hypothesis → `llm_empty=False` → the campaign "researches" one generic placeholder
across every domain, run looks healthy, no error. **Domain-agnostic silent engine
failure.** CFG3 (same root) — NO provider whitelist/validation anywhere. **Fix:** raise
`LLMConfigError` on unknown provider / missing required key; never return fabricated data.

**CFG4 · LOW-MED · VERIFIED (auditor said HIGH — DOWNGRADED by me) — `_parse_action`
defaults to `stop` on a malformed decision response.** `think_act.py:603-606`. Auditor
claimed silent zero-work sub-agents, but I read the surrounding code: `think_act.py:576-598`
enforces a significance gate — an agent stopping without a significance tool (past
min_steps) gets a correction prompt, then `_fallback_significance_action`. So it's NOT a
silent empty trace in the main path; residual risk is only the pre-min_steps early-stop.
Real but LOW-MED. **Fix:** treat unparseable decisions as error/retry, not implicit stop.

**CFG5 · MED · FIXED (fix/cfg5-embed-default, merged+verified: `resolve_embed_model` rewrites the OpenAI cross-provider default to `gemini-embedding-2` for gemini/google, with a loud warning; 9 tests) — WAS: `embed_model` default assumes OpenAI, mismatches gemini.**
`config.py:32` `embed_model="text-embedding-3-small"` (OpenAI id) but `embeddings.py:64`
routes gemini/google to `_google_embed`. A gemini deployment that doesn't override
`EMBED_MODEL` sends an OpenAI id to the Google endpoint → 400/throw → callers catch
broadly and `return None` → retrieval silently drops to non-embedding ranking. Same
class as CFG1. **Fix:** derive embed-model default from `embed_provider` or validate the
pair. **CFG6 · LOW · VERIFIED — token usage never captured** (`llm.py:235-236`
`input_tokens/output_tokens` hardcoded `None` though all 3 providers return counts) → no
honest usage-based budget signal (compounds BUD1).

## CAMPAIGN READINESS (2026-07-07, overnight)

**LIT-WIRE · CRITICAL · FIXED (feat/wire-literature-service + docker-compose, merged+verified) —
the REAL dedicated literature service is now wired into campaigns.** Was: campaigns
used the OLD orchestrator-embedded `build_prior`; the new LitQA2-benchmarked service
(`services/literature/`, sources arxiv/oeis/semantic_scholar/zbmath/pubmed/…) was
never called (`literature_service_url` defined-but-unread). Fix: `literature_client.build_prior_via_service`
POSTs `{literature_service_url}/prior` (health-probe first), maps PriorResponse→Prior
(contradictions→contested_claims, tabulated_values preserved, citation_verification_rate→
evidence_coverage), and on ANY error logs + emits `literature.service_fallback` + falls
back to the OLD path with a recorded diagnostic (never silent). Gated on
`literature_service_url` (empty→OLD path, backward-compat). Added `literature` to
docker-compose (port 8020, in-memory backends, GOOGLE_API_KEY) + `LITERATURE_SERVICE_URL=
http://literature:8020` on orchestrator+worker. Verified: 7 client tests + full suite 739.
Math (Sidon) is a viable target — the service has arxiv/oeis/zbmath connectors and the
math domain declares a rich Sidon profile (Erdős-Turán, Croot-Lev-Pach, OEIS A005282).

**LIT-PERF · HIGH · FIXED (fix/lit-perf-fast-standard, merged + LIVE-verified by me) —
cold `/prior` now returns a real prior within the campaign budget.** Fix: `standard`
depth uses an ABSTRACT-ONLY path (no full-text PDF fetch; `raw_to_abstract_document`
→ existing extractor over the search-hit abstract), capped to top-4 priority sources ×
5 docs; the depth budget is now a soft `deadline_sec` threaded into the pipeline
(`_gather_with_deadline`) so a slow build returns PARTIAL real results, not empty.
**LIVE result (I ran it, real Gemini keys):** cold Sidon `/prior` standard depth →
HTTP 200 in ~65s, `papers_indexed=2`, `established_facts=2` (a genuine cap-set bound
from arXiv "Bounds on sizes of general caps in AG(n,q)"), `tabulated_values=3`,
sources arxiv/oeis/semantic_scholar/zbmath. The ~65s is the standard deadline returning
partial content — WELL within the campaign's 600s `literature_service_timeout_sec`, so a
campaign now gets a genuine literature prior from the REAL service (no more empty/block →
old-path fallback). 162 literature tests. Original finding below.
**LIT-PERF (orig) · was HIGH · the cold `/prior` was too slow to feed a campaign in-budget.** I ran the service locally
(`uvicorn`, `.env` keys): `/health` = ok, math sources healthy (arxiv/semantic_scholar/
zbmath/mathoverflow up; oeis down). But a cold Sidon `/prior` at `depth="standard"`
(60s budget) **timed out → returned EMPTY** (`papers_indexed=0`, `sources_consulted=[]`);
`depth="deep"` (300s) **blocked past even the client's 330s timeout**. Root cause
(`retriever/query.py`): for every doc from every source, `_fetch_and_process` downloads
the FULL-TEXT PDF (`fetch_full_text`) + LLM-extracts claims (`process_document`), and
`deep`/`exhaustive` add a semantic-scholar citation crawl. Thorough (that's the LitQA2
0.76) but far too slow cold, and `asyncio.wait_for` doesn't cleanly cancel the in-flight
PDF/LLM work (blocks past the deadline). **Impact:** with LIT-WIRE, a campaign would hit
the timeout and fall back to the OLD embedded prior path — so the REAL literature layer
is NOT actually exercised. **This is the reason a campaign is NOT launched yet.**
**Fix (assigned):** a fast ABSTRACT-ONLY path for `standard` depth (skip PDF fetch;
extract from abstracts across a few top sources) that returns a real prior in <60s, and
have the timeout return PARTIAL results (what was gathered) instead of empty.

**Readiness checklist:** ✅ 16 verified fixes across 3 waves + LIT-WIRE + TOOL6 ✅ full
suite 750 ✅ literature wired + in compose + LIVE-verified (real Sidon prior in ~65s,
within the 600s campaign budget) ✅ all 5 benchmarks re-confirmed (verdict 0.0, binding
1.0/1.0, generation 0.0/1.0, convergence 3.2) ✅ LIT-PERF fixed ⏳ clean stack REBUILD
(running containers are stale/pre-fix) + full bring-up + /health ⏳ short observed Sidon
campaign. Remaining LOW backlog (CFG4/6, TOOL7 ML-only, DOM5 masked, flaky graph-preflight
30s budget) — not campaign-blockers.

## LAYER BASELINES — quantified metrics (the engineering-loop scoreboard)

Each layer is being turned into a measured engineering problem (like LitQA2 for
literature). Harnesses live in `bench/` (drive the REAL code; metric-moves
sanity-checked). Baselines established so far (verified by me — read + ran):

| Layer | Harness | Metric | Baseline | Read |
|-------|---------|--------|----------|------|
| Literature | LitQA2 eval | accuracy | **0.76–0.78** (n=100) | done |
| Convergence | `scripts/bench_campaign_convergence.py` | confirmed-lineage depth / narrow-reject | **3.2 / 0.001** | done |
| Verdict | `bench/bench_verdict.py` | **false-confirm rate** / recall | **0.0 / 1.0** (clean) | ✓ ran |
| Evidence-binding | `bench/bench_binding.py` | precision / recall | **0.5 → 1.0 / 1.0** (BND1 FIXED) | ✓ ran |
| Generation | `bench/bench_generation.py` | dup-pass / off-topic-reject | **0.0 / 1.0** (filters clean) | ✓ ran |

**BND1 · MED · FIXED (fix/bnd1-binding-precision, merged+verified: I re-ran the bench myself — precision 0.5→1.0, recall held 1.0, false-accepts 5→0; subject-discriminating overlap via `_RELATIONSHIP_TERMS` stoplist + guarded `_subject_mismatch` veto; 22 tests) — WAS: binding accepts cross-domain supporters →
precision 0.5.** The binding benchmark shows recall 1.0 in ALL five domains (A4
domain-generality confirmed) but precision 0.5: irrelevant + fabricated citations
are cleanly rejected (0/5 each), yet all 5 "genuine-for-a-different-subject"
cross-domain nodes bind, because they share ≥2 relationship/methodology salient
terms (`elasticity`+`reduce`, `scales`+`spacing`, …) — `_MIN_SHARED_SALIENT_TERMS=2`
counts generic relationship words. **Improvement target:** raise binding_precision
from 0.5 toward 1.0 while KEEPING recall 1.0 — require the shared salient terms to
be SUBJECT-specific (nouns/entities), not generic relationship verbs; or add a
subject-mismatch reject. This is the first measurable A4-followup.

**Verdict baseline caveat:** `false_confirm_rate=0.0` measures the gate given
HONEST provenance. TOOL1 (a tool that stamps fabricated variance as `computed`)
is upstream of this benchmark — the gate is honest, the tool lies. Fixing TOOL1 is
what actually protects the ML domain; a future verdict-bench case could add a
"tool-fabricated computed-provenance" probe once TOOL1 is fixed.

## LAYER 7 — Domain layer (audited 2026-07-07) [deep]

Auditor's worktree was STALE (pre-D1 base), so I re-verified each finding against
the primary `campaign-convergence` tree. Its "D1 not present" note is a FALSE ALARM
— `base.py:212-270` has `has_verification_capability()` + fail-closed `preflight()`;
D1 is intact. Do not re-chase. The rest verified real:

**DOM1 (was L7-1) · HIGH · FIXED (fix/dom1-profile-preflight, merged+verified: _enforce_domain_preflight now fails closed when a profile resolves but no plugin owns it, finalize STOP_REASON_DOMAIN_PREFLIGHT_FAILED; generic path preserved; 28 tests pass) — WAS: a domain PROFILE
without a PLUGIN launches with no verification capability and no preflight, yet the
artifact gate applies its standards.** `econometrics` has a `domain_profiles/econometrics.py`
profile but NO plugin dir (verified: `ls domain_modules/` has no econ). So
`resolve_domain_plugin` → None → `_enforce_domain_preflight` (campaign_loop.py:1522-1523)
`if plugin is None: return True` → launches with no power gate; worker falls to the
generic sandbox path (no econ adapter); but the artifact gate still applies econometrics
profile standards. **D1 does NOT cover this** — D1 fails-closed a plugin with no
capability, but a profile-with-no-plugin never reaches the base preflight. Any new
domain author who adds a profile (natural — profiles drive the gate) creates the
appearance of support with zero executable verification. Central-thesis-in-miniature.
**Fix:** `_enforce_domain_preflight` must fail-closed when a domain PROFILE resolves
but no plugin owns it (or require every profile to have a verifying plugin).

**DOM2 (was L7-2) · HIGH · FIXED-labeling (fix/dom2-synthetic-provenance, merged+verified: `uses_synthetic_data()` flag flows adapter→`data_provenance` on evidence→paper labels "synthetic dataset (illustrative)" in table/narrative/methods; `_effective_verdict` untouched; 28 tests incl honesty regression). RESIDUAL OPEN: the graph_invariants `modularity=0.25·clustering+…` TAUTOLOGY (a real generator flaw, labeling doesn't fix it) — track as DOM2b. WAS: three demo domains "confirm"
findings on SYNTHETIC seed-42 data presented as real datasets; one is a tautology.**
`genomics/adapter.py` (`_synthetic_gtex_frame`, used unconditionally, meta
`synthetic:True`) presents as "GTEx v8 subset"; `graph_invariants/adapter.py`
(`_synthetic_frame`) as "SNAP subset"; enzyme_kinetics similarly (per auditor). A LOFO
"confirmation" detects the seeded structure, not science. Worse, `graph_invariants:125`
`modularity = 0.25*clustering + 0.1*(avg_deg/n)` is a DETERMINISTIC function of
clustering → a "modularity↔clustering" finding is a tautology of the generator. The
`synthetic:True` flag is discoverable in meta but NOT surfaced at verdict/paper time,
so these flow through the same pipeline as real results. (Refines the earlier
"enzyme/graph are not stubs" note: real verification LOGIC, but synthetic DATA dressed
as real.) **Fix:** surface synthetic-data provenance into the verdict/paper and
downgrade/flag, or load real datasets; never present a seed-42 confirmation as a
real-dataset finding.

**DOM3 (was L7-3) · MED · FIXED (fix/dom3-routing-confidence, merged+verified: `match_score` count-of-markers; `resolve_domain_plugin` picks MAX not first; ambiguity warning on near-tie; explicit-tag fast path preserved; 25 tests) — WAS: routing
collisions resolve by REGISTRATION ORDER, no confidence.** `registry.py` `resolve_domain_plugin`
returns the FIRST plugin whose `matches()` is True. A question with both combinatorics
and graph terms fires `graph_invariants` AND `math_combinatorics` → silently routes to
graph_invariants (registered first) → verified against synthetic SNAP data (DOM2). As
domains proliferate, an older plugin shadows a correctly-authored newer one. **Fix:**
`matches()` returns a score; pick max; emit a routing-ambiguity event on a near tie.

**DOM4/DOM2b · MED · FIXED (fix/dom4-dom2b-graph-honesty-v2, merged+verified: `_plugin_verification_path` calls `hypothesis_on_topic` (fail-OPEN guard) → off-topic short-circuits to inconclusive; `from_hypothesis` raises `GraphInvariantNotIdentified` instead of defaulting; DOM2b modularity is now real Newman-Q on a Fiedler bipartition [not a fn of clustering — verified within-family |corr|≤0.29] + fixed a bonus `algebraic_connectivity==spectral_gap` identity; 15 tests). Original finding below.**
**DOM4 (was L7-4) · MED · (was VERIFIED-by-auditor) — graph_invariants `from_hypothesis`
defaults to `spectral_gap→clustering` for ANY text, and the worker never re-checks
on-topic.** `_plugin_verification_path` calls `run_verification` directly without
`hypothesis_on_topic`, so an off-topic/misrouted hypothesis is verified against a fixed
default correlation on synthetic data → can yield a confirmed verdict decoupled from the
claim. Combines with DOM3. **Fix:** `from_hypothesis` returns a sentinel/raises when no
invariant is confidently identified; `_plugin_verification_path` calls `hypothesis_on_topic`
first and short-circuits to inconclusive if off-topic. **DOM5 (L7-5) · LOW · SUSPECTED —
`_run_ap_free_sweep` marks `verified_true=1` for ≥3 points regardless of claim**, but
`apply_claim_validation` masks it in the routed path (couldn't construct a live exploit).

---

## External report triage — `fixes.md` (weaker-model audit, 2026-07-07)

A separate lower-capability agent mapped the codebase and produced `fixes.md`
(3 sub-audits: core / services / tests-config). Every claim was verified against
the actual code. **Result: ~90% false or already-covered; 2 genuinely-new items,
both LOW / arguably by-design.** Recorded here so these are never re-investigated.

**FALSE (invented behavior — the model did not read the code):**
- *"Dedup is an LLM call" (1a)* — FALSE. `campaign_synthesis` dedup is
  `difflib.SequenceMatcher` structural + scope/param signatures
  (`campaign_synthesis.py:79-97`). No `_deduplicate_candidates`/`_is_near_duplicate`
  LLM fn exists; it was invented.
- *"Evidence-binding is an LLM call" (1b)* — FALSE. `evidence_binding.py` is pure
  regex/structural (`re.compile`, `_structured_overlap`); no LLM anywhere.
- *"Sandbox is subprocess, not Docker" (services 2c)* — FALSE. It is Docker
  (`services/worker/sandbox.py:29` `docker.from_env`, `containers.run`, image +
  bounded wait + kill). The model misread the `"subprocess"` token in a *denylist*
  of blocked identifiers.
- *"Literature prior is 100% LLM hallucination, no real DB" (3b / services 1a,2b)* —
  FALSE, and this was its headline claim. `build_prior` queries arXiv/PMC over
  httpx, runs hybrid retrieval, gates corpus quality, and either synthesizes from
  **real** papers or returns `insufficient_prior(skipped_llm=True)`
  (`literature.py:884-901`). Never silently falls back to a hallucinated prior.
- *"enzyme/graph plugins are stubs; preflight hardcoded True" (1e)* — FALSE. Both
  have real `scope_template`, overridden `run_verification`, `domain_profile`,
  `adapter.py`/`verifier.py`, and a `preflight()` that runs a real LOFO/invariant
  timing check and returns `PreflightResult(False,…)` on failure.
- *"Worker domain routing is a hardcoded table; new domain needs a code edit"
  (services 3a)* — FALSE. `_worker_verification_paths` (`sub_agent_loop.py:1464`)
  is a fast-path for the two LOFO domains only; all others fall through to
  `_plugin_verification_path` (generic) with no edit. (That generic path is what
  D2/Agent-G is hardening.)
- *"Dispatch failures silently skipped, no retry" (services 4a)* — FALSE. Baseline
  `.delay()` except logs (`campaign_loop.py:856`); frontier refill has explicit
  retry (`:1954`, `:1983`).
- *"Synthesis parse failure silently succeeds" (4a)* — FALSE. Detected + flagged:
  `metrics["parse_error"]=True`, returns empty (`campaign_synthesis.py:391`).
- *"Breakthrough = LLM enthusiasm" (services 2a)* — FALSE. `is_breakthrough`
  (`campaign.py:74`) is metric-vs-baseline gated by replications + confirmed-count.

**ALREADY COVERED (duplicates of existing registry issues):**
- Binding relevance is a judgment call (1b-impact) → **A4** (structured-overlap
  integrity guard, FIXED).
- Generic/stub domain verification produces no real null → **D1** (fail-closed
  preflight, FIXED) + **D2** (real permutation null, in flight).
- `.env.example` OpenAI default (config 1d) → **CFG1** (OPEN LOW).
- Lifetime JSON last-writer-wins (5a) → **L8** (known, documented).
- ML metric trust upstream of breakthrough (2a) → **O1** / **W1b**.
- Verdict stages short-circuit without a skip-log (1c) → partial overlap **V2**;
  the deterministic-bypass hole it gestures at is already closed.

**GENUINELY NEW (both LOW; recorded, not assigned — do not burn an opus agent):**
- **HM1 · LOW · VERIFIED** — No code halts a campaign on a collapsed health metric;
  health metrics are observability-only (no `should_stop`/abort reads them). This
  is *honest* (architecture states metrics are observational) — a candidate
  circuit-breaker *feature*, not a dishonesty bug. Related to the A2 symmetric
  rejection-rate warning (which warns but does not gate). Status: WONTFIX-ish /
  backlog.
- **BUD1 · LOW · VERIFIED** — `Campaign.compute_budget_seconds` measures
  **wall-clock** (`campaign.py:270-277` prefers `started_at` elapsed), so the
  `compute_` name overstates it. For a continuously-running single campaign
  wall ≈ compute, so impact is minimal; a rename/comment would improve honesty.
  Status: backlog LOW.

*Triage takeaway: the report validated the "verify regardless of who reports it"
rule in the opposite direction — a weaker model's confident claims were mostly
hallucinated code that does not exist. No claim rose to subagent-worthy.*

---

*Audit loop status: LAYERS 1–2 deep; 3 partial; 4–12 pending. `fixes.md` triaged
(2/23 real, both LOW). Continue reading files, add issues with IDs, keep this the
source of truth.*
