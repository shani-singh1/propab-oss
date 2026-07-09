# Convergence analysis — campaign ad5fd802 (Sidon-set exploration)

**Question this answers:** *Will this architecture work — can Propab work a research
problem correctly over a long horizon and converge to a novel finding?* Grounded in the
full DB record (hypotheses, orchestrator reasoning, worker think-act, tool calls +
params + outputs, generated code + results), not counts.

**Setup:** general-agent mode ON, gemini-3.5-flash, 60-tool catalog, reasoning ON.
Question: max size f(N) of a Sidon set in {1..N} for small N; conjecture growth vs √N;
separate certified maxima from conjectured asymptotics. Stopped ~10 min in.

## Verdict: the hard parts WORK; the last mile is BROKEN (and it's one precise gap)

| Capability | Works? | Evidence |
|---|---|---|
| Orchestrator **convergence** (steers back to the real question) | ✅ | reasoning repeatedly caught off-target attempts and pivoted: "…failed on scope integrity; we pivot back to the core question of exact max Sidon sizes for N≤40"; "…refuted and had low relevance; we pivot back". By round 2-3 it had converged on hyps 5/6/7 = "compute f(N) exactly, test the √N bound". |
| Worker **experimentation** (design + write + run correct code) | ✅ | a worker wrote a real MILP (`scipy.optimize.milp`) and computed **verified f(N) for N≤71** with witnesses (`f(4)=3, [1,2,4]`, …) — the correct exact values. |
| **Honesty** | ✅ (bulletproof) | `sequence_oracle` returned `verdict: unknown` because the conjecture "did NOT reproduce the 15 held-out terms"; off-target results → inconclusive/refuted; **0 false confirms**. |
| **Recognizing a correct deterministic computation as a FINDING** | ❌ | **0 confirmed** despite a complete, verified computation of f(N). The right answer was computed, then discarded as "inconclusive — metric direction ambiguous". |

**So: the brain steers, the workers compute the right answer, and nothing false gets
through — but the system cannot CLOSE THE LOOP and declare a finding for
deterministic/computational math.** It solved the problem and threw the solution away.

## Root cause: the evidence→verdict + gate layer is ML-shaped

The whole verdict path assumes an **ML measurement**: a single scalar metric to optimize
+ a significance test. Deterministic math produces a different evidence shape (an exact
computed object / a sequence / a bound / a conjecture-with-held-out-evidence). Concretely:

1. **The ML significance gate is forced on a deterministic result.** After computing f(N)
   exactly, the worker was made to satisfy the significance gate → it flailed calling
   `literature_baseline_compare` (7 ERRORS: "missing our_results", "baseline_value
   required") and `bootstrap_confidence` on deterministic values like `[7,16,16,16,20.6]`.
   A bootstrap CI on exact f(N) values is meaningless. My S0 gate fix exempted
   `verification_capable` **tools** — but a worker who computes an exact result in **code**
   never routes through such a tool, so the exemption doesn't apply.
2. **The "metric" framing can't hold the finding.** `sidon_max_size` as one scalar with a
   direction is nonsensical — the finding is the SEQUENCE f(N) + the ASYMPTOTIC ratio
   f(N)/√N. The orchestrator kept concluding "metric's direction was ambiguous →
   inconclusive". A verified exact computation has no place to land as "confirmed".
3. **No verdict path for computational evidence.** A code result `{"verified": true,
   "f_values": {…N≤71…}}` is a certified computation, but `classify_evidence_type` /
   `run_verdict_pipeline` only know deterministic-PROOF (verification_method ∈ proof set),
   lofo, or statistical — a verified code computation falls to "unknown" → not judgeable.

This is the SAME class as the certified-witness gap fixed in S0 (commit b777c98), but one
level more general: S0 covered a trusted TOOL's certified output; this needs the same for
a **worker-computed, sandbox-verified deterministic result**.

## Secondary issues (real, lower-rank)

- **Tool selection.** The agent wrote its own MILP/backtracking instead of calling
  `constraint_solve` (the audited tool), and mis-picked `extremal_set_search` (B₃) and
  `enrichment_analysis` (biology) for a Sidon question. It got the right answer anyway (by
  writing code), so this is lower-rank than the verdict gap — but a reproducible audited
  tool is preferable to ad-hoc agent code, and sharper tool descriptions + a selection
  skill would help.
- **No asymptotic-fit capability.** The real finding is f(N) ~ c·√N (ratios
  [1.06,1.15,1.24,…]). `sequence_oracle` tests exact recurrences/closed-forms (correctly
  returned `unknown` — f(N) has no simple recurrence). There is no tool for "fit/verify an
  asymptotic trend", which is the actual shape of this finding.
- **Hypothesis redundancy.** Hyps 5/6/7 are near-duplicates of "compute f(N) exactly + test
  the bound". The orchestrator converges but re-spawns parallel copies instead of deepening
  one line — slower convergence, wasted budget.

## Answer to the question + next steps (ranked)

**Will it converge to a novel finding over a long horizon?** The *mechanism* will — the
brain demonstrably corrects course and the workers compute correct, verified results, with
honesty intact. But **today it cannot DECLARE the finding for computational/deterministic
math**, so a long run would keep computing correct answers and marking them inconclusive.
Fixing the last mile is what turns this from "wanders honestly" into "converges and reports".

1. **[THE fix] Recognize deterministic-computation evidence as a first-class finding.**
   Generalize the S0 certified-witness verdict path: a worker's sandbox-**verified** exact
   result (`{"verified": true, …}` — an exact computation, an exhaustive check, a
   constructed+certified object) maps to a CONFIRMED verdict for the computational claim,
   and is EXEMPT from the ML significance gate (extend the evidence-shape-aware gate from
   `verification_capable` tools to `verification_method`-tagged code results). Mirror
   `enrich_certified_witness_evidence` for code evidence.
2. **Support structured findings, not a single scalar metric.** A math campaign's
   "finding" is a computed object / sequence / bound / asymptotic + its certification +
   any conjecture-with-held-out-evidence — not "maximize sidon_max_size". The
   breakthrough/verdict framing needs a finding abstraction beyond one ML metric.
3. **Add an asymptotic-fit tool** (fit + held-out-test f(N) ~ c·n^a·log^b n), the actual
   shape of growth findings; and sharpen tool descriptions so the agent reaches for
   `constraint_solve` over hand-written MILP and never `extremal_set_search`/bio tools on a
   Sidon question (or a lightweight tool-selection skill / relevance-scoped catalog).
4. **Deepen, don't re-spawn.** Bias the orchestrator to iterate one converged line rather
   than emit near-duplicate hypotheses.

**Bottom line:** the architecture is closer than it looks — it already steered to the right
question, computed the right answer, and stayed honest. The blocker isn't intelligence or
honesty; it's that the verdict/gate layer speaks "ML metric" and can't hear "I computed
the exact answer." Close that (a direct generalization of a fix already shipped for tools),
and the same run would have reported: *"computed f(N) exactly for N≤71 (certified); f(N)/√N
trends to ~[c] — conjecture, held-out-tested, not proven"* — a real, honest, closed-loop
finding.
