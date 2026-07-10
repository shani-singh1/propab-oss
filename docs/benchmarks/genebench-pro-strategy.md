# Cracking GeneBench-Pro — strategy (no hardcoding)

**Date:** 2026-07-10
**Status:** design. Grounds out the pivot doc ([active-external-benchmarks](active-external-benchmarks.md))
and the GPT-5.6 audit findings G1/A3/A4 ([frontier-science-audit-2026-07-10](../audits/frontier-science-audit-2026-07-10.md)).

## 1. The problem, precisely

Measured, same base model (gemini-3.5-flash), same 10 public problems, same deterministic grader:

| Configuration | Score |
|---|---|
| Bare code agent (base + sandbox) | 1/10 (10%) |
| + hand-written rigor prose | 1/10 (10%) |
| + Propab's real tool registry + skills | **0/10 (0%)**, **0 tool calls** |

Forensics (per-field agent vs. truth) show the agent **reaches for the right method** — it loads the
empty-droplet file and computes an ambient profile for the eQTL task, fits a Wright-Fisher likelihood
by MLE for the selection task, attempts founder reconstruction for the QTL task — but **executes it
imprecisely** (wrong ambient offset, wrong link, over-corrected winner's curse, a founder call off by
one). It is an **execution-precision** problem, not a method-knowledge problem.

Why the scaffold gave 0 lift (audit A3/A4/G1): the 60-tool registry is a **flat, data-disconnected
catalog** — tools run *outside* the staged-data sandbox on caller-supplied scalars (e.g.
`enrichment_analysis` wants query genes + a background set, not a file handle), so the agent
**cannot** feed them `/work/data_files`; rational result = 0 calls. The skills are research-*framing*
prose that dilute rather than compute.

**Contrast with the win we want to repeat.** litQA2 went 0→0.78 because the scaffold (live retrieval)
*directly supplied the missing capability* (answer-bearing evidence). To lift GeneBench the scaffold
must likewise supply the missing capability — here that is **precise, correct execution of standard
genomics estimators** — and it must do so **generally**, never per question.

## 2. The key insight (why this is tractable without hardcoding)

The 129 problems span 10 domains, but they are drawn from a **finite set of ~20–30 standard
genomics estimators**. The same estimator recurs across every problem of a family:

- every single-cell eQTL task needs *ambient correction → cell-state gating → count GLM with a
  library-size offset → per-allele log rate ratio* — the **same** procedure, different data;
- every multi-parent QTL task needs *HMM founder-haplotype reconstruction → LMM association →
  peak position*;
- every allele-frequency-trajectory task needs a *Wright-Fisher selection MLE with the drift
  schedule*;
- every cis-MR task needs *winner's-curse correction + LD-aware instrument combination*.

A real computational biologist does **not** re-derive these from scratch each time; they call
PLINK / limix / scikit-allel / statsmodels. GPT-5.6 Sol likely scores 31.5% partly because its
*from-scratch* execution is stronger. We give a modest model the **validated estimators** so it does
not have to implement them perfectly by hand — the creative/judgment act (which method, which
columns, which design, how to interpret) stays with the agent.

## 3. The solution: a general, validated, data-native genomics primitive library

Three components. None reference any specific problem.

### 3a. Primitive library (`propab_genomics`, importable inside the sandbox)
General, unit-tested estimators — textbook methods, one per recurring analysis, operating on the
**staged data files / dataframes**, returning typed results with provenance. Organized by the
benchmark's own domain families (each covers *any* problem of that type, e.g.):

| Family | Example primitives (general, not per-question) |
|---|---|
| Single-cell / eQTL / regulatory | `estimate_ambient_fraction(counts, empty_droplets)`, `gate_cell_state(cells, markers)`, `fit_count_glm(y, design, offset, family="nb")` → per-allele log-rate coef |
| Statistical genetics | `lmm_association(pheno, geno, kinship)`, `winners_curse_mle(beta_hat, se)`, `cis_mvmr(exposures, outcome, ld)`, `ld_prune(geno, r2)` |
| Quantitative genetics | `hmm_founder_reconstruct(geno, founders, markers)`, `heritability_reml(pheno, grm)` |
| Population genetics | `wright_fisher_selection_mle(freq_traj, Ne_schedule, seq_error)`, `admixture_pulse_infer(...)`, `sex_bias_contrast(autosome, X)`, `fst(...)` |
| Clinical / PGx | `cox_time_to_event(...)`, `carrier_residual_risk(...)`, `cnv_calibrate(...)` |
| Cancer / structural / 3D | `hic_loop_strength(contacts, mask)`, `sv_expression_effect(...)`, `somatic_vaf_model(...)` |

Each primitive is a thin, **correct** wrapper over statsmodels/scipy/scikit-allel/lifelines/etc.,
with a fixed input/output contract and QC hooks. The agent still writes glue code to load, clean,
and shape the staged data and to read the result.

### 3b. Data-native interface
The sandbox imports the library; primitives take **file paths / dataframes** (the staged
`/work/data_files`), not scalars copied through an LLM prompt. This removes the A3 interface mismatch
that caused 0 tool calls. Tool outputs are typed handles (tables, fitted models) with a seed/version
manifest → replayable provenance (audit G1).

### 3c. Task-family router + executable playbook (methodology, not answers)
A lightweight classifier maps the **task text → analysis family**, and emits a *playbook checklist*
for that family: `estimand → required inputs/QC → primitive calls → acceptance/sanity check →
answer schema`. This is the audit's A4 fix ("router-bound executable playbooks", not prose) and it is
keyed on **analysis type**, never on the specific question. A `sanity_check(estimate, ...)` primitive
lets the agent catch an implausible sign/magnitude and revise (the missing feedback loop, audit A2).

## 4. The anti-hardcoding line (the crucial part)

The difference between a **tool** and **hardcoding**, made operational:

- **Hardcoding (forbidden):** anything keyed to a specific problem id, dataset, or answer — a solver
  for `statgen_scrna_ambient_state_eqtl`, a constant, a branch on the problem's data schema, tolerance
  tuning to pass a public case, or building a primitive by reverse-engineering a public answer.
- **A tool (allowed):** a *general* estimator that works on **any** dataset of that analysis type,
  chosen and wired by the agent. `fit_count_glm` is a tool the way PLINK is a tool.

Guardrails that keep it honest and general:

1. **Build from method literature, not from the 10 public answers.** Primitives implement the
   standard estimator; we never look at a public problem's `ground_truth` while writing one.
2. **Validate each primitive on SYNTHETIC data with known truth** (GeneBench's own SCM philosophy):
   generate data from a known model, assert the primitive recovers the parameter within tolerance.
   A primitive that only works on one public problem is a bug, not a tool.
3. **Held-out measurement.** The public 10 are a *smoke test only*. Real evaluation is the 50-question
   Artificial Analysis subset (disjoint from the public 10) and, ultimately, the hidden 119. Never
   tune anything to the public cases. If a change helps the 10 but not held-out, it's overfitting —
   revert it.
4. **Generality test per primitive:** it must plausibly serve problems in its family we have *not*
   seen. If we cannot state that, it's too specific.

## 5. Staging + how we prove it (honestly)

- **Stage 1 — proof (target: lift the public 10 from 10% toward ~40%+).** Build the 4 primitives the
  public families need — count-GLM eQTL w/ ambient correction, WF selection MLE, QTL HMM+LMM,
  cis-MR winner's-curse — each validated on synthetic data, data-native, importable in the sandbox +
  a minimal family router. If the same library (unchanged) also lifts held-out problems, the thesis
  is proven and worth scaling. If it only moves the 10, we overfit — stop.
- **Stage 2 — breadth.** Expand to the full ~25 estimator families across all 10 domains; measure on
  the 50-question subset.
- **Stage 3 — honesty + production fidelity.** Route through validated tools with provenance and the
  self-check/verifier loop; report paired traces (chosen family, primitive calls, estimate, score) so
  a lift is attributable and reproducible (audit G3/#8).

## 6. Why this is the right bet

It supplies the actual missing capability (precise execution) the way litQA2's retrieval did, it is
**general by construction** (estimators, not answers), it is **verifiable** (synthetic-truth
validation + deterministic grading), and it directly implements the audit's highest-priority
capability fix (G1 data-native substrate, A3/A4 workflow primitives + playbooks). It also compounds:
the same primitive library is exactly what the open-problem **search** track and any future genomics
work would reuse — it is capability, not a benchmark trick.

**Honest caveat.** Even done well, some problems are base-model-bound (multi-step research taste the
library can't supply); ~40–50% on the public set is a realistic ceiling for a modest base model with
strong tools, not 90%. That is still a legible headline (scaffolded small model beating much larger
un-scaffolded ones on an active benchmark) and, unlike the 0-lift scaffold, it is real capability.
