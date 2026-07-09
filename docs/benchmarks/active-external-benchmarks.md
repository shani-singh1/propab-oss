# Active External Benchmarks — Targeting Plan for Propab

**Author:** benchmark-targeting research (Claude, in-session)
**Date:** 2026-07-10
**Status:** Planning doc — the basis for building + measuring Propab against live, headline benchmarks.
**Companion:** [general-agent-redesign](../architecture/general-agent-redesign.md), [domain-capabilities](../architecture/domain-capabilities.md), [discovery/targets](../discovery/targets.md).

---

## 0. Why this pivot (strategy)

Instead of "shooting an arrow in the dark" for a novel internal discovery (real, but the
odds are gated by raw search strength we don't yet have — see [discovery/targets](../discovery/targets.md)),
we target **active, externally-visible benchmarks that frontier labs are being measured on
right now.** In July 2026 OpenAI shipped the **GPT‑5.6 (Sol / Terra / Luna)** family and its
life-sciences model **GPT‑Rosalind**, and reported results on a cluster of new science/research
benchmarks. These benchmarks are *hard and non-saturated* — the best models score 27–36% — which
means there is real headroom, and a strong number is **instantly legible** to the field.

**The precedent:** on **litQA2** (an external literature-QA benchmark) Propab's scaffold took a
weak base model from ~0 to **0.78 accuracy** (see `services/literature/app/evaluator/litqa2_live.py`,
`services/literature/scripts/run_litqa2_real.py`). The thesis: Propab's *scaffold* — a general
agent + trusted tools + verification/honesty gate + iteration — can lift a modest base model
(we run **gemini‑3.5‑flash**) well above its raw score, the same way it did on litQA2. On
GeneBench‑Pro, raw **Gemini 3.5 Flash scores just 8.1%**; the whole bet is that the Propab
scaffold on top of that base is worth many multiples of the base number.

**The sharpest fit:** two of these four benchmarks are **deterministically graded** (objective
ground truth, no rubric-judge noise) — which is *exactly* what Propab's honesty architecture is
built for (certify-a-witness, verified computation, no false confirms). That is our natural edge.

---

## 1. At-a-glance

| Benchmark | Domain | # tasks | Grading | Agentic / code? | Public data? | Best reported | Base (Gem 3.5F) | Propab fit | Priority |
|---|---|---|---|---|---|---|---|---|---|
| **GeneBench‑Pro** | Genomics / quant-bio / translational | 129 | **Deterministic** (synthetic from known SCM) | Yes (Python + PLINK 2.0 + sci libs) | Partial: 10 open on HF + 50 via Artificial Analysis | GPT‑5.6 Sol Pro **31.5%** | **8.1%** | **Very high** | **P0** |
| **LifeSciBench** | Life-science research (7 domains) | 750 | Rubric (19,020 criteria, ~25/task; pass ≥70%) | Free-response + artifacts | Unclear (reproducibility questioned) | GPT‑Rosalind **36.1%** | — | High (litQA2-like) | **P1** |
| **Automated LLM Speedrun** ("NanoGPT") | ML training / research engineering | 19 records | **Deterministic** (wall-clock to val-loss 3.28) | Yes (writes training code) | **Open-source** (academic) | SoTA LLMs "struggle" | — | High but off-domain + GPU cost | **P2** |
| **MedChemBench** | Medicinal chemistry / drug design | n/d | n/d (rubric-like) | Multimodal (chem structures) | No (proprietary) | GPT‑Rosalind **27.5%** | — | Low (multimodal, closed) | **P3 / defer** |

> Base-model column = the raw score of the model Propab runs on (gemini‑3.5‑flash) where reported.
> The gap between that and the scaffolded score is the opportunity.

---

## 2. GeneBench‑Pro  — **the top target (P0)**

**Source:** OpenAI, *Introducing GeneBench‑Pro* (openai.com/index/introducing-genebench-pro/);
tech report PDF (cdn.openai.com/pdf/…/genebench-pro.pdf); bioRxiv *"GeneBench‑Pro: Evaluating
Multistage Statistical Reasoning in Genomics, Quantitative Biology, and Translational Biomedicine."*
Builds on the earlier **GeneBench** (bioRxiv, 2026‑04‑22, "AI Agents for Multi‑Stage Inference").

### What it tests
Realistic **multi-stage** computational-biology analyses where the agent must produce a conclusion
that a downstream scientific/translational decision depends on. **129 problems**, **10 primary
domains / 21 subdomains**:

| Domain | ~count | Domain | ~count |
|---|---|---|---|
| Clinical / PGx / diagnostics | 26 | Cancer somatic genomics | 10 |
| Population genetics | 21 | Functional genomics | 9 |
| Statistical genetics | 17 | Proteomics | 7 |
| Quantitative genetics | 17 | Microbial genomics | 3 |
| Regulatory omics | 17 | Forensic genetics | 2 |

Example problems: CRISPR target validation; linked-locus mapping; carrier screening; parent-specific
ancestry; structural-variant-guided tumor therapy benefit–risk; pharmacogenomic time-to-event
(marginal structural Cox models with treatment–confounder feedback). Reviewers estimate a typical
problem is **20–40 human-expert hours** (~$4–8k).

### Task structure (what the agent gets / returns)
- **Input:** brief experimental context; a **target estimand** tied to a decision; an **isolated
  workspace** with Python, scientific libraries, and **PLINK 2.0**; **data files that simulate
  realistic, messy inputs**; instructions that explicitly discourage shortcuts.
- **Output:** JSON `{ "answer": <numerical result>, "reasoning": <analytical explanation> }`.
- **Agentic:** yes — the agent runs code in the workspace. "Problems do not require exotic
  domain-only tooling beyond that baseline."

### Grading — **deterministic, and this is why it fits us**
Every dataset is **simulated from a known Structural Causal Model (SCM)**, so there is a true
ground-truth answer. Grading is deterministic (numerical correctness within **acceptance bands**
for reasonable subjective choices), plus reasoning-quality checks, **ablation proof** (wrong
analyses must fail), and **trace audits for information leakage**. No rubric ambiguity.

### Reported scores
GPT‑5.6 Sol Pro **31.5%**, GPT‑5.6 Sol **28.7%**, Claude Opus 4.8 **16.0%**, **Gemini 3.5 Flash 8.1%**.
(GPT‑5 was <5% when the work began.) The benchmark is deliberately far from saturated.

### Access
**Partial public release:** ~**10 open problems on Hugging Face** + an interactive web UI, and a
**50-question subset routed to Artificial Analysis** for independent evaluation. The **full 129
remain proprietary.** → We can *self-measure* on the ~10–50 public problems; a full number requires
either OpenAI eval access or building our own SCM-simulated equivalents.

### Why Propab fits (specifically)
- **Deterministic grading == our honesty gate.** GeneBench‑Pro rewards exactly what Propab is
  designed to guarantee: a verifiable, exact answer, no hallucinated conclusion. Our verdict
  pipeline (deterministic `verification_method`, certify-before-confirm) maps directly onto
  "numerical correctness within an acceptance band."
- **We already built the tools.** This session's domain-capabilities work shipped the relevant
  primitives: `differential_expression`, `enrichment_analysis`, `sequence_align`, `motif_scan`,
  `structure_analysis`, plus the math/stats layer (`multiple_testing_correction` / BH‑FDR,
  `power_analysis`, `exact_linear_algebra`, `number_theory`, `combinatorial_enumeration`) and the
  bio methodology skills. Statistical/population genetics reasoning is squarely in that set.
- **Missing pieces to add:** PLINK 2.0 in the sandbox image; genetics libs (e.g. `hail`,
  `scikit-allel`, `statsmodels` survival, `pysam`); GWAS/PGx skills (association testing, LD, MSM
  Cox). These are *reusable tools/skills*, not per-question hacks — consistent with the
  domain-independence rule.

### How we'd test it
1. Pull the ~10 open HF problems + web-UI examples; encode each as an inspect_ai `Task` (reuse the
   `integrations/astabench/propab_solver.py` pattern: **one Propab campaign per problem**, agentic).
2. Provision the sandbox with PLINK 2.0 + genetics libs; run Propab (general-agent mode,
   gemini‑3.5‑flash) with a deterministic-answer output contract (`{answer, reasoning}`).
3. Grade against the released ground truth / acceptance bands; if not released for a problem,
   grade the reasoning trace manually to find failure modes.
4. **Loop:** forensic analysis → add the missing tool/skill → re-run → repeat (our proven loop).

### Effort / risk
- Effort: **medium** — sandbox provisioning + ~a dozen task encodings + genetics tools/skills.
- Risk: only a partial public set to self-score on; the tasks are genuinely hard (multi-stage).
- **Upside:** highest — deterministic grading + our existing bio/stats stack + a legible headline
  ("open-source scaffold on a small model vs. the 8.1% base"). **Recommend starting here.**

---

## 3. LifeSciBench — **P1 (closest to the litQA2 win)**

**Source:** OpenAI, *Introducing LifeSciBench* (openai.com/index/introducing-life-sci-bench/);
MarkTechPost / AI Weekly coverage. Built **with 173 PhD scientists** (biotech/pharma).

### What it tests
**750 expert-authored** real-world life-science research tasks. **7 workflows** — evidence
handling; analysis; design/optimization/prediction; scientific reasoning; validation & operations;
translation; scientific communication. **7 domains** — genomics → medicinal chemistry → clinical
→ translational. **79% are multi-step** (~4 steps avg). Each task is framed "as a scientist would
brief a colleague": prompt + context/artifacts + **free-response** answer. **1,062 attached
artifacts** (sequences, figures, tables, PDFs, chemical structures); **53% of tasks need ≥1 artifact.**

### Grading — **rubric (LLM-judge over expert rubrics)**
Not single reference answers: **19,020 rubric criteria (~25/task)**, each awarding points for a
specific fact, a reasoning step, or a numeric answer within tolerance. Two metrics:
- **Normalized rubric score** = points earned / total (partial credit).
- **Task pass rate** = fraction of tasks scoring **≥70%**.
Validated by **453 independent reviewers** (97% PhDs).

### Reported scores
| Model | Normalized | Pass rate |
|---|---|---|
| GPT‑Rosalind | 0.576 | **36.1%** |
| GPT‑5.5 | 0.519 | 25.7% |
| Gemini 3.1 Pro | 0.515 | 23.6% |
| GPT‑5.4 | 0.479 | 20.7% |
| Grok 4.3 | 0.399 | 13.0% |

### Access
Paper + "technical details" referenced, but **public availability of tasks/rubrics is unclear** and
has been explicitly questioned for reproducibility. → **Biggest practical blocker.** If the rubric
set isn't released we can't self-grade faithfully; we'd need to reconstruct a rubric-judge.

### Why Propab fits
- **This is the litQA2 shape** (rubric-graded, free-response, artifact-grounded research tasks) — the
  exact profile where our literature service went 0→0.78. We already have the retrieval/RAG +
  rubric-judge machinery (`services/literature/…`, `evaluator/astabench.py`).
- Multi-step + artifacts + scientific-communication categories map to our agent + literature +
  bio/stats tools.

### How we'd test it
1. If public: wrap tasks as inspect_ai samples; run one Propab campaign per task; grade with the
   released rubric via an LLM-judge harness (extend `services/literature/app/evaluator/`).
2. If not public: build a **held-out proxy set** in the same 7-workflow shape from public sources
   (papers + PDB/UniProt/ChEMBL artifacts) and a mirrored rubric — measure *relative* lift from our
   scaffold rather than the exact leaderboard number.

### Effort / risk
- Effort: **medium** (harness mostly exists) — dominated by data access + rubric-judge fidelity.
- Risk: **access/reproducibility** is the gate; rubric-judge variance can flatter or punish us.

---

## 4. Automated LLM Speedrunning Benchmark ("NanoGPT — improving model training") — **P2**

**Source:** *The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements*
(arXiv **2506.22419**, OpenReview `w98hMEjzu8`), built on **Keller Jordan's nanoGPT speedrun**
(train GPT‑2 to **val cross-entropy 3.28 on FineWeb** on **8×H100**, community-driven from ~45 min
down to <3 min since June 2024).

> **Honesty flag:** the user's framing ("start from a human baseline, improve time-to-train to a
> fixed quality target") matches this benchmark exactly, but I could **not independently confirm a
> specific GPT‑5.6 score on it** in search. Treat the *benchmark* as verified and well-specified;
> treat the "OpenAI reported X" as unconfirmed until we find the primary number.

### What it tests
Agent is given the **previous record's training script** (+ optionally one of **three hint formats**,
pseudocode → paper-like) and must **reimplement the next known speedup** (algorithmic and/or
hardware-aware). **19 records** in the ladder. Scored **deterministically** by wall-clock to reach
the target loss (speedup fraction vs. the known record). Finding: even SoTA reasoning LLMs +
scaffolds **struggle to reproduce already-known innovations even with detailed hints** — very
non-saturated.

### Why Propab (partly) fits — and the caveats
- **Deterministic, verifiable target** → fits our honesty gate perfectly; and it's an "improve X
  starting from a baseline, iterate" task — literally our research-iteration loop (construct →
  measure → escalate).
- **But:** it's **ML-systems engineering, not science discovery** — off our core domain — and it
  needs **8×H100** to score faithfully (real infra cost). We'd need CUDA/Triton/torch-compile skills
  our agent doesn't have.
- **Open-source**, so unlike the others we can run the *real* thing end-to-end.

### How we'd test it
1. Clone the benchmark repo; stand up an 8×H100 runner (cloud) — or a **scaled-down proxy** (fewer
   steps / smaller target) to iterate cheaply, accepting it's not the official number.
2. Wrap as an agentic task; let Propab's code sandbox propose+run training-script edits; grade by
   wall-clock-to-target.

### Effort / risk
- Effort: **high** (GPU infra + systems skills). Risk: off-domain; expensive to run for real.
- Verdict: **strong signal if we can afford it**, but sequence it after the bio benchmarks.

---

## 5. MedChemBench — **P3 / defer**

**Source:** OpenAI, *Introducing new capabilities to GPT‑Rosalind*
(openai.com/index/introducing-new-capabilities-to-gpt-rosalind/); RDWorld / TechTimes coverage.

### What it tests
Realistic medicinal-chemistry workflows: **multimodal chemical-structure understanding**,
structure–activity relationships (SAR), potency/toxicity prediction, ADME modeling, multi-parameter
**lead-optimization** decisions, and **retrosynthesis** planning. Reported: **GPT‑Rosalind 27.5%**
vs. GPT‑5.5 25.1% (7.2% fewer tokens). **Task count, grading detail, and public availability are
not disclosed** in coverage; appears **proprietary** and **multimodal**.

### Why it's lowest priority
- **Multimodal** (chemical-structure images) — Propab has no vision/chem-structure stack.
- **No public dataset** → can't self-measure.
- Needs cheminformatics tooling (RDKit, retrosynthesis models) we don't have.
- **Recommendation:** defer. Revisit only if we invest in a chemistry tool cluster (RDKit-based
  SAR/ADME + a structure encoder) — a large, separate build.

---

## 6. Reuse: Propab already has an external-benchmark harness

We do **not** start from scratch. Existing, proven scaffolding:
- **`integrations/astabench/propab_solver.py`** — an **inspect_ai solver** that runs **one Propab
  campaign per benchmark problem** (`inspect eval … propab_campaign`). This is the drop-in pattern
  for GeneBench‑Pro and the speedrun benchmark.
- **`integrations/astabench/discoverybench_solve.py` / `data_mount.py` / `replay_solver.py`** — data
  mounting + solve loop + replay.
- **litQA2:** `services/literature/app/evaluator/litqa2_live.py`, `scripts/run_litqa2_real.py` — the
  0→0.78 precedent and rubric/answer-grading path for LifeSciBench.
- **`asta-bench/inspect_evals/`** vendored — inspect_ai eval definitions to model new tasks on.
- Runners: `scripts/run_astabench_*.py`, `summarize_astabench_propab_run.py`, `score_option_b_solutions.py`.

**New work is mostly:** (a) task encodings for each benchmark, (b) sandbox provisioning (PLINK 2.0 +
genetics libs for GeneBench‑Pro), (c) domain skills (GWAS/PGx), (d) a rubric-judge for LifeSciBench.

---

## 7. Recommended sequencing

1. **P0 — GeneBench‑Pro (start now).** Best fit (deterministic grading = our honesty edge), we
   already own most of the bio/stats tools, partial public set exists, headline is legible
   ("scaffolded gemini‑3.5‑flash vs. 8.1% base, vs. 31.5% GPT‑5.6 Sol Pro"). First concrete step:
   pull the ~10 open HF problems, provision PLINK 2.0 + genetics libs in the sandbox, encode them
   via the `propab_solver` pattern, and run a baseline to see where we land before scaffolding.
2. **P1 — LifeSciBench** in parallel *iff* the tasks/rubrics are obtainable; it's the litQA2 shape and
   reuses the literature stack. If not public, build a mirrored proxy to measure relative lift.
3. **P2 — Automated LLM Speedrun** once GPU budget is justified; the only fully-open one, deterministic,
   but off-domain + expensive.
4. **P3 — MedChemBench** deferred pending a chemistry/multimodal tool investment.

**North-star framing:** the win condition is a **litQA2-style jump** — take our modest base model to
a number that beats much larger models via the scaffold, on a benchmark the field is actively
watching. GeneBench‑Pro is the highest-probability place to produce that headline.

---

## 8. Honesty caveats (what is verified vs. not)

- Model/date context (GPT‑5.6 Sol/Terra/Luna, GPT‑Rosalind, these benchmarks) is **past my training
  cutoff**; everything here is from **live web search on 2026‑07‑10**, not prior knowledge.
- **Verified via multiple sources:** GeneBench‑Pro (OpenAI + bioRxiv), LifeSciBench (OpenAI +
  MarkTechPost/AIWeekly), MedChemBench (OpenAI GPT‑Rosalind coverage), the Automated LLM
  Speedrunning Benchmark (arXiv 2506.22419).
- **Not independently confirmed:** the exact "GPT‑5.6 on NanoGPT" number the user referenced (the
  benchmark is real; the specific score is not confirmed). Several **public-availability** claims are
  uncertain (LifeSciBench rubric release; the proprietary majority of GeneBench‑Pro / MedChemBench).
- **All reported scores** above are OpenAI/press figures; we must **reproduce our own numbers on the
  public subsets** before claiming anything — same trust-nothing discipline as the discovery loop.

### Sources
- https://openai.com/index/previewing-gpt-5-6-sol/
- https://openai.com/index/introducing-genebench-pro/ · https://www.biorxiv.org/content/10.64898/2026.06.29.735386v2 · https://cdn.openai.com/pdf/21938268-21af-442f-af93-3b2249afb241/genebench-pro.pdf
- https://openai.com/index/introducing-life-sci-bench/ · https://www.marktechpost.com/2026/06/17/openai-releases-lifescibench-a-750-task-benchmark-grading-ai-models-on-real-life-science-research-with-expert-written-rubric/
- https://openai.com/index/introducing-new-capabilities-to-gpt-rosalind/
- https://arxiv.org/abs/2506.22419 · https://openreview.net/forum?id=w98hMEjzu8 · https://github.com/karpathy/nanogpt
