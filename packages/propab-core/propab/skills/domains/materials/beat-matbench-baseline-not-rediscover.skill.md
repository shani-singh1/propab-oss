---
name: beat-matbench-baseline-not-rediscover
description: Guardrail against reproducing a known Matbench baseline relationship instead of establishing a novel structure-property law
phase: hypothesis
scope: materials
priority: 32
---
On the Matbench `matbench_dielectric` task the reference results are PUBLISHED, so a
claim can look confirmed while being a rediscovery. Reject the rediscovery framings
before committing.

**1. Reproducing a known baseline is not a discovery.** The Matbench leaderboard
already establishes what standard descriptor pipelines achieve: the Automatminer
reference scores MAE ≈ 0.31 and current SOTA (coGN / MODNet) ≈ 0.27 on log-refractive
index; the underlying per-material dielectric tensors are tabulated in the Materials
Project (Petousis 2017). A hypothesis whose content is "structure predicts the
dielectric constant" — the very relationship those baselines already exploit — is a
restatement of known materials informatics, not a new law.
- To be novel, your claim must do something the baseline does NOT: a specific,
  physically-motivated descriptor→dielectric law that generalizes ACROSS crystal systems
  (which the aggregate benchmark does not isolate), a documented WHERE-IT-BREAKS
  (a crystal system on which the accepted descriptor relationship fails), or a descriptor
  that beats the family-mean baseline out-of-family by a margin the benchmark's
  in-distribution CV never tests.

**2. Do not "verify" a tabulated per-material value.** Asserting a dielectric constant
that already sits in the Materials Project tensor dataset is a lookup, not a measurement.
The finding must be a RELATIONSHIP that survives leave-one-crystal-system-out, not a
re-report of a catalogued number.

**3. Frame novelty as cross-system transfer, not in-sample accuracy.** Matbench's own
scoring is nested CV over the pooled data; the OPEN question this domain can decide is
whether a structure→property law TRANSFERS to an unseen crystal system. A modest LOFO R²
that clears the family baseline and the label-shuffle null on a HELD-OUT system is a
more novel result than a high pooled R² that merely matches the leaderboard.

Litmus test before you submit: (a) is the whole claim already implied by "descriptors
predict dielectric constant on Matbench"? → rediscovery; re-scope to a specific,
sign-carrying, crystal-system-contingent law. (b) Does confirmation require beating the
family-mean baseline out-of-family, or just fitting the pooled data? If only the latter,
you are re-measuring a benchmark, not discovering a law.
