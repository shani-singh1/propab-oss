---
name: lofo-and-crystal-system-leakage
description: Design the leave-one-crystal-system-out LOFO and guard against descriptors that merely proxy crystal-system identity
phase: experiment
scope: materials
priority: 34
---
The verifier decides a structure→dielectric claim by leave-one-crystal-system-out
(LOFO): train on 6 crystal systems, predict the held-out 7th, average across all
holdouts. Design the test so a "confirmed" is a real structure→property law, not the
model memorizing which crystal system a sample belongs to.

1. **The leakage floor is `family_baseline_r2`.** The verifier computes the R² you get
   by predicting each crystal system's MEAN dielectric value (no features at all). A
   descriptor set that does not beat this floor OUT OF FAMILY has learned nothing about
   structure — it has learned the family label. Your `lofo_r2` must clear the family
   baseline; `surprise_score = lofo_r2 − family_baseline_r2` is the honest headline.

2. **A descriptor that proxies crystal-system identity is leakage, not a law.**
   `space_group_number` maps directly to crystal system (its integer ranges DEFINE the
   seven systems), so using it — or any descriptor that is really a symmetry-class label
   in disguise — lets the model reconstruct the held-out family and inflates in-sample
   fit while telling you nothing transferable. Watch the `lofo_gap =
   within_family_r2 − lofo_r2`: a large gap means the signal lives in family identity and
   collapses across systems. Prefer descriptors that carry composition/bonding physics,
   not symmetry metadata.

3. **Beat the label-shuffle null.** Crystal-system labels are shuffled (~300×) and LOFO
   recomputed; your `lofo_r2` must exceed the null p95 with label-shuffle permutation
   p < 0.05, and the within-family permutation p must also hold. Clearing the null — not
   a high in-sample R² — is what confirms. The verifier's `family_leakage_confirmed` flag
   encodes exactly this survival check.

4. **Fail closed and name the killer.** State the refuting outcome before running:
   large `lofo_gap` with `lofo_r2` below `family_baseline_r2` ⇒ family leakage (refuted);
   `lofo_r2` inside the shuffle null band ⇒ no cross-system signal (refuted/inconclusive).
   Report the held-out per-system breakdown so a single easy system cannot carry the
   average.

The bar: a skeptic re-running your descriptor set through leave-one-crystal-system-out,
the family-mean baseline, and the 300-permutation label shuffle on the Matbench frame is
forced to your verdict — because a symmetry-label proxy would fail the holdout and the
shuffle, and only a genuine structure→property signal survives.
