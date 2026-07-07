---
name: evidence-honesty-and-calibration
description: Match the verdict to the kind and strength of evidence — confirm only what the evidence forces, stay inconclusive otherwise
phase: evidence
scope: core
priority: 30
---
The verdict you emit must be entailed by the evidence you hold — no more. The failure mode
to avoid is upgrading a suggestive result into a confirmed finding. Grade the evidence by
KIND first, then calibrate the claim to its strength.

1. **Classify the evidence before judging it.**
   - **Deterministic / exact** — a proof, an exhaustive check, or a construction that produces
     a concrete object. It can CONFIRM outright, but only when accompanied by a checkable
     WITNESS (the object plus an independent recomputation), not a narrative that it must be so.
   - **Statistical** — an effect measured against a null. It can support a claim only with the
     null quantified: effect size, the null distribution (e.g. a permutation p95), and the
     resulting p-value or equivalent. A number without its null is not statistical evidence.
   - **Shapeless / unknown** — a plausible story, an agent-typed value, a single unreplicated
     observation with no null and no witness. It CANNOT confirm anything. Its only honest
     verdicts are "suggestive, inconclusive" or "refuted".

2. **Never let one run confirm.** A single favorable outcome is consistent with luck, a seed,
   a leak, or a bug. Confirmation requires the result to survive something it could have failed:
   a replication across seeds/splits, a hold-out, a null it beats, or an independent
   recomputation of the witness. If the result was never given a way to fail, it is not evidence.

3. **Calibrate the words to the evidence.** Reserve "confirmed" for a beaten null or a
   verified witness. Use "supported" for a real but single-condition effect that has not yet
   been stress-tested out of distribution. Use "suggestive / inconclusive" for anything
   shapeless. Use "refuted" when the predicted outcome failed. Do not smuggle strength through
   adjectives ("strong", "clear", "striking") that the statistics do not license.

4. **Report the disconfirming evidence too.** State the effect size AND its uncertainty, the
   failed conditions alongside the passing ones, and the assumptions the verdict rests on. A
   confirmation that hides the runs that went the other way is not a confirmation.

5. **When exact and statistical evidence conflict, the exact witness wins** — but first
   re-examine the witness for a bug, since a correct exact object cannot be overruled by a
   noisy estimate, yet a broken one can fool you into thinking it was.

The honest default is inconclusive. Move off it only in the direction the evidence forces,
and only as far as the weakest link in the evidence allows.
