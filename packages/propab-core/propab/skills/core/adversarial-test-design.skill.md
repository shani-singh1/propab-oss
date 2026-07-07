---
name: adversarial-test-design
description: Design a test whose confirmation survives a real null, not a plausible story
phase: experiment
scope: core
priority: 20
---
A result is only a finding if it survives an adversarial attempt to explain it away.
When you specify how a hypothesis will be tested, design the test so a confirmation
CANNOT be an artifact:

1. **Specify the null explicitly.** State the concrete null model the effect must beat
   — e.g. a label/outcome permutation, a leave-one-group-out hold-out, a shuffle of the
   putative driver against the outcome. "It looks significant" is not a test; "the
   observed effect exceeds the 95th percentile of a 200-permutation shuffle null with
   p < 0.05" is.

2. **Prefer computation/measurement the agent controls over agent-supplied numbers.**
   Evidence whose inputs were typed by the reasoning agent (rather than computed in the
   sandbox) is untrusted and cannot confirm — design tests that run on real data or real
   computation.

3. **Guard against leakage and circularity.** Ensure the test cannot confirm by using
   the answer to produce the result: never "verify" a claim by looking up the very
   value the claim asserts. If features could proxy the group label (leakage), the
   hold-out must break that proxy.

4. **Deterministic vs statistical.** If the claim is exact (a proof, an exhaustive
   check, a construction), provide a checkable WITNESS (the object + an independent
   recomputation), not a confidence score. If the claim is statistical, provide the null
   statistics (effect size, null p95, permutation p). Shapeless evidence that is neither
   cannot confirm.

5. **State the failure regime up front.** Name at least one out-of-distribution or
   hold-out condition under which the effect should vanish, and test it BEFORE claiming
   confirmation. A finding that was never given a chance to fail is not a finding.

The bar: if a skeptic reran your exact test, they would be forced to the same verdict —
because the null, the witness, or the hold-out leaves no room for a just-so story.
