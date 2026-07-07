---
name: scope-and-generalization-discipline
description: Bound a claim to where it holds and pre-commit the out-of-distribution test that would break it
phase: hypothesis
scope: core
priority: 15
---
An unscoped claim is unfalsifiable by default: if you never said where it should hold,
no observation can contradict it. Before a hypothesis leaves the drawing board, pin down
its boundaries so that "it works" has a precise meaning and "it broke" is decidable.

1. **Population — over what instances is the claim asserted?** Name the exact set of
   instances/regime the claim ranges over. "In general" is not a population. A claim about
   a handful of cherry-picked instances is a claim about those instances only; do not let it
   silently inflate into a universal statement.

2. **Distribution — on what generating process was it (or will it be) observed?** State the
   dataset, ensemble, sampler, or generator the supporting evidence comes from. A claim is
   only ever supported *on the distribution it was measured on*; everything beyond that is a
   generalization that must be earned, not assumed.

3. **Claimed generalization — where should it transfer, and why?** Name a target regime that
   DIFFERS from the observation distribution (larger scale, a shifted parameter, an unseen
   subgroup, a different generator) where you predict the claim still holds, and give the
   mechanistic reason you expect transfer. If you cannot name a transfer target different from
   where you measured, the "finding" is a description of one dataset, not a general result.

4. **Failure modes — where should it break?** A claim that predicts success everywhere
   predicts nothing. Name at least one regime where the mechanism should stop working and
   the claim should be FALSE. If you cannot describe a condition under which it fails, the
   claim is either trivial or unfalsifiable — reformulate it until it has an edge.

5. **Out-of-distribution / hold-out test — pre-committed, decided before confirmation.**
   Specify a concrete test drawn from OUTSIDE the observation distribution (a held-out split,
   a scaled-up regime, an unseen subgroup) whose outcome you commit to BEFORE running it, and
   state in advance which outcome confirms and which refutes. A generalization "verified" only
   on the same distribution it was fit on is circular and confirms nothing.

Calibrate the claim's strength to its scope. A narrow, well-bounded result you can defend
beats a sweeping one you cannot. If the evidence only supports the narrow version, claim the
narrow version — and say explicitly that the broader generalization remains open.
