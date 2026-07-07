---
name: falsifiable-hypothesis-design
description: Turn an open gap into a novel, falsifiable, scoped hypothesis
phase: hypothesis
scope: core
priority: 10
---
You are proposing a hypothesis whose answer is currently UNKNOWN and, if resolved,
would change what this research area accepts as established. Follow this method:

1. **Anchor on an open gap.** Start from a specific gap in the provided literature
   (an unresolved bound, an uncharacterised mechanism, a conjecture without a proof
   or counterexample). If no gap is catalogued, identify precisely where the frontier
   of THIS question lies and target that — never settled ground.

2. **State a sharp, falsifiable claim.** The claim must predict a specific, checkable
   outcome. A reader must be able to say exactly what observation would REFUTE it.
   Avoid unfalsifiable hedges ("may", "could", "is related to"), tautologies, and
   restatements of the question.

3. **Make it novel, not a rediscovery.** Do NOT propose a claim whose answer is a
   known/tabulated value, a re-derivation of an established result, or a re-measurement
   of an already-characterised method. Do NOT restate a previously-tested claim with a
   different parameter (a parameter sweep of a settled finding is not a hypothesis).
   Prefer a claim that challenges or improves a current best-known result, proposes a
   genuinely new construction/mechanism, or states a structural conjecture the
   literature has not resolved.

4. **Scope it explicitly.** Every hypothesis must declare: the population (which
   instances/regime), the distribution (dataset/ensemble/simulator it holds on), the
   claimed generalization (where it should transfer — must differ from the
   distribution), at least one expected failure mode (a regime where it should break),
   and a concrete out-of-distribution / hold-out test that decides it BEFORE
   confirmation.

5. **Name a real test methodology** — an actual analysis the engine can run (e.g. a
   permutation/label-shuffle null, a hold-out comparison, an exhaustive check, a
   literature-baseline comparison), not a vague "we will analyze it".

A good hypothesis is one where you already know the single experiment that would kill
it, and that experiment is worth running because its outcome is genuinely uncertain.
