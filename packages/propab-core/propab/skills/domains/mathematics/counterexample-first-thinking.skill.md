---
name: counterexample-first-thinking
description: Try hard to BREAK a universal conjecture — bounded exhaustive plus targeted randomized search — before spending any effort proving it; "no counterexample found" is evidence, never a proof
phase: experiment
scope: mathematics
priority: 26
---
When you hold a universal conjecture — "for all n, P(n)", "every such object has this property" —
the cheapest and most informative first move is to attack it, not defend it. A single
counterexample settles the question outright and saves an entire proof effort; failing to find one
after real effort tells you the conjecture is worth proving. Reach for the disproof first.

1. **Search adversarially before you invest in a proof.** Enumerate the small regime exhaustively
   (`combinatorial_enumeration`, `counterexample_search`) and probe the large regime with
   randomized and targeted search. The goal is refutation: you are trying to make P(n) false, not
   to admire cases where it holds.

2. **Aim at where it is most likely to break.** Random sampling under-weights the dangerous
   cases. Deliberately test boundaries, extremal and degenerate configurations, the smallest and
   largest admissible parameters, and any case the conjecture's own structure makes precarious. A
   counterexample almost always lives at an edge, not in the bulk.

3. **A counterexample is self-certifying — verify it, then you are done.** Re-evaluate the
   predicate P on the candidate independently; if it genuinely fails, the conjecture is REFUTED,
   full stop, and no proof of it can exist. This is the one place a single computed object closes a
   question completely, so make sure the failing case is real and not a bug in your predicate.

4. **"No counterexample found" is EVIDENCE, and only over the range you covered.** Not finding a
   counterexample in N cases makes the conjecture more plausible — but it is never a proof, and its
   strength is exactly the region searched. Report the region precisely ("checked all n ≤ 40
   exhaustively and 10^6 random instances up to n = 10^4, none found"), and never round that up to
   "holds" or "confirmed".

5. **Only then commit to a proof.** A conjecture that has survived a serious refutation attempt is
   a good candidate for exhaustive/UNSAT or algebraic proof (see `experiment-design-in-mathematics`
   and `what-counts-as-a-proof`). The search that failed to break it does not substitute for that
   proof.

The bar: report the exact search you ran and the exact range it covered, and keep the verdict
"no counterexample found in [range] — unproven" distinct from "proved". The dishonest move is
promoting a failed refutation into a confirmation: absence of a counterexample is the beginning of
a proof effort, not the end of one.
