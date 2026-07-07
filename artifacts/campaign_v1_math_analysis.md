# Math v1-acceptance campaign — honest analysis (campaign 1ae74abd)

Question: novel cap-set / Sidon / AP-free constructions in F_3^n (n 6..10) beating the
greedy lower bound, with checkable witnesses. Budget 1h; ran 63 min to budget_exhausted.

## Headline (honest)
- 25 hypotheses, depth to gen ~5, **0 confirmed**, 2 refuted, 23 inconclusive, 4 pending.
- 0 fabricated confirms; real computations produced real metrics + p-values where they ran.
- Clean multi-round dynamics + budget management (9->13->18->29 nodes, budget_exhausted).

## What the campaign VALIDATES (v1 pipeline works end-to-end + honest)
- **Generation is diverse + novel** (skills working): AP-free density power-law decay,
  additive-energy of greedy Sidon vs Bose-Chowla, k = cN^0.5/(log N) scaling, sumset-size
  bimodality, symmetry-seeded / cyclic-invariant hybrid constructions, product constructions.
  Expansion types spread across alternative(11)/mechanistic(6)/generalization/diagnostic/
  boundary — NOT the old "greedy CLP ratio = X" monoculture. Genuine frontier math.
- **Honesty is intact**: computed constructions honestly scored BELOW the 0.875 baseline
  (ratios 0.50-0.67) -> inconclusive, never a false "improvement". Real p-values.
- The whole path (skills -> generation -> real compute -> honest verdict -> budget) ran.

## Bottlenecks the campaign SURFACED (the value of running it)
1. **Verification code TIMEOUT is the #1 blocker — 16 of 23 inconclusive are `code_timeout`.**
   The LLM proposes parameters (cap sets n=10-12, Sidon N=10^5) whose exact computation
   exceeds the sandbox code budget -> timeout -> inconclusive. Even a genuinely computable
   result never lands. FIX CANDIDATES (deliberate, not reactive): (a) guide generation to
   computable parameter ranges via the domain skill; (b) raise the deterministic-math code
   budget; (c) a graceful "computed partial / best-so-far witness" instead of a hard timeout.
2. **When it DOES compute, the proposed constructions don't beat known** (ratios < baseline).
   This is the honest discovery ceiling: proposing a construction that actually improves a
   cap-set/Sidon bound is a genuine open problem. Not a bug — the honest hard truth.

## Verdict
v1 math pipeline is HONEST and works end-to-end; generation is genuinely good. The gap to a
*confirmed* result is (1) a fixable compute-budget/parameter-scoping issue and (2) the real
mathematical difficulty of beating known bounds. No reactive fix+recampaign — record + decide.
