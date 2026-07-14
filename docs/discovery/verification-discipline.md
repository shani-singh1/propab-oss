# The verification discipline (how we avoid fooling ourselves)

> The operating protocol for the discovery engine (`propab/evolve/`). Derived from two sources: the
> prompt OpenAI used to drive the CDC proof, and our own record of reporting results that dissolved.

## The core claim

There is **no magical harness**. We have the actual CDC prompt; every mechanism in it is prompt-level
and bookkeeping-level. What OpenAI built was not secret infrastructure — it was a **strong
verification layer**, so that when the model proposes something, the proposal can be checked
accurately and adversarially.

That is the whole method, and it is fully portable. The one thing that does *not* port is the base
model's research-level mathematical ability — which is exactly why our targets are **search-shaped**
(the verifier carries the result, not the model's IQ). The transferable part is precisely the part we
need.

## Our own track record (why this doc exists)

We have twice reported a positive result that dissolved under scrutiny:

1. A **+0.04** "win" that was 100% LLM-judge noise on two subjective tasks. The objective tasks moved
   by exactly zero.
2. A **+0.60** "win" where the baseline model had been asked to analyze a CSV file **it was never
   given**. It scored 0.00 by construction.

In both cases **the math was fine and the comparison was garbage.** An exact verifier checks the
*math*. Nothing was checking the *claim*. That gap is what `evolve/auditor.py` exists to close.

**The lesson, stated as a rule: interrogate the comparison, not just the mechanism.**

## The five mechanisms (from the CDC prompt)

**1. Negative specification.** Enumerate every way to fool yourself *before* the search starts, and
ban it explicitly. The CDC prompt ruled out: special classes, wrong-multiplicity covers, bounded-length
variants, *reductions to another unproved conjecture*, computational verification up to a fixed size,
and counterexamples without a nonexistence certificate.
→ **Our version:** every `Problem.describe()` must state what does **not** count. Reproducing a
tabulated code is not a discovery. A graph outside the required class is not a counterexample. A finite
set never settles Erdős #143.

**2. Adversarial agents throughout.** "*Every candidate proof must be checked for … circular use of an
equivalent CDC statement.*" There are agents whose only job is to **kill** the result.
→ **Our version:** `Auditor` (WS-6). Default to REJECT. If a check cannot positively confirm, that is a
kill, not a pass. Nothing reaches the Ledger without surviving it.

**3. Anti-groupthink.** "*Do not tell most agents the currently favored approach.*" Plus an explicit
**registry of approach families** grouped by mathematical *idea*, redirecting agents when one family
dominates, and cross-pollinating **only after** independent development.
→ **Our version:** islands sample from their *own* parents, not the global champion; migration is late
and limited; programs carry a family tag and sampling is biased toward underexplored families.

**4. Anti-bullshit.** "*Require concrete lemmas, constructions, equations, or counterexamples. Reject
status reports, vague optimism, and claims that an unproved statement is 'routine.'*"
→ **Our version:** the verifier is the **sole** authority on score. Never trust anything a generated
program claims about itself. A result with no witness is not a result.

**5. Persistence.** "*Do not return merely because current approaches fail… spend at least 8 hours.*"
→ **Our version:** the kill criterion is **"the engine is broken"** (it cannot rediscover a known
positive control), never **"we haven't found anything yet."** Those are completely different signals
and must never be confused.

## The one-command rule

A result is only a result if a third party can re-check it **from the exported bundle alone** — the
witness plus a runnable verifier, in one command, in a fresh process, without trusting us. This is
copied deliberately from the CDC release: they didn't ask anyone to believe them; they shipped a proof
the kernel re-checks from a clean clone. `Ledger.export_publishable()` must meet this bar.

## Branch awareness

The CDC prompt says: *"Assume for purposes of this task that a complete affirmative proof exists."*
They told the model **which branch to search**, because they were confident of the answer.

We usually are not. Be explicit about the branch and its prior:
- **ECC records:** a better code plausibly exists in open cells → search is well-posed.
- **Graph conjectures:** we hunt counterexamples → a positive control proves the hunt can succeed.
- **Erdős #143:** every partial result (Erdős, Behrend, ESS, KLL) points toward the conjecture being
  **true**, so the refutation branch is probably **empty**. Bounded side-bet only. Never imply a finite
  search could settle it.
