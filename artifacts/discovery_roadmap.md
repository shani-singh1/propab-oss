# Propab → Actual Novel Discovery: Roadmap & State

Living record of the push to make Propab do real frontier science (novel discovery,
not rediscovery). Sourced from a real 2h campaign analysis + a per-domain completeness
audit + live smoke checks. Trust-nothing: every claim here was verified against code
or a live run.

## Central thesis (from the domain completeness audit)
Propab's **verification machinery is strong and largely honest**; the **inputs are what
block novel discovery** — real data + real computation + populated novelty-grounding
sources are missing for 5 of 7 domains. Fixing inputs, not machinery, is the path.

## Domain completeness scorecard (0–3 per axis; audit-verified)
| Domain | Data real | Verifier rigor | Lit profile | Tools | Tab. knowns | Routing | Tests | Total | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| math_combinatorics | 3 | 2 | 3 | 3 | 3 | 3 | 3 | 20 | DEEP |
| materials | 3 | 3 | 2 | 3 | 1 | 1 | 2 | 15 | DEEP |
| mandrake | 2 | 3 | 2 | 2 | 0 | 2 | 2 | 13 | PARTIAL |
| graph_invariants | 1 | 1→2 | 2 | 2 | 0 | 2 | 2 | 10 | PARTIAL |
| genomics | 0 | 2 | 2 | 2 | 0 | 2 | 1 | 9 | PARTIAL |
| enzyme_kinetics | 0 | 2 | 2 | 2 | 0 | 2 | 1 | 9 | PARTIAL |
| network_diffusion | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 5 | SKELETAL |

## Verification machinery — DONE and verified (honesty layer)
- **DISC1/DISC2** — rediscovery demotion (novelty gate + best_known_table flag). 765 tests.
- **F1** — the adversarial artifact gate now applies to EVERY domain's plugin verdict
  (was materials/mandrake only). genomics/enzyme lofo confirms now face a real null;
  deterministic proofs pass; shapeless evidence can't confirm. `sub_agent_loop.py`.
- **V3** — closed the `deterministic` gate-bypass hole: classify_evidence_type now
  requires an explicit proof method / deterministic flag (a bare counter + innocuous
  method name no longer bypasses). graph_invariants now emits a REAL label-shuffle
  permutation null (routes as lofo, gated). `verdict_pipeline.py` + graph verifier.
- Audit F3/F4/F5/F7 were already fixed by V2/DOM2b/A1 (stale audit base).

## Generation — DONE and validated LIVE (novelty engine, input side)
- **GEN-OVERHAUL** — all template/fabrication machinery deleted; gap-driven prompt
  ("ADVANCE what is known, not re-measure it"); honest `[]` on empty; domain-general
  anti-retest guard. Zero domain vocabulary in the core prompt.
- **Live smoke check (real Gemini + real prior):** produced genuinely NOVEL cap-set
  hypotheses — AGL(n,3)-invariant local search to beat the ~2.11 growth exponent,
  Fourier-coefficient-minimizing search, Hamming-weight ordering, structured product
  perturbation, a real null hypothesis. All scope-valid, none dropped, NONE rediscovery.
  Night-and-day vs the campaign's "greedy CLP ratio = X" monoculture.

## The remaining blockers = INPUTS (in-flight build wave, audit-prioritized)
- **Agent A · genomics + enzyme real data** — replace RNG-baked synthetic (signal baked
  into the generator) with real GTEx / BRENDA data so the honest LOFO machinery finds
  real signal. + honesty tests. [running]
- **Agent B · novelty-input engine** — populate tabulation_sources/open_problem_sources
  for materials/enzyme/genomics/mandrake (only math has them), AND fix `/prior` returning
  `open_gaps=0` (live finding) so gap-driven generation gets real frontier gaps. [running]
- **Agent C · math real cap-set computation** — verifier COMPUTES real caps (with a
  witness) above n≤8 instead of reading CAP_SET_BEST_KNOWN; the one domain whose novelty
  is real mathematics. [running]

## Next waves (not yet started)
- graph_invariants real SNAP data (on disk) — honest now (V3 null) but synthetic → can
  only rediscover; real data unlocks novelty.
- network_diffusion — build a real SIS/SIR verifier on SNAP graphs, or mark scope-only
  (it can't verify anything today — dead weight).
- materials/mandrake content routing (matches() is explicit-tag-only).
- Base contract: block `discovery_worthy` for synthetic-data domains.

## External repo review — K-Dense-AI/scientific-agent-skills
**NO-GO on wholesale integration.** It's ~148 prompt-wrapper "agent skills" around
existing packages/APIs — an assistant toolkit for rediscovery + report generation, with
essentially NO adversarial verification (Propab's differentiator). Narrow PARTIAL:
(1) borrow **Arbor**'s held-out dev/test merge-gate discipline as a concept (Propab's
null-based gate already satisfies the spirit); (2) optionally wrap **HypoGeniC**
(peer-reviewed) as a candidate GENERATOR feeding Propab's own verdict pipeline — never as
a verifier. Skip the ~140 bio/chem wrappers and all ideation/peer-review skills (they are
the unfalsifiable-fabrication pattern we deleted).

## Campaign discipline (standing)
No new campaign until there is genuinely nothing but a campaign that can validate a fix.
Validate with live smoke checks (cheap, few LLM calls), unit/integration suites, and
per-component probes first. Reactive fix→campaign→fix loops are forbidden.
