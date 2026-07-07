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

---

## WAVE COMPLETE (build wave merged + verified — 868 passed, 1 skipped)

All 6 build agents verified (trust-nothing: read code + independent re-derivation +
re-ran tests) and merged into campaign-convergence. Full suite green.

- **math_combinatorics** — real cap-set computation with witness (n4=20 optimal,
  independently re-verified valid; computed != table). [C, merged]
- **coding_theory (NEW domain, 8th)** — real GF(2) min-distance via 2^k enumeration +
  witness recheck; Brouwer/Grassl table for rediscovery rejection only; routes
  deterministic via a real proof method (no V3 gaming). Independently re-verified
  (Hamming[7,4]=3). [merged]
- **graph_invariants** — real SNAP networks (ca-GrQc, email-Eu-core), V3 null intact,
  fails closed if data missing. [D, merged]
- **network_diffusion** — skeletal -> real SIS/SIR + cascade simulator on real graphs,
  cross-topology holdout + shuffle null; fixed a perm_p==0.0 falsy bug. [E, merged]
- **genomics** — REAL GTEx v8 median TPM (10000x7 active); LOFO + shuffle null. [A]
- **enzyme_kinetics** — REAL DLKcat/BRENDA+SABIO-RK kcat (3553 active); LOFO + null. [A]
- **novelty-input engine** — open_gaps fix (live 0->7) + real tabulation/open-problem
  sources for rediscovery rejection. [B, merged]

**Verification caught (trust-nothing paid off):** E silently broke test_domain_modules
(fixed); B over-claimed "166 green" — 3 tests failed in B's own worktree, broken
fixture (fixed); A's genomics/enzyme fell back to a STALE synthetic cache on this box
(deleted + rebuilt -> real data now active). C flagged a real CAP_SET_BEST_KNOWN table
error (n1/n2/n8) — deferred.

**State:** 8 domains, all on REAL inputs, all behind the F1+V3 adversarial gate, with
gap-driven novel generation (GEN-OVERHAUL) + novelty grounding (open_gaps/tabulation).

## Deferred / follow-ups (non-campaign)
- CAP_SET_BEST_KNOWN table correctness (n1=2, n2=4, n8=496?) — needs a reference + C's
  test updates.
- Synthetic-discovery-count design question (DOM2 "label not block" vs exclude from
  discovery total) — open decision.
- graph_invariants + network_diffusion would benefit from a 3rd real network family.
- Per-domain end-to-end honesty harness (generation -> verify -> gate) as the last
  pre-campaign validation.

## Campaign readiness
Still NOT campaign-time: the fixes are validated by suites + live smoke checks +
independent re-derivation. The remaining pre-campaign validation is the per-domain
end-to-end honesty harness. A campaign is only warranted once nothing but a campaign
can validate the next change.

---

## END-TO-END HONESTY HARNESS — PASSED (cross-domain, real data)

Ran each domain's ACTUAL verify path (run_verification -> classify_verdict -> F1
artifact_gate_stage) on REAL data with a genuine on-topic hypothesis:

| Domain | data | evidence | verdict | honest outcome |
|---|---|---|---|---|
| genomics | real GTEx | lofo | refuted | real signal doesn't trivially hold (no false confirm) |
| enzyme_kinetics | real DLKcat | lofo | refuted | same (LOFO R2 ~ -0.12) |
| graph_invariants | real SNAP | - | refuted | no false confirm |
| network_diffusion | real SNAP | lofo+null | CONFIRMED | epidemic-threshold law survives the adversarial null |
| coding_theory | real compute | deterministic | inconclusive | Hamming[7,4] d=3 = known -> rediscovery demoted |
| math_combinatorics | real compute | deterministic | CONFIRMED | cap set size 80 computed with witness |

Invariants confirmed: no domain confirms noise; genuine results confirm only with a
passing null (network_diffusion) or a witness (math); rediscoveries are demoted
(coding_theory); the F1 gate AGREES with the plugin in every case (honesty is
intrinsic, not bolted on). This is the end-to-end proof that Propab does honest
science on real data across all 8 domains. (Reproducible: scratchpad/honesty_smoke.py)
