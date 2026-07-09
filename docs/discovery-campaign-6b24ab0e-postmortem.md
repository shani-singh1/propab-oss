# Discovery campaign 6b24ab0e — forensic postmortem (quantified)

**Question:** construct a B_3 (Sidon-type) set beating the best-known size for its
range — e.g. a 17-element B_3 subset where OEIS A396704 a(7)=16 — or rigorously
show 16 is optimal. Domain: `math_combinatorics`, reasoning ON.
**Outcome:** stopped after ~40 min (churning). 0 records found; honesty intact.

> Written from the DB (events / hypotheses / llm_calls), not from guesses. It
> **corrects** the initial stop-rationale (I had guessed "sandbox timeouts at large
> n" — the data shows experiments completed fine; the real issue is they never ran
> at large n).

## 1. Activity (quantified)
| metric | value |
|---|---|
| elapsed before stop | ~40 min (2384 s of 14400 s budget) |
| hypotheses (nodes) | 37 |
| worker dispatches | 127 → **~3.4 dispatches/node** (retunes) |
| domain-verification tool runs | 125 |
| orchestrator reasoning LLM calls | **128** (116K in / 14.6K out tokens, avg 9.2 s) |
| **worker think-act LLM calls** | **0** (math uses the domain verification path, not think-act) |
| setup LLM calls | 3 (skill_selection, hypothesis_ranking, hypothesis_generation) |

**Cost note:** the orchestrator's reasoning is the entire LLM cost here — 126
`reason_next` calls ≈ 130K tokens ≈ ~19 min of model time. Workers spent 0 LLM
(deterministic domain verification).

## 2. Outcome (quantified)
| verdict | count |
|---|---|
| inconclusive | 35 |
| refuted | 2 |
| confirmed | 0 |

- **4 `finding.certified`, ALL witness size = 16 at n=7** — valid size-16 B_3 sets
  in {0,1}^7 (the *correct* target object), independently certified, **none falsely
  claimed as a record** (matches known a(7)=16). The certifier + honesty gate
  worked exactly as intended.

## 3. Scale actually tested (the smoking gun)
| n tested | # experiments | metric |
|---|---|---|
| 200 | **14** | sidon_density |
| ~100–200 (sweeps) | ~19 | sidon_ratio_to_sqrt_n / bose_chowla_vs_greedy_ratio |
| 1000 | 1 | sidon_density |
| **2500** | **1** | sidon_density |
| 7 | 2 | b3_binary_cube_size / sidon_density |

The verifier's own output said: *"Greedy Sidon at n=7 … is a known-range
computation, not open-problem evidence. Use multi-n sweep (n≥500)."* — yet **14
experiments still ran at n=200** and only **2** ever reached n≥1000.

## 4. Root causes (ranked, with quantified severity + evidence)

### R1 — Retune changes never reach the experiment (SEVERITY: HIGH — the primary failure)
- **Evidence:** the orchestrator correctly + repeatedly reasoned "known range n=200;
  retune with a multi-n sweep n≥500" — **57 of 128 reasoning steps (45%)** carried
  this same theme — yet 14 experiments ran at n=200 and only 2 at n≥1000. Retune is
  applied by appending a hint to the node's `test_methodology` text; the
  `math_combinatorics` verifier does **not** parse that hint, so it re-runs its
  default construction at the default scale. **Retunes are effectively no-ops on the
  actual computation.**
- **Impact:** the campaign can never reach the regime where a record could exist; it
  spends budget re-deriving known-range values and the orchestrator loops on the
  same diagnosis.

### R2 — Object/metric spread away from the target (SEVERITY: MEDIUM)
- **Evidence:** the target is a B_3 set in {0,1}^7 (metric `b3_binary_cube_size`), but
  ~33/37 experiments computed `sidon_density` / `sidon_ratio_to_sqrt_n` /
  `bose_chowla_vs_greedy_ratio` — Sidon sets in [1,N], a *related but different*
  problem. Only the 4 certified runs used the correct object.
- **Nuance:** exploring Sidon theory to inform a B_3 construction is *reasonable*
  strategy, so this is partly the orchestrator's choice — but it burned most of the
  budget on known-range Sidon density instead of escalating the actual B_3 search.

### R3 — No worker adaptivity for math (SEVERITY: MEDIUM, enables R1)
- **Evidence:** 0 worker think-act LLM calls. Math workers run the fixed domain
  verification path, so they can't adapt params to the orchestrator's request. The
  only way a retune could change the scale is if the domain verifier accepts a scale
  parameter — it currently doesn't.

### R4 — Reasoning churn (SEVERITY: MEDIUM — symptom of R1, not a cause)
- **Evidence:** 45% of reasoning stuck on the same unresolved scale issue; ~3.4
  dispatches/node. The reasoning was coherent; it couldn't escape because execution
  never changed. The tuned prompt (commit f88de41: "drop after repeated failure")
  reduces the churn but does not fix the root (R1).

## 5. What actually worked (honest positives)
- **Honesty architecture: perfect.** 0 false confirms, 4 certified size-16 sets none
  claimed as records, verifier self-diagnosed known-range vs open-problem correctly.
- **Orchestrator reasoning: coherent + correct.** It correctly identified both the
  scale problem and (once) a B_2-vs-B_3 object mixup. The intelligence is there — the
  *execution layer* failed it.
- **No crashes, no timeouts, deterministic verification ran clean.**

## 6. Fixes (ranked)
1. **[R1, do first] Thread retune/strategy parameters into the domain experiment.**
   The orchestrator's `retune_changes` (e.g. "n≥500", "use B_3 binary cube not Sidon
   density") must become structured params the domain verifier consumes — not free
   text appended to methodology. Add a params channel: `orchestrator_reason_next` →
   retune params → worker dispatch payload → domain `run_verification(..., params)`.
2. **[R2] Make the math domain escalate on the target object.** For an A396704-style
   question, run the `b3_binary_cube` search (ILS + CP-SAT for exact/optimality) at
   the target n, rather than defaulting to Sidon-density sweeps.
3. **[R3] Let the domain verifier accept a scale/param spec** (n range, construction
   family, budget) so retunes can change what's computed.
4. **[R4] Keep the f88de41 "drop on repeated failure" prompt** to bound churn while
   R1 lands.

## 7. What to CHECK in every future discovery campaign (the watch-metrics)
- **dispatches/node** (≈3.4 here) — high means churn / no-op retunes.
- **% reasoning on a single repeated theme** (45% here) — high means the orchestrator
  is stuck on an issue execution won't resolve.
- **scale/param actually executed vs requested** — did the retune's params reach the
  computation? (Here: no.)
- **object/metric match** — is the tested metric the question's target object? (Here:
  mostly no.)
- **think-act vs domain-path** (0 worker LLM here) — know which mode the workers ran.
- **certified-but-not-record count** — honest rediscoveries (4 here) confirm the
  certifier works.
- **confirmed/false-confirm** — must stay honest (0 false confirms here ✓).
