# General-agent redesign — dissolve the plugin architecture

> Status: **DESIGN** (2026-07-09). Re-derived from a clear-eyed, unbiased read of
> the actual execution paths, after the 6b24ab0e postmortem exposed that
> plugin-domain "workers" don't experiment at all.

## 1. The clear questions (answered from code, no bias)

**Q: Why do plugins exist?** A shortcut — a pre-built, hand-audited, deterministic
verifier per domain, so a campaign gets fast + honest verification without an LLM
writing buggy verification code. The honesty primitives (correct null, certifier,
leakage guards) live inside these verifiers.

**Q: Does a worker actually experiment?** Mostly no.
- `sub_agent_loop.py:1844-1879` — for **any** domain with a plugin (all 12), the
  worker calls `domain_plugin.run_verification(...)` and returns. No experiment.
- `sub_agent_loop.py:1979` — even for non-plugin domains, the **default**
  `sub_agent_plan_source` is `heuristic` (a static tool plan). The real experimenter
  (`think_act.decide_next_action`) runs only when `plan_source ∈ {llm, hybrid}`.
- So the LLM experimenter is the **exception**, gated by config + non-plugin.

**Q: Does the plugin model scale to arbitrary research questions?** No. A plugin is
one verifier per domain, with hardcoded branches per *known problem*
(`math_combinatorics.run_verification` → cap_set / sumset / ap-free / **else sidon**;
`run_combinatorics_experiment` → `_is_b3_binary_cube_hypothesis` else sidon). A new
question type needs a new hardcoded branch. You cannot pre-author a verifier for
every possible research question. It is fundamentally not scalable, and it silently
mis-routes anything it wasn't pre-built for (postmortem: 33/37 → wrong object).

**Q: Where does the orchestrator's instruction get lost?** There is no interpreter
between the orchestrator and the computation. The verifier regex/keyword-parses the
hypothesis text (`_extract_n = n=(\d+)` can't read "n≥500"; object defaults to
sidon). The rich instruction is compressed to text and lossily parsed.

## 2. The correct design (orchestrator ↔ workers + tools + skills)

```
question
  ▼
ORCHESTRATOR (brain) — reasons, uses tools+skills, writes hypotheses,
  dispatches each to a worker with a structured instruction
  ▼                                              ▲ raw evidence
WORKER = general LLM research agent (think-act, the ONLY mode)
  · given {hypothesis, orchestrator instruction, tools_allowed, skills}
  · DESIGNS the experiment, WRITES + RUNS code in the sandbox,
    CALLS TOOLS (never reimplements trusted primitives), reads SKILLS
  · returns RAW evidence (no verdict)
  ▼
ORCHESTRATOR judges with the central deterministic honesty gate, decides
  next (deepen/retune/spawn/drop), loops.
```

- **No domain plugins. No domain routing. No domain-gated tool clusters.** The agent
  is general; tool/skill selection is by the problem, with the *full* catalog
  available (audience-scoped orch/worker) and cheap descriptions to choose from.
- **Honesty is preserved by three things, not by plugins:**
  1. **Trusted tools** the worker MUST call instead of writing its own: a correct
     permutation/label-shuffle-null tool, the B_3 `certify_b3_record` certifier tool,
     significance/bootstrap tools (already exist, with anti-cheat + zero-variance
     guards), data-loader tools. The worker can't fake a null — it calls the tool.
  2. **The orchestrator's central verdict/honesty gate** (C2 — deterministic; already
     built) run on the raw evidence.
  3. **Rigor skills** (experiment design, confound/leakage control, which null when).

## 3. What each plugin part becomes
| plugin part | becomes |
|---|---|
| domain `run_verification` (hardcoded construction) | **DELETED** (the "vibe computation") |
| the trusted verifier primitive (correct null, `find_max_b3`, `certify_b3_record`, LOFO+null) | a **TOOL** in `propab.tools` the worker calls |
| data adapter (GTEx, materials, enzyme frames) | a **data-loader TOOL** |
| `objective_spec` (is_ml, metric, baseline frame) | inferred by the orchestrator/worker from the experiment; the honesty frame is a property of the evidence shape (already how `classify_evidence_type` works), not a domain |
| `classify_verdict` (domain override) | **DELETED**; one central `verdict_pipeline` (C2) judges by evidence shape |
| `matches`/routing | **DELETED**; no domain routing |

## 4. Migration — verifiable stages (trace ONE real question end-to-end each stage)
This is a large re-architecture; do it staged, and at each stage run a real campaign
and trace one hypothesis through the whole pipe (the discipline missed before).

- **S0 — make the experimenter the default + only mode.** Set think-act as the worker
  path; remove the heuristic-by-default and the plugin-verification bypass so EVERY
  question goes through the general agent. Keep the significance-gathering gate.
  *Trace:* one genomics + one math hypothesis actually run agent-written code.
- **S1 — expose trusted primitives as tools.** Wrap the correct null, `find_max_b3` +
  `certify_b3_record`, the LOFO+null helpers, and the data loaders as `TOOL_SPEC`
  tools (audience/worker), with the anti-cheat guards. *Trace:* a worker calls the
  certifier tool and the null tool on real data.
- **S2 — structured instruction channel.** Orchestrator emits a structured brief
  (target object, scale, constructions, budget) into the dispatch payload; the worker
  agent consumes it as its task (not free text parsed by a regex). *Trace:* an
  "n≥500 B_3 search" instruction actually runs at n≥500.
- **S3 — delete the plugin verifiers + domain routing.** Remove `run_verification`
  hardcoded constructions, `route_domain`, `question_domain`(gone), domain tool
  clustering → full catalog with selection. Keep only the tools + skills + the central
  honesty gate. *Trace:* a brand-new question type (never pre-built) runs correctly.
- **S4 — reconcile the register** (this supersedes D1/D2/E-plugin entries): the
  domain-plugin abstraction is retired; "domain-independence" is achieved by a general
  agent, not per-domain plugins.

## 5. Risks / honest tradeoffs
- **Correctness of agent-written code.** The reason plugins existed. Mitigation: the
  worker CALLS trusted tools for the honesty-critical steps (null, certifier, stats) —
  it never writes its own null. Verification correctness lives in tools, not in
  agent-written code. The orchestrator's central gate is the backstop.
- **Cost/latency.** Every question becomes an LLM experimenter loop (more tokens than
  a fixed verifier). Acceptable — it's the point, and matches the frontier-science goal.
- **Determinism/reproducibility.** Agent experiments are less reproducible than a fixed
  verifier; tools + seeds + the certifier restore the reproducible core for records.
- **Do NOT lose the honesty guarantees** during migration — they move into tools +
  the central gate, and must be tested at each stage (the null tool must have the
  within-group target-shuffle; the certifier must be the independent re-verify).
