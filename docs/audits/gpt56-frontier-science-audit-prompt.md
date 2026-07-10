# Prompt for GPT‑5.6 — Read‑only frontier‑science audit of the Propab codebase

> Paste everything below the line into GPT‑5.6 (Sol/Terra), with the repository mounted read‑only.
> The model's ONLY write is the single report file named at the end. It must not modify source,
> run mutating commands, commit, or push.

---

You are a principal research‑systems engineer and auditor. Your task is to **investigate and map the
entire Propab codebase, strictly READ‑ONLY**, and produce **one** detailed report. Read, grep, trace,
and reason. Do **not** edit any source file, run any command that mutates state (no installs, no DB
writes, no `git` mutations, no deploys), and do not push. The only file you may create is the final
report named at the end.

## What Propab is (context, verify against the code — don't trust this summary)

Propab is an autonomous, multi‑domain **scientific‑discovery engine**. Its stated north star is to do
**frontier science**: (a) make a genuine *novel discovery* on an open problem (e.g. beat a record in
extremal combinatorics), and (b) top **active external science benchmarks** (GeneBench‑Pro,
LifeSciBench, etc.) the way it once took the litQA2 benchmark from ~0 to 0.78.

Rough architecture (confirm precisely from the code): a Dockerised stack — `services/orchestrator`
(campaign loop, hypothesis‑tree growth, verdict authority), `services/worker` (a general‑agent
think‑act loop: `sub_agent_loop.py`, `think_act.py`, `sandbox.py`), a trusted **tool registry**
(`packages/propab-core/propab/tools/`), a **skills** system (methodology markdown,
`packages/propab-core/propab/skills/`), a **verdict/honesty pipeline**
(`packages/propab-core/propab/verdict_pipeline.py`, `significance.py`, the domain plugins in
`domain_modules/`), and a `services/literature` retrieval service. Start from `CLAUDE.md`,
`docs/architecture/`, and `docs/component_map.md`.

## The core question you must answer

**Why can't Propab do frontier science yet?** There are two concrete, observed symptoms — find the
*root* architectural and implementation causes of each, and everything adjacent:

1. **Weak search on open problems.** On genuinely open discovery targets its finders (greedy /
   random‑restart / metaheuristic / CP‑SAT) stall and cannot beat well‑optimised records. Is this
   fundamental, or is the architecture starving/mis‑driving the search?
2. **No scaffold lift on benchmarks.** On GeneBench‑Pro (a quantitative‑execution genomics benchmark)
   Propab scores at *base‑model level* (~10% on the public 10, ≈ the raw base model). Layering
   Propab's real scaffold (tool registry + skills) did **not** help and slightly **hurt** (0 tool
   calls — no tool fit the task; the injected skills were research‑*framing* prose that diluted
   focus). Contrast: litQA2 lifted 0→0.78 because the scaffold (literature retrieval) *directly
   supplied the missing capability*. Diagnose exactly where and why the scaffold fails to add value.

## Investigation dimensions — map each thoroughly, with file:line evidence

1. **Correctness bugs & latent defects** across the whole pipeline: orchestrator campaign loop,
   worker think‑act loop, the code sandbox, each tool, the verdict/honesty pipeline, the literature
   service. Look for silent‑failure paths, mis‑wired gates, prompt/format bugs, race conditions,
   swallowed exceptions, and anything that would make a run *look* fine while being wrong.
2. **The general‑agent think‑act loop** (`services/worker/think_act.py`, `sub_agent_loop.py`): is the
   planning/decision quality good? Assess the stop‑gates (`_stop_needs_evidence`), the code‑step
   budget, the finalize/answer‑extraction handling, tool selection over a flat catalog, the
   significance‑vs‑verification two‑path gate, and the warm‑start seeding. Where does it waste
   steps, mis‑select, or terminate early?
3. **Tool + skill scaffold** (`propab/tools/`, `propab/skills/`): are the tools the right primitives
   for the domains targeted, or a mismatched/under‑covering set? Are the skills the right *content*
   (executable methodology vs. hypothesis‑framing prose)? How good is tool selection on a 60‑tool
   flat catalog? Why did 0 tools get used on a genomics‑analysis task?
4. **Honesty / verdict pipeline** (`verdict_pipeline.py`, `significance.py`, `domain_modules/*`):
   soundness of the confirm/refute/inconclusive logic; over‑ or under‑gating; the false‑confirm and
   false‑negative surface; whether the recent record‑reference‑corroboration guard is complete.
5. **Search strength**: the finder/construction machinery for open problems. Is the architecture even
   *capable* of strong search (portfolio, restarts, learned guidance, exact methods), or is it
   structurally weak? What would it take to make search a first‑class, strong component?
6. **Orchestration & long‑horizon convergence**: hypothesis‑generation quality (are hypotheses
   templated/boilerplate — check any scope‑gate failures), deepen‑vs‑respawn behaviour, whether a
   campaign actually *converges* over a long horizon or drifts/repeats.
7. **Base‑model dependence**: where the outcome is bottlenecked by the base model's own reasoning
   (so no scaffold can help) vs. where the scaffold *could* add value but doesn't. Be explicit about
   which failures are model‑bound and which are fixable in the harness.
8. **Anything else** blocking a litQA2‑style benchmark lift or a genuine discovery.

## How to work

- Build a **component map** first (entry points → orchestrator → worker → tools/skills → verdict →
  literature). Cite the real files.
- **Trace real execution paths** end‑to‑end: (i) a campaign from launch to verdict; (ii) a single
  worker sub‑agent run (think‑act loop, a code step, a tool call, the stop‑gate, the verdict); (iii)
  a benchmark run under `integrations/`. Follow the actual data, not the docs.
- Read the **actual code**. Every finding must cite `path:line` and quote the relevant snippet.
- **Rank findings by impact** on frontier‑science capability, and clearly separate three buckets:
  (a) correctness bugs, (b) architectural flaws, (c) capability gaps (incl. model‑bound limits).
- For each finding: **root cause**, **evidence (file:line)**, **impact on frontier science**, and a
  **concrete fix direction** (what to change, and the expected effect).
- Be honest and specific. No hand‑waving, no restating docs as findings. If something is fine, say so.
  If the real blocker is the base model rather than the harness, say that plainly.

## Output (the only file you create)

Write **one** report to `docs/audits/frontier-science-audit-<YYYY-MM-DD>.md` containing:

1. **Executive summary** — the 3–5 root causes that most block frontier science, in priority order.
2. **Component map** — how the system actually fits together (with file references).
3. **Ranked findings** — bucketed (correctness bugs / architectural flaws / capability gaps), each
   with root cause, file:line evidence, impact, and fix direction.
4. **The scaffold‑lift diagnosis** — a focused section on *why* the tools+skills scaffold fails to
   lift the base model, and what class of scaffold (if any) would.
5. **Prioritised roadmap** — the ordered set of changes that would most move Propab toward frontier
   science, with the expected impact and rough effort of each.

Do not modify anything else in the repository.
