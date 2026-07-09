# Propab Frontend Design Guide — from 1 to 1000

> The campaign view today is at ~1: a vertical stack of bordered cards
> (`OrchestratorGroupCard`, `RoundGroup`, worker rows), every event boxed and
> **labeled** with equal weight ("ORCHESTRATOR", "Round 2", verdict bars,
> sparklines, stat pills), inside a narrow `max-w-[720px]` centered column that
> leaves big empty gutters. This guide is how we get to 1000.

## 0. The one idea
**A campaign is a research narrative unfolding live. The UI's only job is to let
the user *read the science as it happens* — like watching a great scientist think
and work — with zero mental overhead.** Signal loud, mechanics quiet, everything
else gone.

If a screen element doesn't help the user understand *what was discovered, what
the orchestrator is thinking, or what's happening right now*, it should not exist.

## 1. Non-negotiable principles
1. **Hierarchy by IMPORTANCE, not by event type.** A confirmed finding, a
   reasoning step, and a tool call are NOT visual peers. Rank them; render them at
   different weights. Today they're all cards — wrong.
2. **Narrative, not widgets.** The orchestrator's reasoning is *prose in a
   timeline*, read top-to-bottom like a lab notebook — not a stack of labeled
   collapsible group-cards. Kill `OrchestratorGroupCard`'s "Orchestrator" header
   and box; the reasoning is just… the text.
3. **Progressive disclosure.** Collapse the noise (tool calls, dispatches, node
   marks, raw events) into subtle, dim, inline expanders. Surface only the
   conclusion. Never a stack of cards the user must scan.
4. **Full-bleed + typographic.** The app owns the whole viewport — no page
   margins, no wasted gutters. Treat text as a well-set document: a comfortable
   reading *measure* for the narrative column only, generous line-height, a real
   type scale. Chrome is tight; reading area breathes.
5. **One accent.** Monochrome base (neutral inks). Color is reserved for
   **verdicts** (green confirmed / red refuted / muted inconclusive) and the
   single **breakthrough** state, so those *pop*. No decorative color, no rainbow
   of chips.
6. **Calm.** Remove sparklines, verdict bars, stat-pill rows, uppercase labels,
   and anything that reads as "dashboard." Every remaining element earns its place.

## 2. The three tiers (the whole hierarchy)
- **Tier 1 — Findings / discovery** (rare, LOUD). A confirmed/certified result or
  a breakthrough. This is the *only* place a distinct, prominent block ("card") is
  warranted: the claim, the evidence (metric + null + certification), stated
  plainly and proudly. Appears inline in the timeline where it happened *and*
  summarized in the header.
- **Tier 2 — The reasoning narrative** (the main story, ~80% of attention). The
  orchestrator's thoughts + decisions as a **flowing timeline of prose**, newest
  at the live edge. This is what the user reads. No boxes.
- **Tier 3 — Mechanics** (constant, QUIET). Tool calls, worker dispatches, node
  marks, brief intermediate thoughts. Dimmed, small, and **collapsed** into a
  single subtle inline row when they occur in a run ("· ran 3 tools, dispatched 2
  experiments ▸"). Expandable for the curious; invisible to everyone else.

## 3. Layout — full-bleed, three zones
```
┌───────────────────────────────────────────────────────────────┐
│  HUD  (one slim line: question · status · hyps/confirmed ·      │  fixed
│        elapsed/budget · breakthrough meter)                     │
├──────────────────────────────────────────┬────────────────────┤
│                                          │  WORKERS            │
│   REASONING NARRATIVE (timeline)         │  (compact rows,     │
│   — the orchestrator narrating,          │   click → drawer)   │  fills
│     prose, findings inline, mechanics    │                     │  height
│     collapsed. Auto-follows live edge.   │  TREE (mini-map,    │
│   readable measure (~640–760px), but     │   nodes by verdict) │
│   the ZONE spans the remaining width.    │                     │
├──────────────────────────────────────────┴────────────────────┤
│  Composer (ask / steer)                                         │  docked
└───────────────────────────────────────────────────────────────┘
```
- **No page margins.** The shell is `h-screen w-screen`, flex column; header fixed,
  body flex-1 min-h-0, composer docked. The center scrolls; the right rail scrolls
  independently.
- **Center zone** spans the available width; the *prose* inside it is capped to a
  readable measure and left-aligned (not centered in a sea of whitespace) with the
  timestamp gutter on the left. The empty right side of the measure is where
  inline finding-blocks and pull-quotes can breathe — not dead space.
- **Right rail** (collapsible, ~320px): workers as one-line rows (status dot ·
  hypothesis (truncated) · elapsed), click opens a drawer with that worker's *real*
  experiment (design, code, tools, result). All worker `llm.*`/`tool.*` chatter
  lives HERE — never in the center narrative. Below workers, a small live tree map
  (nodes colored by verdict) so the user sees the search shape.

## 4. The narrative timeline — exactly how to render (the heart)
Replace `OrchestratorGroupCard` + `RoundGroup` with a single continuous timeline of
rows sharing one left **timestamp gutter** (dim, mono, `+m:ss`). Row types:

- **Reasoning** → a plain sentence/short paragraph in the reading color. No label,
  no box, no dot-with-"Orchestrator". Just: *"Since variance failed to predict
  held-out expression, testing tau + mean instead."*
- **Decision** → the verdict inline: a small colored dot + one line
  *"Refuted — no signal beyond the null"* with the hypothesis as a quiet quoted
  clause beneath, a tiny metric/`p` chip, and the next move as a subtle
  *"→ deepening this line"*. The verdict *color* is the only emphasis.
- **Generation boundary** (new round) → a **hairline divider** with a tiny gutter
  label ("round 2"), NOT a boxed "Round 2" card with bars/sparklines/stats.
- **Mechanics run** → when ≥2 mechanic events sit between two reasoning beats,
  render ONE dim inline line: *"· ran 3 tools · dispatched 2 experiments"* with a
  ▸ that expands to the raw list. Single stray mechanics can be dropped entirely.
- **Finding / breakthrough** → the Tier-1 block: breaks the flow, colored rule,
  the claim in larger type, evidence line, "independently certified" badge when
  true. This is the payoff — make it feel like one.

Live edge: keep the streaming "what it's doing right now" line pinned at the bottom
(the typewriter `LiveEdge` is good — keep it), auto-follow with the jump-to-live
pill.

## 5. Type & color
- **Type scale:** finding headline (16–18px, semibold) › reasoning body (13.5–14px,
  1.6 line-height, normal weight) › decision line (13px) › mechanics + timestamps
  (10.5–11px, dim, mono). The narrative should read like prose, so *body weight is
  normal*, not the current semibold-everywhere.
- **Color:** one neutral ink ramp for text (ink / ink-2 / ink-3 / ink-4). Accent
  ONLY: `--green` confirmed, `--red` refuted, `--text3`/muted inconclusive, and one
  celebratory treatment for breakthrough. Kill per-widget colors.
- **Density:** tight vertical rhythm between rows (8–10px), a little more around
  generation dividers and findings. No card padding stacking up.

## 6. Do / Don't
**Don't:** wrap events in labeled cards · show an "ORCHESTRATOR" header · stack
collapsible group-cards · give mechanics the same weight as reasoning · render
sparklines/verdict-bars/stat-pill rows in the main flow · center a 720px column in
a wide empty page · scatter chips · use semibold for body prose.
**Do:** one continuous timeline that reads like a document · findings loud, reasoning
plain, mechanics dim+collapsed · verdict color is the only accent · full-bleed shell
with a readable prose measure · workers + tree in the right rail · timestamps in a
quiet gutter.

## 7. Concrete component changes
- **`NarrativeStream.tsx`:** delete `OrchestratorGroupCard`, `RoundGroup`,
  `VerdictBar`, `Sparkline`, `BestDelta`, `VerdictChips` from the main flow.
  Introduce a flat `TimelineRow` union (reasoning | decision | generation-divider |
  mechanics-run | finding) driven off the model. Remove the `max-w-[720px]` shell
  wrapper; put the measure on the prose rows only.
- **`Campaign.tsx`:** make the shell full-bleed (drop `px-[26px]` gutters into the
  content, not the frame); the narrative zone fills width, prose capped by measure;
  ensure `h-screen`/`min-h-0` chain so it uses full height with no outer margin.
- **`model.ts`/`events.ts`:** the model already coalesces orchestrator + round
  groups — repurpose it to emit a *flat* ordered timeline with a `tier`
  (finding/reasoning/decision/mechanics) + `generationBreak` markers, instead of
  nested group objects. Mechanics get folded into `mechanics-run` items.
- **Right rail (`RightPanel`/`WorkersPanel`/`WorkerCard`):** keep worker drill-down;
  ensure worker chatter never leaks to center; add the small verdict-colored tree
  map.
- **Remove "weird stuff at random places":** the always-pinned `DiscoveryHero` when
  empty, the round stat-pill rows, the setup-card "no verdicts yet / live" quirk —
  surface findings inline + in the HUD instead.

## 8. Definition of done (beta)
- Full viewport used, no page margins; narrative reads as one clean document.
- A newcomer can watch a live campaign and, without clicking anything, follow *what
  the orchestrator is thinking and finding*; mechanics are present but never noisy.
- Findings are unmistakable; verdicts are the only color; everything else is calm.
- Verified live (preview) on a running campaign in both light and dark, no console
  errors, full width + height.
