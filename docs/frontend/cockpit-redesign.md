# Propab Frontend — the Discovery Cockpit (0→1000 design spec)

Status: design spec for the frontend upgrade. Owner: Agent 3. Target: turn the
current (solid, functional) campaign UI into a best-in-class **autonomous-discovery
cockpit** — legible when an agent works for hours across hundreds of hypotheses and
~400 LLM calls, and *thrilling* the moment it crosses a frontier.

---

## 0. Honest assessment of what exists today

The current frontend is **not** at 0/1 — it's a competent ~level-4. What's already good
(and must be PRESERVED, not rebuilt):

- A coherent design system: CSS-variable tokens (`--text/2/3/4`, `--green/red`, `--chip`,
  `--border`, per-surface `--*Bg`), dark-default + light, macOS-window shell
  (`App.tsx`), `pp-scroll`/`pp-row` utilities, `animate-ppulse/pdots`, `motion-reduce`.
- A clean event-derivation core (`lib/model.ts`): one pure O(n) pass → `rounds`,
  `workers`, `inFlight`, `narrative`, `counts`. This is the right architecture — extend it.
- Progressive disclosure: collapsible round cards, expandable worker rows, lazy raw-event
  reveal. Store de-dupes by `event_id` and backfills on reconnect (`store.ts`).
- Three-panel shell: Navigator (left) · NarrativeStream (center) · RightPanel with
  Workers/Tasks/Tree/Beliefs tabs + Ticker (right).

What keeps it at level 4 (the gaps 1000 closes): it presents *activity* well but not the
**discovery** — the single most important thing (best-vs-best-known, distance to a record,
the witness) is nowhere on screen; special moments (breakthrough, confirmed, paper) look
like ordinary rows; it doesn't yet scale calmly to thousands of events (no virtualization,
search, filter, or jump); the tree, beliefs, and cost views are thin; there's no command
palette, keyboard nav, or persisted UI state. It informs; it doesn't yet *command attention
and reward*.

---

## 1. Vision & principles

**It's a cockpit, not a log.** A researcher (or an investor watching a demo) should, in 3
seconds, know: *is it winning?* Then be able to drill from that one number down to the exact
line of code a sub-agent is running right now — without ever being firehosed.

Principles:
1. **Discovery-first.** The hero of every screen is progress toward a verified result:
   best-so-far vs best-known, distance to the breakthrough threshold, the current witness.
2. **Calm at scale.** Hundreds of events must feel like a tidy story. Group, summarize,
   virtualize; reveal detail only on intent.
3. **Moments matter.** A confirmed hypothesis, a candidate record, a finished paper are
   *events with weight* — rich cards, motion, color, not a gray row.
4. **Transparency on demand.** Every worker, every sandbox run, every LLM call is openable
   to its raw truth — one click, never forced.
5. **One design language.** Everything uses the existing tokens; it must be beautiful in
   both themes and respect `motion-reduce`. No jarring, no monster dependencies.

---

## 2. Workstreams (concrete specs)

### A. Persistent Campaign HUD (the vital-signs bar)
Replace the thin header line in `Campaign.tsx` with a compact, always-visible HUD strip
(its own component `CampaignHud.tsx`). Data is already in `campaign.summary` /
`campaign.campaign` (`types.ts`): `baseline_metric`, `best_metric`, `improvement_pct`,
`elapsed_sec`, `remaining_sec`, `breakthrough_threshold_pct`, `total_hypotheses`,
`total_confirmed`, `tree.*`, plus `model.counts`.
- **Vital signs, left→right:** status dot+label · question (truncate, full on hover) ·
  **Best vs baseline** (with a mini distance-to-breakthrough meter) · hypotheses tested ·
  confirmed · running workers (live) · LLM count · errors · budget burn-down (elapsed /
  remaining as a thin progress bar) · paper link.
- The **distance-to-breakthrough meter** is the centerpiece: a slim horizontal bar from
  baseline → threshold, with a marker at `best_metric`. When `best_metric` crosses the
  threshold, the bar turns green and pulses.

### B. Discovery-first: hero status + special cards  *(user: "nice UI cards for certain events")*
- **`DiscoveryHero.tsx`** — a card pinned at the top of the center column that summarizes
  the discovery state: the best finding so far (`campaign.best_finding`), what it beats /
  falls short of, and — for a certified record — the witness and certification checks. In
  the "no result yet" state it shows the target and the current best-so-far honestly
  (e.g. "best B_3 set found: 16 · published best-known: 16 · need: 17").
- **Special event cards** in the narrative (extend `NarrativeStream` with a card registry
  keyed by `event_type`), each visually distinct from a plain round row:
  - `campaign.breakthrough` / candidate record → **full-width celebratory hero card**:
    green, animated, the metric it beat, the witness JSON (collapsible), the certification
    booleans, a "verify independently" affordance. This is the money shot.
  - hypothesis **confirmed** → highlight card: the statement, evidence summary, confidence,
    the metric vs best-known.
  - `paper.ready` → paper card: title/abstract preview, page/figure counts, download.
  - `baseline.measured` → a small framed card (what baseline, how).
  - `round.completed` with high `marginal_return` → accented.
  - **belief changed** (strengthened/weakened/abandoned) → a subtle inline card.
- A `refuted` result is honest, not sad — show it as *information gained* (a closed branch),
  never red-alarm.

### C. Center narrative upgrades  *(user: orchestrator messages stream; group events as they complete)*
- **Streaming orchestrator voice.** Surface `campaign.phase`/orchestrator narration as a
  live, single-line "what it's doing right now" at the active edge (typewriter-ish, calm),
  and fold discrete orchestrator milestones into the story. The center is the orchestrator's
  narrative; workers live on the right (as designed).
- **Richer round cards:** a stacked mini verdict bar (confirmed/refuted/inconclusive as
  proportional segments), an activity sparkline, per-round best-metric delta, and the
  compute stat line already present. Keep collapse; live round auto-expands.
- **Motion:** new narrative items enter with a soft translate/fade (respect `motion-reduce`).
  Use the View Transitions API or CSS transitions — no heavy animation dep.
- **Follow-the-live-edge:** auto-scroll while pinned to bottom; when the user scrolls up,
  show a "↓ jump to live (N new)" pill.

### D. Right panel — depth per tab + new tabs
- **Workers:** add a sticky filter/search bar (by status chip + free-text over hypothesis
  text), sort (running → recent → confidence), and optional group-by-round. Each worker card
  gains a one-line result summary (verdict · metric vs best-known · confidence) and, for
  math, the best-so-far witness size. Virtualize the list.
- **Tasks (Background Tasks):** this is the "watch the machine think" tab — make it excellent.
  Live per-task duration, grouped by kind (already), each openable into a detail drawer that
  shows the *actual* running code (sandbox), the LLM purpose/model/prompt preview, or the
  tool args. A subtle throb while in-flight. Empty state that reads as "idle, healthy," not
  "nothing here."
- **Tree:** the weakest panel — rebuild as an **interactive** graph: zoom/pan, click a node
  → detail popover (statement, verdict, confidence, evidence_summary, expansion_type), verdict
  coloring, depth rings, hover to highlight the path to root, and a legend. Auto-fit; handle
  100+ nodes without becoming a hairball (collapse deep/exhausted subtrees).
- **Beliefs:** show belief **evolution** — status (active/strengthened/weakened/abandoned)
  with a transition indicator, supporting vs contradicting node counts as a tiny diverging
  bar, and exhaustion_rounds. Order by most-recently-changed.
- **New tab — Compute/Cost:** LLM calls, tool calls, code runs, errors over time; per-purpose
  breakdown; latency and token/cost once the backend emits them (see §3); budget burn-down.
- **New tab (or HUD-integrated) — Metrics:** the metric trajectory (best_metric over rounds)
  vs baseline and threshold — a small line/step chart. (Follow `dataviz` skill conventions.)

### E. Event overflow at scale  *(user: "events overflow")*
- **Virtualized lists** for workers, tasks, raw events, and the Ticker (windowing; no dep
  needed — a tiny intersection/measure-based virtualizer, or `@tanstack/react-virtual` if a
  dep is acceptable). Target: smooth at 2000+ items.
- **Global search + filter:** a `/`-triggered search over hypotheses, events, and workers
  (by text, verdict, domain, status). Results jump to the item.
- **Round table-of-contents / jump:** a slim rail or the HUD lets you jump to any round;
  collapse-all / expand-all.
- **Density toggle** (comfortable/compact) persisted.
- Raise/clarify `store.ts` `MAX_EVENTS` behavior with a visible "history trimmed" affordance
  so a long campaign never silently loses the head without a cue.

### F. Global craft & smoothness  *(user: "smooth to use", "1000 level")*
- **Command palette (⌘K):** jump to a round, a worker, a tab; toggle theme; open paper.
- **Keyboard nav:** `j/k` across narrative items, `o` to open/close, `/` search, `g t` go to
  tree, etc.
- **Persisted UI state** (localStorage): right-panel open/closed, active tab, theme, density.
- **Theme toggle** surfaced in the Navigator or HUD (tokens already support light/dark).
- **Empty/loading/error states** with personality and skeletons (not bare text).
- **Toasts** for weighty events (breakthrough, first confirm, fatal error) — subtle,
  dismissible, respect reduced motion.
- **Responsive:** right panel becomes an overlay/drawer under ~1000px; HUD wraps; center
  stays readable. Verify at mobile/tablet/desktop.
- **A11y:** focus rings, `aria-expanded`/`aria-live` on the live edge, color-not-the-only-signal.

### G. Navigator (left) upgrades
Campaign list with: search/filter, status chips, sort (recent/best/confirmed), per-item
live indicator + micro-stats (best vs baseline, confirmed, running), grouped by status, and
a prominent New Campaign CTA. (Read the current `Navigator.tsx` first; extend, don't replace
the shell.)

---

## 3. Backend event-contract dependencies (assign to a backend agent)
The UI can only be as rich as the stream. Required emissions (some already flagged by the
frontend build):
1. `llm.response`: add `tokens_in`, `tokens_out`, `duration_ms` (and a cost estimate hook) —
   enables the Compute/Cost tab. (`packages/propab-core/propab/llm.py`.)
2. A `call_id` correlating `llm.prompt`→`llm.response` (replaces fragile FIFO pairing in
   `model.ts::buildInFlight`).
3. A `round` field on `agent.*`/worker events (authoritative round attribution instead of
   positional).
4. SSE `id:`/`Last-Event-ID` replay on `/stream` for seamless reconnection.
5. Worker **heartbeat/progress** events so "running" ≠ merely "no terminal event yet"
   (distinguishes a live worker from a stuck one).
6. **Discovery events:** emit a first-class `finding.best_updated` / candidate-record /
   `certification` event carrying the witness + certification booleans + metric-vs-best-known,
   so the Discovery Hero and breakthrough card render from real data, not inference.

---

## 4. Implementation plan → parallel agents (worktree-isolated)
Each workstream is a subagent on its **own git worktree** off `main`, disjoint file sets,
merged via PR. Order: F-util (virtualizer/persistence/palette scaffolding) and §3 backend can
start immediately; A/B depend on nothing; C/D build on the model; E depends on the virtualizer.

- **FE-1 — HUD + Discovery Hero + special cards** (§A, §B): `CampaignHud.tsx`,
  `DiscoveryHero.tsx`, narrative card registry; touches `Campaign.tsx` header + new components.
- **FE-2 — Center narrative** (§C): `NarrativeStream.tsx` + round-card internals + motion +
  follow-the-live-edge.
- **FE-3 — Right panel depth** (§D): `WorkersPanel`, `TasksPanel`, `WorkerDetail`,
  `HypothesisTreeView` (interactive rebuild), `BeliefsView`, new `ComputePanel`/`MetricsPanel`.
- **FE-4 — Global craft** (§E, §F, §G): virtualizer util, command palette, keyboard, persisted
  store, Navigator, responsive/a11y pass.
- **BE-1 — event contract** (§3).

Shared contract: extend `lib/model.ts` (add cost/metric-trajectory/discovery derivations) as
the single source; agents coordinate on its exported types. `model.ts` is the one file
multiple FE agents read — freeze its public shape early (I'll land the type additions first).

---

## 5. Guardrails / non-goals
- Keep the existing design tokens + macOS-window aesthetic; this is an *elevation*, not a
  reskin. Both themes must stay first-class; `motion-reduce` always honored.
- No heavyweight animation/charting frameworks; prefer CSS/SVG + the existing system. A small,
  well-justified virtualizer dep is acceptable.
- Every panel must degrade gracefully when its data/events are absent (early campaign, trimmed
  history, missing backend fields).
- Performance budget: the model recompute is O(n) per event; keep it memoized and, if a real
  campaign pushes it, move to incremental derivation rather than re-deriving from scratch.
- Verify in the live preview (dev server) at desktop + mobile + both themes before calling any
  workstream done.

---

## 6. Definition of done (per workstream)
Green production build (`tsc -b && vite build`), no console errors in preview, the model-logic
assertions still pass, visibly correct at desktop/tablet/mobile in light+dark, and — for the
discovery-facing pieces — a convincing render against the offline demo (`/campaign/demo`).
