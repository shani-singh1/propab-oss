// Derives structured, legible state from the flat Propab event stream.
//
// A real campaign emits hundreds of events over hours (orchestrator phases,
// hypothesis generation, ~400 LLM calls, worker experiments, verification,
// synthesis, paper compilation). Showing them raw is a firehose. This module
// coalesces them into three views the UI can render calmly:
//
//   • rounds     — the campaign segmented into research rounds, each summarized
//                  (hypotheses generated, verdict tallies) and expandable.
//   • workers    — sub-agents grouped by hypothesis, with lifecycle + activity.
//   • inFlight   — the units of work running *right now* (worker experiments,
//                  sandbox code, LLM calls, tool calls) for the tasks panel.
//   • narrative  — the ordered center-panel story: setup + round groups +
//                  standalone lifecycle milestones.
//
// Everything is a pure function of the event list so it can be memoized and
// recomputed cheaply on each streamed event (single O(n) pass).

import type { CampaignState, CampaignSummary, PropabEvent, Verdict } from "../types";

export type WorkerStatus = "running" | "confirmed" | "refuted" | "inconclusive" | "failed";

export interface WorkerStep {
  index: number;
  event: PropabEvent;
  kind: "tool" | "code" | "think" | "stop" | "other";
  label: string;
  detail?: string;
}

export interface Worker {
  hypothesisId: string;
  shortId: string;
  text: string;
  status: WorkerStatus;
  verdict: Verdict | null;
  confidence: number | null;
  startedAt: string | null;
  endedAt: string | null;
  /** monotonic ms since start, or since last activity while running */
  lastActivityAt: string | null;
  steps: WorkerStep[];
  /** latest generated code, if any */
  currentCode: string | null;
  llmCalls: number;
  toolCalls: number;
  codeRuns: number;
  errors: number;
  events: PropabEvent[];
  round: number | null;
}

export type InFlightKind = "experiment" | "code" | "llm" | "tool";

export interface InFlightTask {
  id: string;
  kind: InFlightKind;
  title: string;
  detail: string | null;
  hypothesisId: string | null;
  shortId: string | null;
  source: string;
  startedAt: string;
  /** the event that opened this task (for the detail view) */
  event: PropabEvent;
}

export interface RoundView {
  key: string;
  /** 0 is the pre-round "Setup" bucket; real rounds are 1-based. */
  number: number;
  isSetup: boolean;
  status: "running" | "done";
  startedAt: string | null;
  endedAt: string | null;
  hypothesesGenerated: number;
  confirmed: number;
  refuted: number;
  inconclusive: number;
  marginalReturn: number | null;
  /** workers (hypotheses) attributed to this round */
  workers: Worker[];
  events: PropabEvent[];
  llmCalls: number;
  toolCalls: number;
  codeRuns: number;
  errors: number;
}

export type NarrativeItem =
  | { kind: "round"; round: RoundView }
  | { kind: "milestone"; event: PropabEvent };

export interface CampaignModel {
  rounds: RoundView[];
  workers: Worker[];
  inFlight: InFlightTask[];
  narrative: NarrativeItem[];
  counts: {
    llm: number;
    tool: number;
    code: number;
    errors: number;
    workersRunning: number;
    workersDone: number;
  };
}

// Lifecycle events prominent enough to break out of a round group and show as
// their own row in the center narrative.
const MILESTONE_STANDALONE = new Set<string>([
  "campaign.started",
  "campaign.baseline_measured",
  "campaign.breakthrough",
  "synthesis.breakthrough",
  "finding.certified",
  "campaign.budget_exhausted",
  "campaign.completed",
  "session.completed",
  "session.failed",
  "paper.ready",
]);

function shortId(id: string | null | undefined): string {
  if (!id) return "";
  return id.length > 8 ? id.slice(0, 8) : id;
}

function isErr(t: string): boolean {
  return t.endsWith(".error") || t.endsWith(".timeout") || t.endsWith(".failed");
}

// Authoritative round from an event payload — an integer only. Mirrors the
// backend `_round_of` contract: a bool is not a round, and a stringified number
// is rejected (we only trust the first-class int the worker events now carry).
function payloadRound(payload: unknown): number | null {
  if (!payload || typeof payload !== "object") return null;
  const v = (payload as Record<string, unknown>).round;
  if (typeof v === "number" && Number.isInteger(v) && !Number.isNaN(v)) return v;
  return null;
}

// ── Workers ──────────────────────────────────────────────────────────────────
// A worker is a sub-agent running one hypothesis. We group every hypothesis-
// scoped event by hypothesis_id and fold it into a lifecycle record.

function buildWorkers(events: PropabEvent[]): Map<string, Worker> {
  const byHyp = new Map<string, Worker>();

  const ensure = (hid: string): Worker => {
    let w = byHyp.get(hid);
    if (!w) {
      w = {
        hypothesisId: hid,
        shortId: shortId(hid),
        text: "",
        status: "running",
        verdict: null,
        confidence: null,
        startedAt: null,
        endedAt: null,
        lastActivityAt: null,
        steps: [],
        currentCode: null,
        llmCalls: 0,
        toolCalls: 0,
        codeRuns: 0,
        errors: 0,
        events: [],
        round: null,
      };
      byHyp.set(hid, w);
    }
    return w;
  };

  for (const e of events) {
    const hid = e.hypothesis_id;
    if (!hid) continue;
    const w = ensure(hid);
    w.events.push(e);
    w.lastActivityAt = e.timestamp;
    const p = e.payload || {};
    const t = e.event_type;

    if (typeof p.text === "string" && p.text && !w.text) w.text = p.text;
    if (isErr(t)) w.errors += 1;

    // Authoritative round: worker agent.* / agent.progress events now carry a
    // first-class `round`. Trust the first one seen (a worker belongs to one
    // round); buildRounds respects this over positional bucketing.
    if (w.round == null) {
      const r = payloadRound(p);
      if (r != null) w.round = r;
    }

    switch (t) {
      case "agent.started":
        w.startedAt = e.timestamp;
        w.status = "running";
        break;
      case "agent.completed": {
        w.endedAt = e.timestamp;
        const v = (p.verdict as Verdict | undefined) ?? null;
        w.verdict = v;
        w.confidence = p.confidence != null ? Number(p.confidence) : w.confidence;
        w.status =
          v === "confirmed" ? "confirmed" : v === "refuted" ? "refuted" : "inconclusive";
        break;
      }
      case "agent.failed":
      case "agent.time_budget_exceeded":
        w.endedAt = e.timestamp;
        w.status = "failed";
        break;
      case "agent.step_started": {
        const kind =
          p.action === "stop"
            ? "stop"
            : p.tool || p.action === "tool"
              ? "tool"
              : p.action === "code"
                ? "code"
                : "think";
        w.steps.push({
          index: w.steps.length,
          event: e,
          kind,
          label: stepLabel(e),
          detail: typeof p.reasoning === "string" ? p.reasoning : undefined,
        });
        break;
      }
      case "tool.called":
        w.toolCalls += 1;
        break;
      case "code.generated":
        if (typeof p.code === "string") w.currentCode = p.code;
        break;
      case "code.result":
        w.codeRuns += 1;
        break;
      case "llm.prompt":
        w.llmCalls += 1;
        break;
    }
  }

  return byHyp;
}

function stepLabel(e: PropabEvent): string {
  const p = e.payload || {};
  if (p.action === "stop") return "Decided to stop";
  if (p.tool) return `Tool · ${p.tool}`;
  if (p.action === "tool") return "Tool call";
  if (p.action === "code") return "Write & run code";
  if (typeof p.expected_outcome === "string") return p.expected_outcome;
  return "Reasoning step";
}

// ── In-flight tasks ──────────────────────────────────────────────────────────
// The "what's running now" set. We pair each open event with its completion and
// keep the leftovers. Precise keys (step / hypothesis) avoid mismatches; closing
// an unknown key is a harmless no-op (handles trimmed history).

function buildInFlight(events: PropabEvent[], workers: Map<string, Worker>): InFlightTask[] {
  const openCode = new Map<string, PropabEvent>();
  const openTool = new Map<string, PropabEvent>();
  // Prompts pair with their response by the authoritative `call_id` when present
  // (exact, no ordering assumptions); otherwise fall back to FIFO per
  // (source|hypothesis) context for older streams that don't carry one.
  const openLlmById = new Map<string, PropabEvent>();
  const openLlm = new Map<string, PropabEvent[]>();
  const runningWorkers = new Set<string>();

  const llmKey = (e: PropabEvent) => `${e.source}|${e.hypothesis_id ?? "root"}`;
  const callId = (e: PropabEvent): string | null => {
    const v = e.payload?.call_id;
    return typeof v === "string" && v ? v : null;
  };

  for (const e of events) {
    const t = e.event_type;
    const hid = e.hypothesis_id;
    switch (t) {
      case "agent.started":
        if (hid) runningWorkers.add(hid);
        break;
      case "agent.completed":
      case "agent.failed":
      case "agent.time_budget_exceeded":
        if (hid) runningWorkers.delete(hid);
        break;
      case "code.generated":
      case "code.submitted":
        openCode.set(e.step, e);
        break;
      case "code.result":
      case "code.error":
      case "code.timeout":
        openCode.delete(e.step);
        break;
      case "tool.called":
        openTool.set(e.step + "|" + (e.payload?.tool ?? ""), e);
        break;
      case "tool.result":
      case "tool.error":
        openTool.delete(e.step + "|" + (e.payload?.tool ?? ""));
        break;
      case "llm.prompt": {
        const cid = callId(e);
        if (cid) {
          openLlmById.set(cid, e);
        } else {
          const k = llmKey(e);
          (openLlm.get(k) ?? openLlm.set(k, []).get(k)!).push(e);
        }
        break;
      }
      case "llm.response":
      case "llm.parse_error": {
        const cid = callId(e);
        if (cid) {
          openLlmById.delete(cid);
        } else {
          const k = llmKey(e);
          openLlm.get(k)?.shift();
        }
        break;
      }
    }
  }

  const tasks: InFlightTask[] = [];

  for (const hid of runningWorkers) {
    const w = workers.get(hid);
    const startEvt = w?.events.find((x) => x.event_type === "agent.started") ?? w?.events[0];
    if (!startEvt) continue;
    tasks.push({
      id: `exp:${hid}`,
      kind: "experiment",
      title: w?.text ? truncate(w.text, 60) : `Experiment ${shortId(hid)}`,
      detail: w ? `${w.steps.length} steps · ${w.toolCalls} tools · ${w.codeRuns} runs` : null,
      hypothesisId: hid,
      shortId: shortId(hid),
      source: "worker",
      startedAt: w?.startedAt ?? w?.lastActivityAt ?? startEvt.timestamp,
      event: startEvt,
    });
  }

  for (const e of openCode.values()) {
    tasks.push({
      id: `code:${e.event_id}`,
      kind: "code",
      title: "Running code in sandbox",
      detail: firstCodeLine(e.payload?.code),
      hypothesisId: e.hypothesis_id,
      shortId: e.hypothesis_id ? shortId(e.hypothesis_id) : null,
      source: e.source,
      startedAt: e.timestamp,
      event: e,
    });
  }

  for (const e of openTool.values()) {
    tasks.push({
      id: `tool:${e.event_id}`,
      kind: "tool",
      title: `Tool · ${e.payload?.tool ?? "call"}`,
      detail: e.hypothesis_id ? `hypothesis ${shortId(e.hypothesis_id)}` : null,
      hypothesisId: e.hypothesis_id,
      shortId: e.hypothesis_id ? shortId(e.hypothesis_id) : null,
      source: e.source,
      startedAt: e.timestamp,
      event: e,
    });
  }

  const pushLlm = (e: PropabEvent) => {
    const purpose = e.payload?.purpose ?? "call";
    tasks.push({
      id: `llm:${e.event_id}`,
      kind: "llm",
      title: `LLM · ${String(purpose).replace(/_/g, " ")}`,
      detail: e.payload?.model ? String(e.payload.model) : null,
      hypothesisId: e.hypothesis_id,
      shortId: e.hypothesis_id ? shortId(e.hypothesis_id) : null,
      source: e.source,
      startedAt: e.timestamp,
      event: e,
    });
  };
  for (const e of openLlmById.values()) pushLlm(e);
  for (const arr of openLlm.values()) for (const e of arr) pushLlm(e);

  // Most recently started first.
  tasks.sort((a, b) => (b.startedAt || "").localeCompare(a.startedAt || ""));
  return tasks;
}

function firstCodeLine(code: unknown): string | null {
  if (typeof code !== "string") return null;
  const line = code.split("\n").find((l) => l.trim() && !l.trim().startsWith("#"));
  return line ? truncate(line.trim(), 56) : null;
}

// ── Rounds + narrative ───────────────────────────────────────────────────────
// Segment the ordered stream into a Setup bucket and per-round buckets by the
// round.started / round.completed boundaries (positional — so worker/LLM events
// that occur during a round are captured even though they don't carry a round
// number). Lifecycle milestones break out as standalone narrative rows.

function buildRounds(events: PropabEvent[], workers: Map<string, Worker>): {
  rounds: RoundView[];
  narrative: NarrativeItem[];
} {
  const narrative: NarrativeItem[] = [];
  const rounds: RoundView[] = [];

  let current: RoundView = newRound(0, true);
  let hasSetupContent = false;

  const pushCurrent = () => {
    if (current.isSetup && !hasSetupContent) return;
    rounds.push(current);
    narrative.push({ kind: "round", round: current });
  };

  for (const e of events) {
    const t = e.event_type;

    if (t === "round.started") {
      pushCurrent();
      const num = Number(e.payload?.round ?? rounds.length) || rounds.length + 1;
      current = newRound(num, false);
      current.startedAt = e.timestamp;
      current.events.push(e);
      continue;
    }

    if (MILESTONE_STANDALONE.has(t)) {
      narrative.push({ kind: "milestone", event: e });
      // campaign.started still counts as setup context but is shown standalone.
      continue;
    }

    current.events.push(e);
    if (current.isSetup) hasSetupContent = true;
    tallyRound(current, e);

    if (t === "round.completed") {
      current.status = "done";
      current.endedAt = e.timestamp;
      const p = e.payload || {};
      current.confirmed = Number(p.confirmed ?? current.confirmed) || 0;
      current.refuted = Number(p.refuted ?? current.refuted) || 0;
      current.inconclusive = Number(p.inconclusive ?? current.inconclusive) || 0;
      current.marginalReturn = p.marginal_return != null ? Number(p.marginal_return) : null;
      pushCurrent();
      current = newRound(current.number, false);
      current.status = "done"; // interlude bucket, only surfaces if it gets content
      hasSetupContent = false;
      // reset interlude as a fresh setup-like bucket that won't show unless filled
      current.isSetup = true;
      continue;
    }
  }
  pushCurrent();

  // Attribute workers to rounds. Prefer the authoritative `round` the worker's
  // events now carry (set in buildWorkers); fall back to positional membership
  // when it's absent (older streams / trimmed history).
  const roundOfEvent = new Map<string, number>();
  rounds.forEach((r, i) => r.events.forEach((e) => roundOfEvent.set(e.event_id, i)));
  const realByNumber = new Map<number, number>();
  rounds.forEach((r, i) => {
    if (!r.isSetup) realByNumber.set(r.number, i);
  });
  for (const w of workers.values()) {
    let idx: number | undefined;
    if (w.round != null && realByNumber.has(w.round)) {
      idx = realByNumber.get(w.round);
    } else {
      idx = w.events.map((e) => roundOfEvent.get(e.event_id)).find((x) => x != null);
    }
    if (idx != null) {
      // Only overwrite the display round when it wasn't authoritatively set.
      if (w.round == null) w.round = rounds[idx].isSetup ? null : rounds[idx].number;
      rounds[idx].workers.push(w);
    }
  }

  // If a round never emitted an explicit tally, derive it from its workers.
  for (const r of rounds) {
    if (r.confirmed + r.refuted + r.inconclusive === 0 && r.workers.length) {
      for (const w of r.workers) {
        if (w.status === "confirmed") r.confirmed += 1;
        else if (w.status === "refuted") r.refuted += 1;
        else if (w.status === "inconclusive") r.inconclusive += 1;
      }
    }
    if (!r.hypothesesGenerated && r.workers.length) r.hypothesesGenerated = r.workers.length;
  }

  return { rounds, narrative };
}

function newRound(number: number, isSetup: boolean): RoundView {
  return {
    // Stable across rebuilds so React preserves each card's collapsed state.
    key: isSetup ? `setup-${number}` : `round-${number}`,
    number,
    isSetup,
    status: "running",
    startedAt: null,
    endedAt: null,
    hypothesesGenerated: 0,
    confirmed: 0,
    refuted: 0,
    inconclusive: 0,
    marginalReturn: null,
    workers: [],
    events: [],
    llmCalls: 0,
    toolCalls: 0,
    codeRuns: 0,
    errors: 0,
  };
}

function tallyRound(r: RoundView, e: PropabEvent) {
  const t = e.event_type;
  const p = e.payload || {};
  if (t === "llm.prompt") r.llmCalls += 1;
  else if (t === "tool.called") r.toolCalls += 1;
  else if (t === "code.result") r.codeRuns += 1;
  if (isErr(t)) r.errors += 1;
  if (t === "hypothesis.generated") {
    const n = Array.isArray(p.hypotheses) ? p.hypotheses.length : Number(p.count ?? 0);
    if (n) r.hypothesesGenerated += n;
  }
  if (t === "hypothesis.ranked" && Array.isArray(p.hypotheses)) {
    r.hypothesesGenerated = Math.max(r.hypothesesGenerated, p.hypotheses.length);
  }
}

function truncate(s: string, n: number): string {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

// ── Public entry point ───────────────────────────────────────────────────────

export function buildCampaignModel(events: PropabEvent[]): CampaignModel {
  const workerMap = buildWorkers(events);
  const inFlight = buildInFlight(events, workerMap);
  const { rounds, narrative } = buildRounds(events, workerMap);

  const workers = Array.from(workerMap.values());
  // Running first, then most-recent activity.
  workers.sort((a, b) => {
    const ar = a.status === "running" ? 0 : 1;
    const br = b.status === "running" ? 0 : 1;
    if (ar !== br) return ar - br;
    return (b.lastActivityAt || "").localeCompare(a.lastActivityAt || "");
  });

  let llm = 0,
    tool = 0,
    code = 0,
    errors = 0;
  for (const e of events) {
    const t = e.event_type;
    if (t === "llm.prompt") llm += 1;
    else if (t === "tool.called") tool += 1;
    else if (t === "code.result") code += 1;
    if (isErr(t)) errors += 1;
  }

  const workersRunning = workers.filter((w) => w.status === "running").length;

  return {
    rounds,
    workers,
    inFlight,
    narrative,
    counts: {
      llm,
      tool,
      code,
      errors,
      workersRunning,
      workersDone: workers.length - workersRunning,
    },
  };
}

// ── Discovery derivations (§A meter · §B hero/cards) ─────────────────────────
// Small PURE helpers layered on top of the event-derived model. They read the
// campaign summary/snapshot (not the event stream) and never reshape any of the
// exports above — added so the HUD's distance-to-breakthrough meter and the
// Discovery Hero can render from what `types.ts` actually provides.
//
// NB: until the backend emits first-class discovery events (design.md §3 item 6
// — `finding.best_updated` / candidate-record / certification carrying the
// witness + certification booleans), the witness/certification below are read
// defensively from whatever `campaign.best_finding` happens to carry, and are
// simply absent when it carries nothing. Nothing here invents a field.

function numOrNull(v: unknown): number | null {
  if (typeof v === "number") return isFinite(v) ? v : null;
  if (typeof v === "string" && v.trim() !== "" && isFinite(Number(v))) return Number(v);
  return null;
}

function firstNum(...vals: unknown[]): number | null {
  for (const v of vals) {
    const n = numOrNull(v);
    if (n != null) return n;
  }
  return null;
}

function firstStr(...vals: unknown[]): string | null {
  for (const v of vals) {
    if (typeof v === "string" && v.trim() !== "") return v.trim();
  }
  return null;
}

function boolRecord(v: unknown): Record<string, boolean> | null {
  if (!v || typeof v !== "object" || Array.isArray(v)) return null;
  const entries = Object.entries(v as Record<string, unknown>).filter(
    ([, x]) => typeof x === "boolean",
  ) as [string, boolean][];
  return entries.length ? Object.fromEntries(entries) : null;
}

export interface BreakthroughMeter {
  /** true when the summary carries a usable (nonzero) baseline+best metric pair. */
  hasMetric: boolean;
  baseline: number;
  best: number;
  /** signed % improvement of best over baseline (positive = better), e.g. 11.3. */
  improvementPct: number | null;
  /** % improvement required to declare a breakthrough, e.g. 15. */
  thresholdPct: number;
  /** best's position on the baseline→threshold track, in [0,1]. */
  progress: number;
  /** best_metric has reached/exceeded the breakthrough threshold. */
  crossed: boolean;
}

type MeterInput =
  | Pick<
      CampaignSummary,
      "baseline_metric" | "best_metric" | "improvement_pct" | "breakthrough_threshold_pct"
    >
  | null
  | undefined;

// The distance-to-breakthrough meter: where `best_metric` sits on the track from
// baseline (0) to the breakthrough threshold (1). Direction-agnostic — it keys off
// `improvement_pct` (already signed so positive = better) vs the threshold %.
export function breakthroughMeter(summary: MeterInput): BreakthroughMeter {
  const baseline = firstNum(summary?.baseline_metric) ?? 0;
  const best = firstNum(summary?.best_metric) ?? 0;
  const improvementPct = summary?.improvement_pct ?? null;
  const thresholdPct = firstNum(summary?.breakthrough_threshold_pct) ?? 0;
  const hasMetric = Math.abs(baseline) > 1e-12 && Math.abs(best) > 1e-12;
  let progress = 0;
  if (improvementPct != null && thresholdPct > 0) {
    progress = Math.max(0, Math.min(1, improvementPct / thresholdPct));
  }
  const crossed = improvementPct != null && thresholdPct > 0 && improvementPct >= thresholdPct;
  return { hasMetric, baseline, best, improvementPct, thresholdPct, progress, crossed };
}

export interface DiscoverySummary {
  /** best_finding carries real content (vs. an early / metric-only campaign). */
  hasFinding: boolean;
  /** one-line human statement of the best finding, if any. */
  statement: string | null;
  metricName: string | null;
  /** best-so-far scalar (finding metric_value, else summary.best_metric). */
  best: number | null;
  /** the value to beat — published best-known / prior record, if derivable. */
  bestKnown: number | null;
  /** next integer to reach when not yet beaten (integer record-chase), else null. */
  need: number | null;
  direction: "higher_is_better" | "lower_is_better";
  /** best strictly beats best-known. */
  beatsBestKnown: boolean;
  /** witness passed every certification check (null = no certification data). */
  certified: boolean | null;
  /** the raw witness the finding carries (set / certificate), if any. */
  witness: unknown | null;
  /** certification check booleans, if the finding carries them. */
  checks: Record<string, boolean> | null;
  note: string | null;
  meter: BreakthroughMeter;
}

// The latest first-class discovery events (design.md §3 item 6). `finding.*`
// events carry authoritative witness/certification/metric data, so — when the
// stream includes them — the Hero renders from real data instead of inferring
// from the `best_finding` snapshot. Returns the freshest of each, or null.
interface FindingOverlay {
  best: number | null;
  bestKnown: number | null;
  metricName: string | null;
  direction: "higher_is_better" | "lower_is_better" | null;
  witness: unknown | null;
  checks: Record<string, boolean> | null;
  certified: boolean | null;
  beatsBestKnown: boolean | null;
}

function findingOverlay(events: PropabEvent[] | null | undefined): FindingOverlay | null {
  if (!events || !events.length) return null;
  let bestUpdated: Record<string, unknown> | null = null;
  let certified: Record<string, unknown> | null = null;
  // Events arrive in chronological order; the LAST of each type is freshest.
  for (const e of events) {
    if (e.event_type === "finding.best_updated") bestUpdated = e.payload || {};
    else if (e.event_type === "finding.certified") certified = e.payload || {};
  }
  if (!bestUpdated && !certified) return null;

  const dirRaw = firstStr(bestUpdated?.direction, certified?.direction);
  const vs = firstStr(certified?.vs_best_known);
  const checks = boolRecord(certified?.checks) ?? boolRecord((bestUpdated?.certification as Record<string, unknown> | undefined)?.checks);
  let cert: boolean | null = null;
  if (typeof certified?.certified === "boolean") cert = certified.certified as boolean;
  else if (checks) cert = Object.values(checks).every(Boolean);

  return {
    best: firstNum(bestUpdated?.best_metric, bestUpdated?.metric_value, certified?.metric_value),
    bestKnown: firstNum(certified?.best_known_size, bestUpdated?.best_known, certified?.best_known),
    metricName: firstStr(certified?.metric_name, bestUpdated?.metric_name),
    direction: dirRaw === "lower_is_better" ? "lower_is_better" : dirRaw === "higher_is_better" ? "higher_is_better" : null,
    witness: certified?.witness ?? bestUpdated?.witness ?? null,
    checks,
    certified: cert,
    beatsBestKnown: vs != null ? vs === "exceeds_best_known" : null,
  };
}

// Derive the discovery state the Hero + breakthrough card render from. Prefers
// the authoritative `finding.best_updated` / `finding.certified` events when the
// stream carries them (§3 item 6), else reads `campaign.best_finding` defensively
// (domain-general, not yet standardized), else falls back to the summary metrics
// for the honest "no result yet" state.
export function discoverySummary(
  campaign: CampaignState["campaign"] | null | undefined,
  summary: MeterInput,
  events?: PropabEvent[] | null,
): DiscoverySummary {
  const f = (campaign?.best_finding ?? null) as Record<string, unknown> | null;
  const crit = (campaign?.breakthrough_criteria ?? {}) as Record<string, unknown>;
  const ov = findingOverlay(events);
  const direction =
    ov?.direction ?? (crit.direction === "lower_is_better" ? "lower_is_better" : "higher_is_better");
  const meter = breakthroughMeter(summary);

  const metricName = firstStr(ov?.metricName, crit.metric_name, f?.metric_name);
  const statement = firstStr(f?.statement, f?.description, f?.summary, f?.key_finding);
  const note = firstStr(f?.note);
  const best = firstNum(
    ov?.best,
    f?.metric_value,
    metricName ? f?.[metricName] : undefined,
    f?.size,
    summary?.best_metric,
    campaign?.best_metric,
  );
  const bestKnown = firstNum(
    ov?.bestKnown,
    f?.best_known,
    f?.published_best,
    f?.record,
    f?.prior_best,
  );
  const witness = ov?.witness ?? f?.witness ?? f?.set ?? f?.certificate ?? null;
  const checks = ov?.checks ?? boolRecord(f?.checks);

  let certified: boolean | null = ov?.certified ?? null;
  if (certified == null) {
    if (typeof f?.certified === "boolean") certified = f.certified as boolean;
    else if (checks) certified = Object.values(checks).every(Boolean);
  }

  // The certified event's `vs_best_known` is authoritative when present; else
  // compare the derived metric against best-known in the objective direction.
  const beatsBestKnown =
    ov?.beatsBestKnown != null
      ? ov.beatsBestKnown
      : best != null && bestKnown != null
        ? direction === "higher_is_better"
          ? best > bestKnown
          : best < bestKnown
        : false;

  // Record-chase "need": for an integer, higher-is-better target that isn't yet
  // beaten, the next value to reach is best-known + 1 (e.g. "found 16 · need 17").
  let need: number | null = null;
  if (
    best != null &&
    bestKnown != null &&
    Number.isInteger(bestKnown) &&
    direction === "higher_is_better" &&
    !beatsBestKnown
  ) {
    need = bestKnown + 1;
  }

  const hasFinding =
    ov != null ||
    (!!f && Object.keys(f).length > 0 && (statement != null || witness != null || firstNum(f?.metric_value, f?.size) != null));

  return {
    hasFinding,
    statement,
    metricName,
    best,
    bestKnown,
    need,
    direction,
    beatsBestKnown,
    certified,
    witness,
    checks,
    note,
    meter,
  };
}

// ── Center-narrative derivations (design.md §C round cards) ───────────────────
// Additive helpers that layer richer per-round texture onto the existing
// RoundView without reshaping it: a proportional verdict split, an activity
// sparkline, a per-round best-metric delta, and the orchestrator's phase voice.
// All pure, all defensive — each degrades to null/empty when the data is absent.

// A metric key heuristic (domain-general): the backend does not emit an
// authoritative per-round best metric, so we sniff it from event payloads —
// preferring the campaign's own metric name, then standard keys, then a coarse
// pattern (…_r2, score, auc, accuracy, rmse, mae, f1, mean_…). Booleans and
// `confidence` are ignored so a verdict payload never masquerades as a metric.
const METRIC_KEY_RE = /(^metric$|metric_value|best_metric|_r2$|^r2$|score|auc|accuracy|^acc$|rmse|mae|^f1$|^mean_)/i;

function extractMetric(payload: unknown, metricName: string | null, depth = 0): number | null {
  if (!payload || typeof payload !== "object" || Array.isArray(payload) || depth > 2) return null;
  const p = payload as Record<string, unknown>;
  if (metricName) {
    const direct = numOrNull(p[metricName]);
    if (direct != null) return direct;
    const mean = numOrNull(p["mean_" + metricName]);
    if (mean != null) return mean;
  }
  for (const k of ["metric_value", "best_metric", "metric"]) {
    const v = numOrNull(p[k]);
    if (v != null) return v;
  }
  const nested = extractMetric(p.stdout_json, metricName, depth + 1);
  if (nested != null) return nested;
  for (const [k, v] of Object.entries(p)) {
    if (k === "confidence" || typeof v === "boolean") continue;
    if (METRIC_KEY_RE.test(k)) {
      const n = numOrNull(v);
      if (n != null) return n;
    }
  }
  return null;
}

// Activity sparkline: bucket a round's events across their own time span into
// `buckets` slots, counting events per slot. Uses event timestamps only (never
// `now`) so completed rounds render a stable series and the live round grows.
export function activitySpark(events: PropabEvent[], buckets = 28): number[] {
  const out = new Array<number>(buckets).fill(0);
  if (events.length < 2) {
    if (events.length === 1) out[buckets - 1] = 1;
    return out;
  }
  let lo = Infinity;
  let hi = -Infinity;
  for (const e of events) {
    const t = new Date(e.timestamp).getTime();
    if (!isFinite(t)) continue;
    if (t < lo) lo = t;
    if (t > hi) hi = t;
  }
  if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return out;
  const span = hi - lo;
  for (const e of events) {
    const t = new Date(e.timestamp).getTime();
    if (!isFinite(t)) continue;
    let idx = Math.floor(((t - lo) / span) * buckets);
    if (idx >= buckets) idx = buckets - 1;
    if (idx < 0) idx = 0;
    out[idx] += 1;
  }
  return out;
}

// The orchestrator's voice within a round: the distinct phase-narration strings
// (`campaign.phase` detail lines) it emitted while the round ran, in order.
export function roundPhaseNotes(round: RoundView): string[] {
  const out: string[] = [];
  for (const e of round.events) {
    if (e.event_type === "campaign.phase" || e.step === "campaign.phase") {
      const p = e.payload || {};
      const d =
        typeof p.detail === "string" && p.detail.trim()
          ? p.detail.trim()
          : typeof p.phase === "string"
            ? `Phase: ${String(p.phase).replace(/_/g, " ")}`
            : null;
      if (d && !out.includes(d)) out.push(d);
    }
  }
  return out;
}

export interface RoundStat {
  /** best metric attained by the end of this round (running max/min), if any. */
  best: number | null;
  /** best metric sampled within just this round, if any. */
  roundBest: number | null;
  /** signed change vs. the prior running best (or baseline for the first). */
  delta: number | null;
  /** per-round activity sparkline. */
  spark: number[];
}

// Per-round stats keyed by `round.key`. Threads a running best across rounds in
// narrative order so each card can show its best-metric delta; the sparkline is
// independent per round. `baseline`/`direction`/`metricName` come from the
// discovery summary. Everything is null when no metric is derivable.
export function roundStats(
  rounds: RoundView[],
  opts: { baseline: number | null; direction: "higher_is_better" | "lower_is_better"; metricName: string | null },
): Map<string, RoundStat> {
  const higher = opts.direction !== "lower_is_better";
  const out = new Map<string, RoundStat>();
  let running: number | null = opts.baseline;
  for (const r of rounds) {
    let roundBest: number | null = null;
    for (const e of r.events) {
      const m = extractMetric(e.payload, opts.metricName);
      if (m == null) continue;
      roundBest = roundBest == null ? m : higher ? Math.max(roundBest, m) : Math.min(roundBest, m);
    }
    let delta: number | null = null;
    let best = running;
    if (roundBest != null) {
      if (running == null) {
        best = roundBest;
      } else {
        const improved = higher ? roundBest > running : roundBest < running;
        best = improved ? roundBest : running;
        delta = roundBest - running;
      }
      running = best;
    }
    out.set(r.key, { best, roundBest, delta, spark: activitySpark(r.events) });
  }
  return out;
}

// ── Worker-card derivations (§D workers) ─────────────────────────────────────
// Pure helpers that read a *single* Worker's own events to surface a card-level
// result line without reshaping the Worker type. Everything is read defensively
// from whatever the terminal `agent.completed` payload carries (domain-general),
// so a math/coding campaign that emits `metric_value`/`best_known`/`witness_size`
// lights up the metric line while other domains simply omit it.

export interface WorkerResult {
  /** name of the objective metric, if the completion payload names one. */
  metricName: string | null;
  /** the worker's achieved metric value, if any. */
  metricValue: number | null;
  /** the value to beat (published/prior best), if the payload carries one. */
  bestKnown: number | null;
  /** worker's value strictly beats best-known (direction-aware). */
  beatsBestKnown: boolean;
  /** higher/lower-is-better; defaults to higher when unstated. */
  direction: "higher_is_better" | "lower_is_better";
  /** size of a combinatorial witness (set / certificate), when present. */
  witnessSize: number | null;
  /** significance gate / verification passed, if the payload reports it. */
  sigGatePassed: boolean | null;
  /** true when there is any metric/witness worth rendering. */
  hasMetric: boolean;
}

const METRIC_KEYS = [
  "metric_value",
  "metric",
  "best_metric",
  "mean_r2",
  "r2",
  "lofo_r2",
  "score",
  "value",
  "objective",
];
const BEST_KNOWN_KEYS = ["best_known", "published_best", "record", "prior_best", "baseline_metric"];

function arrLen(v: unknown): number | null {
  if (Array.isArray(v)) return v.length;
  return null;
}

export function workerResult(w: Worker): WorkerResult {
  const done = [...w.events].reverse().find((e) => e.event_type === "agent.completed");
  const p = (done?.payload ?? {}) as Record<string, unknown>;

  const metricName = firstStr(p.metric_name, p.objective_name);
  let metricValue: number | null = null;
  for (const k of METRIC_KEYS) {
    const n = numOrNull(p[k]);
    if (n != null) {
      metricValue = n;
      break;
    }
  }
  const bestKnown = firstNum(...BEST_KNOWN_KEYS.map((k) => p[k]));
  const direction =
    p.direction === "lower_is_better" || p.objective_direction === "lower_is_better"
      ? "lower_is_better"
      : "higher_is_better";
  const witnessSize =
    firstNum(p.witness_size, p.set_size, p.size) ??
    arrLen(p.witness) ??
    arrLen(p.set) ??
    arrLen(p.certificate);
  let sigGatePassed: boolean | null = null;
  if (typeof p.sig_gate_passed === "boolean") sigGatePassed = p.sig_gate_passed;
  else if (typeof p.verified === "boolean") sigGatePassed = p.verified;
  else if (typeof p.significant === "boolean") sigGatePassed = p.significant;

  const beatsBestKnown =
    metricValue != null && bestKnown != null
      ? direction === "higher_is_better"
        ? metricValue > bestKnown
        : metricValue < bestKnown
      : false;

  return {
    metricName,
    metricValue,
    bestKnown,
    beatsBestKnown,
    direction,
    witnessSize,
    sigGatePassed,
    hasMetric: metricValue != null || witnessSize != null,
  };
}

// ── Compute / cost derivations (§D Compute tab) ──────────────────────────────
// Aggregate LLM/tool/code/error activity with a per-purpose and per-tool
// breakdown, plus the compute-budget burn-down. Latency + tokens + cost are NOT
// derived here: the stream does not yet carry them — see design.md §3 items 1–2
// (`llm.response` needs `tokens_in`/`tokens_out`/`duration_ms` + a `call_id`).
// The Compute panel renders those columns as "—/TODO" until the backend emits.

export interface ComputeBreakdownRow {
  label: string;
  count: number;
}

export interface ComputeStats {
  llm: number;
  tool: number;
  code: number;
  errors: number;
  /** llm.prompt calls grouped by `purpose`, most-frequent first. */
  llmByPurpose: ComputeBreakdownRow[];
  /** tool.called grouped by tool name, most-frequent first. */
  toolByName: ComputeBreakdownRow[];
  /** error/timeout/failed events grouped by event_type. */
  errorsByType: ComputeBreakdownRow[];
  budget: {
    hasBudget: boolean;
    usedSec: number;
    totalSec: number;
    remainingSec: number;
    pct: number;
  };
  /** true once any llm.response carried a duration_ms (design.md §3.1). */
  hasLatency: boolean;
  /** true once any llm.response reported token counts (design.md §3.2). */
  hasTokens: boolean;
  /** per-call latency, derived from llm.response `duration_ms`. */
  latency: {
    /** number of responses that carried a duration. */
    count: number;
    p50Ms: number | null;
    p95Ms: number | null;
    maxMs: number | null;
  };
  /** token totals, derived from llm.response `tokens_in`/`tokens_out`. */
  tokens: {
    /** number of responses that reported token usage. */
    count: number;
    in: number;
    out: number;
    total: number;
  };
}

// Nearest-rank percentile over an unsorted numeric sample (0 <= q <= 1).
function percentile(sorted: number[], q: number): number | null {
  if (!sorted.length) return null;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1));
  return sorted[idx];
}

function topRows(counter: Map<string, number>, limit = 8): ComputeBreakdownRow[] {
  return Array.from(counter.entries())
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, limit);
}

export function computeStats(
  events: PropabEvent[],
  summary?: Pick<CampaignSummary, "elapsed_sec" | "remaining_sec"> | null,
): ComputeStats {
  let llm = 0,
    tool = 0,
    code = 0,
    errors = 0;
  const byPurpose = new Map<string, number>();
  const byTool = new Map<string, number>();
  const byError = new Map<string, number>();
  const durations: number[] = [];
  let tokensIn = 0;
  let tokensOut = 0;
  let tokenCalls = 0;

  for (const e of events) {
    const t = e.event_type;
    const p = e.payload || {};
    if (t === "llm.prompt") {
      llm += 1;
      const purpose = String(p.purpose ?? "other").replace(/_/g, " ");
      byPurpose.set(purpose, (byPurpose.get(purpose) ?? 0) + 1);
    } else if (t === "llm.response") {
      // Cost/latency now ride on the response (design.md §3.1–3.2).
      const d = numOrNull(p.duration_ms);
      if (d != null && d >= 0) durations.push(d);
      const ti = numOrNull(p.tokens_in);
      const to = numOrNull(p.tokens_out);
      if (ti != null || to != null) {
        tokensIn += ti ?? 0;
        tokensOut += to ?? 0;
        tokenCalls += 1;
      }
    } else if (t === "tool.called") {
      tool += 1;
      const name = String(p.tool ?? "call");
      byTool.set(name, (byTool.get(name) ?? 0) + 1);
    } else if (t === "code.result") {
      code += 1;
    }
    if (isErr(t)) {
      errors += 1;
      byError.set(t, (byError.get(t) ?? 0) + 1);
    }
  }

  const usedSec = firstNum(summary?.elapsed_sec) ?? 0;
  const remainingSec = firstNum(summary?.remaining_sec) ?? 0;
  const totalSec = usedSec + remainingSec;
  const hasBudget = totalSec > 0;

  const sortedDur = durations.slice().sort((a, b) => a - b);

  return {
    llm,
    tool,
    code,
    errors,
    llmByPurpose: topRows(byPurpose),
    toolByName: topRows(byTool),
    errorsByType: topRows(byError),
    budget: {
      hasBudget,
      usedSec,
      totalSec,
      remainingSec,
      pct: hasBudget ? Math.max(0, Math.min(1, usedSec / totalSec)) : 0,
    },
    hasLatency: sortedDur.length > 0,
    hasTokens: tokenCalls > 0,
    latency: {
      count: sortedDur.length,
      p50Ms: percentile(sortedDur, 0.5),
      p95Ms: percentile(sortedDur, 0.95),
      maxMs: sortedDur.length ? sortedDur[sortedDur.length - 1] : null,
    },
    tokens: {
      count: tokenCalls,
      in: tokensIn,
      out: tokensOut,
      total: tokensIn + tokensOut,
    },
  };
}

// ── Metric trajectory (§D Metrics tab) ───────────────────────────────────────
// The best-metric-over-rounds series the Metrics chart plots against the
// baseline and the breakthrough threshold. Derived from the round segmentation:
// for each real round we take the best objective value any of its workers
// achieved (direction-aware), carrying the running best forward so the line is
// monotone-improving. Falls back to the summary's best_metric for the final
// point when no per-worker metric is emitted.

export interface MetricPoint {
  round: number;
  /** best value achieved *by the end of* this round (running best). */
  best: number;
  /** this round's own best (may equal running best), for the step marker. */
  roundBest: number | null;
}

export interface MetricTrajectory {
  hasData: boolean;
  baseline: number | null;
  thresholdValue: number | null;
  direction: "higher_is_better" | "lower_is_better";
  points: MetricPoint[];
}

export function metricTrajectory(
  rounds: RoundView[],
  summary?: Pick<
    CampaignSummary,
    "baseline_metric" | "best_metric" | "breakthrough_threshold_pct"
  > | null,
  direction: "higher_is_better" | "lower_is_better" = "higher_is_better",
): MetricTrajectory {
  const baseline = firstNum(summary?.baseline_metric);
  const higher = direction === "higher_is_better";
  const better = (a: number, b: number) => (higher ? a > b : a < b);

  const realRounds = rounds.filter((r) => !r.isSetup);
  const points: MetricPoint[] = [];
  let running: number | null = baseline;

  for (const r of realRounds) {
    let roundBest: number | null = null;
    for (const w of r.workers) {
      const res = workerResult(w);
      if (res.metricValue == null) continue;
      if (roundBest == null || better(res.metricValue, roundBest)) roundBest = res.metricValue;
    }
    if (roundBest != null && (running == null || better(roundBest, running))) running = roundBest;
    if (running != null) points.push({ round: r.number, best: running, roundBest });
  }

  // Ensure the final point reflects the authoritative summary best if higher.
  const summaryBest = firstNum(summary?.best_metric);
  if (summaryBest != null && points.length) {
    const last = points[points.length - 1];
    if (better(summaryBest, last.best)) last.best = summaryBest;
  } else if (summaryBest != null && baseline != null && points.length === 0) {
    points.push({ round: 1, best: summaryBest, roundBest: summaryBest });
  }

  const thresholdPct = firstNum(summary?.breakthrough_threshold_pct);
  const thresholdValue =
    baseline != null && thresholdPct != null
      ? higher
        ? baseline * (1 + thresholdPct / 100)
        : baseline * (1 - thresholdPct / 100)
      : null;

  return {
    hasData: points.length > 0 && baseline != null,
    baseline,
    thresholdValue,
    direction,
    points,
  };
}
