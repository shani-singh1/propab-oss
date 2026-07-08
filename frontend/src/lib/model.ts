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
  // llm prompts pair FIFO per (source|hypothesis) context
  const openLlm = new Map<string, PropabEvent[]>();
  const runningWorkers = new Set<string>();

  const llmKey = (e: PropabEvent) => `${e.source}|${e.hypothesis_id ?? "root"}`;

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
        const k = llmKey(e);
        (openLlm.get(k) ?? openLlm.set(k, []).get(k)!).push(e);
        break;
      }
      case "llm.response":
      case "llm.parse_error": {
        const k = llmKey(e);
        openLlm.get(k)?.shift();
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

  for (const arr of openLlm.values()) {
    for (const e of arr) {
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
    }
  }

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

  // Attribute workers to rounds by membership of their events.
  const roundOfEvent = new Map<string, number>();
  rounds.forEach((r, i) => r.events.forEach((e) => roundOfEvent.set(e.event_id, i)));
  for (const w of workers.values()) {
    const idx = w.events.map((e) => roundOfEvent.get(e.event_id)).find((x) => x != null);
    if (idx != null) {
      w.round = rounds[idx].isSetup ? null : rounds[idx].number;
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

// Derive the discovery state the Hero + breakthrough card render from. Reads
// `campaign.best_finding` defensively (its shape is domain-general and, per §3,
// not yet standardized) and falls back to the summary metrics for the honest
// "no result yet" state.
export function discoverySummary(
  campaign: CampaignState["campaign"] | null | undefined,
  summary: MeterInput,
): DiscoverySummary {
  const f = (campaign?.best_finding ?? null) as Record<string, unknown> | null;
  const crit = (campaign?.breakthrough_criteria ?? {}) as Record<string, unknown>;
  const direction = crit.direction === "lower_is_better" ? "lower_is_better" : "higher_is_better";
  const meter = breakthroughMeter(summary);

  const metricName = firstStr(crit.metric_name, f?.metric_name);
  const statement = firstStr(f?.statement, f?.description, f?.summary, f?.key_finding);
  const note = firstStr(f?.note);
  const best = firstNum(
    f?.metric_value,
    metricName ? f?.[metricName] : undefined,
    f?.size,
    summary?.best_metric,
    campaign?.best_metric,
  );
  const bestKnown = firstNum(f?.best_known, f?.published_best, f?.record, f?.prior_best);
  const witness = f?.witness ?? f?.set ?? f?.certificate ?? null;
  const checks = boolRecord(f?.checks);

  let certified: boolean | null = null;
  if (typeof f?.certified === "boolean") certified = f.certified as boolean;
  else if (checks) certified = Object.values(checks).every(Boolean);

  const beatsBestKnown =
    best != null && bestKnown != null
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

  const hasFinding = !!f && Object.keys(f).length > 0 && (statement != null || witness != null || firstNum(f?.metric_value, f?.size) != null);

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
