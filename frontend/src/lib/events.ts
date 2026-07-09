import type { PropabEvent, Verdict } from "../types";
import { verdictColor } from "./status";
import { fmtMetric } from "./format";

// Coarse phase for an event type, used to group the stream and color it.
export type Phase =
  | "campaign"
  | "orchestrator"
  | "literature"
  | "hypothesis"
  | "experiment"
  | "tool"
  | "code"
  | "llm"
  | "synthesis"
  | "paper"
  | "other";

export function phaseOf(t: string): Phase {
  if (t.startsWith("orchestrator.")) return "orchestrator";
  if (t.startsWith("campaign.") || t === "session.started" || t === "session.completed")
    return "campaign";
  if (t.startsWith("literature.")) return "literature";
  if (t.startsWith("hypothesis.")) return "hypothesis";
  if (t.startsWith("agent.")) return "experiment";
  if (t.startsWith("tool.")) return "tool";
  if (t.startsWith("code.")) return "code";
  if (t.startsWith("llm.")) return "llm";
  if (t.startsWith("synthesis.")) return "synthesis";
  if (t.startsWith("paper.")) return "paper";
  return "other";
}

export const PHASE_LABEL: Record<Phase, string> = {
  campaign: "Campaign",
  orchestrator: "Orchestrator",
  literature: "Literature",
  hypothesis: "Hypothesis",
  experiment: "Experiment",
  tool: "Tool",
  code: "Code",
  llm: "LLM",
  synthesis: "Synthesis",
  paper: "Paper",
  other: "Event",
};

// Human label for an event row.
export function eventLabel(e: PropabEvent): string {
  const t = e.event_type;
  const p = e.payload || {};
  switch (t) {
    case "campaign.started":
      return "Campaign started";
    case "campaign.baseline_measured":
      return p.note
        ? "Baseline skipped (verification campaign)"
        : `Baseline measured: ${p.baseline_metric ?? "?"}`;
    case "campaign.progress":
      return "Progress checkpoint";
    case "campaign.breakthrough":
      return "Breakthrough reached";
    case "campaign.budget_exhausted":
      return "Budget exhausted";
    case "campaign.completed":
      return "Campaign completed";
    case "hypothesis.generated":
      return `Hypotheses generated${p.count ? ` (${p.count})` : ""}`;
    case "hypothesis.dispatched":
      return "Hypothesis dispatched";
    case "hypothesis.tree_expanded":
      return "Tree expanded";
    case "hypothesis.tree_frontier_empty":
      return "Frontier empty — regenerating";
    case "agent.started":
      return "Agent started";
    case "agent.completed":
      return `Agent completed${p.verdict ? ` — ${p.verdict}` : ""}`;
    case "agent.step_started":
      return "Step started";
    case "agent.step_completed":
      return p.action === "stop" ? "Agent decided to stop" : "Step completed";
    case "tool.called":
      return `Tool: ${p.tool ?? "?"}`;
    case "tool.result":
      return `Tool result: ${p.tool ?? ""}`.trim();
    case "tool.error":
      return `Tool error: ${p.tool ?? ""}`.trim();
    case "code.generated":
      return "Code generated";
    case "code.result":
      return "Code ran";
    case "code.error":
      return "Code error";
    case "code.timeout":
      return "Code timeout";
    case "llm.prompt":
      return "LLM prompt";
    case "llm.response":
      return "LLM response";
    case "finding.best_updated":
      return `New best${p.metric_name ? ` ${p.metric_name}` : ""}${p.best_metric != null ? ` ${p.best_metric}` : ""}`.trim();
    case "finding.certified":
      return p.certified === false ? "Record candidate" : "Certified record";
    case "paper.ready":
      return "Paper ready";
    case "paper.section_completed":
      return `Paper section: ${p.section ?? ""}`.trim();
    case "orchestrator.literature":
    case "orchestrator.reasoning":
    case "orchestrator.hypothesis_written":
    case "orchestrator.decision":
      return orchestratorView(e).label;
    default:
      return t.replace(/[._]/g, " ");
  }
}

// ── Orchestrator activity (the mid-panel lane) ───────────────────────────────
// The backend now narrates its own reasoning via four `orchestrator.*` events
// (redesign §4/§5). The UI renders these as the orchestrator's activity feed —
// ALWAYS in plain language, never the raw enum/`decision` string. These helpers
// turn a raw event into a friendly, color-coded view row.

export type OrchestratorKind = "literature" | "reasoning" | "hypothesis" | "decision";

export function orchestratorKind(t: string): OrchestratorKind | null {
  switch (t) {
    case "orchestrator.literature":
      return "literature";
    case "orchestrator.reasoning":
      return "reasoning";
    case "orchestrator.hypothesis_written":
      return "hypothesis";
    case "orchestrator.decision":
      return "decision";
    default:
      return null;
  }
}

export interface OrchestratorView {
  kind: OrchestratorKind;
  /** plain-language, verb-first headline — never a raw enum. */
  label: string;
  /** a hypothesis statement being written or judged — rendered as a quoted claim. */
  claim: string | null;
  /** the orchestrator's reasoning / the "why" — the prominent supporting line. */
  reason: string | null;
  /** the decided next move (deepen / drop / retune …), as a short phrase. */
  next: string | null;
  /** set only for a decision, drives the row's verdict color. */
  verdict: Verdict | null;
  /** small monospace chips (metric, p-value, counts). */
  meta: string[];
  /** color of the row's status dot. */
  dotColor: string;
}

const num = (v: unknown): number | null =>
  typeof v === "number" && isFinite(v) ? v : null;

// A p-value against the null, formatted plainly.
function fmtNullP(v: number): string {
  if (v < 0.001) return "p < 0.001";
  const s = v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  return `p = ${s}`;
}

// Title-case a plain phrase for the rare unmapped `decision` fallback (the
// backend already authors these as prose, so this only tidies capitalization).
function humanize(s: string): string {
  const t = s.replace(/_/g, " ").trim();
  return t ? t.charAt(0).toUpperCase() + t.slice(1) : "";
}

// The orchestrator's decided next move, as a plain phrase. `action` is a small
// backend enum (deepen / drop / retune / …); we narrate it as what happens next.
function nextMove(action: unknown): string | null {
  const a = typeof action === "string" ? action.toLowerCase().trim() : "";
  if (!a) return null;
  if (a.includes("deepen")) return "Deepening this line";
  if (a.includes("drop") || a.includes("prune") || a.includes("retire")) return "Closing the branch";
  if (a.includes("retune") || a.includes("revise") || a.includes("rerun")) return "Retuning the experiment";
  if (a.includes("lateral") || a.includes("pivot")) return "Trying a lateral angle";
  if (a.includes("finaliz")) return "Finalizing the campaign";
  return humanize(a);
}

// Friendly headline for a reasoning/generation `decision` string.
function reasoningHeadline(decision: string, p: Record<string, any>): string {
  const d = decision.toLowerCase();
  const count = num(p.count);
  if (d.includes("baseline")) return "Measured the baseline";
  if (d.includes("anomaly")) return "Chasing an anomaly";
  if (d.includes("seed")) return "Planning the first hypotheses";
  if (d.includes("synthes") || d.includes("follow") || d.includes("expan"))
    return count != null ? `Generating ${count} follow-up hypotheses` : "Generating follow-up hypotheses";
  if (d.includes("finaliz")) {
    const n = num(p.confirmed_findings);
    return n != null ? `Finalizing — ${n} confirmed` : "Finalizing the campaign";
  }
  return decision ? humanize(decision) : "Thinking it through";
}

export function orchestratorView(e: PropabEvent): OrchestratorView {
  const p = e.payload || {};
  const kind = orchestratorKind(e.event_type) ?? "reasoning";

  if (kind === "decision") {
    const verdict = ((p.effective_verdict || p.verdict) as Verdict) || null;
    const np = num(p.null_p);
    // Verb-first headline, with a short statistical qualifier when a null test
    // is present (domain-general — keyed off the `null_p` the backend emits).
    let label: string;
    switch (verdict) {
      case "confirmed":
        label =
          np != null && np < 0.05
            ? "Confirmed a hypothesis — signal beyond the null"
            : "Confirmed a hypothesis";
        break;
      case "refuted":
        label = np != null ? "Refuted a hypothesis — no signal beyond the null" : "Refuted a hypothesis";
        break;
      case "inconclusive":
        label = np != null
          ? "Marked a hypothesis inconclusive — within the noise band"
          : "Marked a hypothesis inconclusive";
        break;
      default:
        label = "Reviewed a result";
    }
    const meta: string[] = [];
    const mv = num(p.metric_value);
    if (p.metric_name && mv != null) meta.push(`${p.metric_name} ${fmtMetric(mv)}`);
    if (np != null) meta.push(fmtNullP(np));
    if (p.downgraded) meta.push("worker overruled");
    const claim = typeof p.hypothesis_text === "string" ? p.hypothesis_text : null;
    const reason =
      (typeof p.why === "string" && p.why) ||
      (typeof p.inconclusive_reason === "string" && p.inconclusive_reason) ||
      null;
    return { kind, label, claim, reason, next: nextMove(p.action), verdict, meta, dotColor: verdictColor(verdict) };
  }

  if (kind === "literature") {
    const facts = num(p.established_facts) ?? 0;
    const gaps = num(p.open_gaps) ?? 0;
    const contested = num(p.contested_claims) ?? 0;
    const papers = num(p.key_papers) ?? 0;
    const meta: string[] = [];
    if (facts) meta.push(`${facts} facts`);
    if (gaps) meta.push(`${gaps} gaps`);
    if (contested) meta.push(`${contested} contested`);
    if (papers) meta.push(`${papers} papers`);
    const status = typeof p.evidence_status === "string" ? p.evidence_status : null;
    return {
      kind,
      label: "Reviewed the literature",
      claim: null,
      reason: status ? `Evidence so far: ${status.replace(/_/g, " ")}.` : null,
      next: null,
      verdict: null,
      meta,
      dotColor: "var(--text3)",
    };
  }

  if (kind === "hypothesis") {
    const k = String(p.kind || "").toLowerCase();
    const label =
      k === "seed"
        ? "Wrote a seed hypothesis"
        : k === "lateral"
          ? "Wrote a lateral hypothesis"
          : k === "child"
            ? "Wrote a follow-up hypothesis"
            : "Wrote a new hypothesis";
    return {
      kind,
      label,
      claim: typeof p.text === "string" ? p.text : null,
      reason: null,
      next: null,
      verdict: null,
      meta: [],
      dotColor: "var(--text2)",
    };
  }

  // reasoning
  const decision = typeof p.decision === "string" ? p.decision : "";
  return {
    kind,
    label: reasoningHeadline(decision, p),
    claim: null,
    reason: typeof p.detail === "string" ? p.detail : null,
    next: null,
    verdict: null,
    meta: [],
    dotColor: "var(--text)",
  };
}
