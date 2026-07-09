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

// Event types that represent campaign milestones (the "Milestones" filter and
// the emphasized timeline dots). Everything else is fine-grained activity.
const MILESTONE_TYPES = new Set<string>([
  "campaign.started",
  "campaign.baseline_measured",
  "campaign.breakthrough",
  "campaign.budget_exhausted",
  "campaign.completed",
  "campaign.progress",
  "hypothesis.generated",
  "hypothesis.tree_expanded",
  "hypothesis.tree_frontier_empty",
  "hypothesis.promoted",
  "hypothesis.retired",
  "agent.completed",
  "synthesis.result_received",
  "synthesis.ledger_updated",
  "synthesis.breakthrough",
  "synthesis.dead_end",
  "synthesis.all_inconclusive",
  "finding.best_updated",
  "finding.certified",
  "paper.ready",
  "session.completed",
  "session.failed",
]);

export function isMilestone(t: string): boolean {
  return MILESTONE_TYPES.has(t);
}

// Color of the timeline dot for an event: verdict-driven when present, else a
// coarse tone (errors red, milestones ink, otherwise muted).
export function eventDotColor(e: PropabEvent): string {
  const v = (e.payload?.verdict as Verdict | undefined) || undefined;
  if (v) return verdictColor(v);
  const t = e.event_type;
  if (isErrorEvent(t)) return "var(--red)";
  if (t === "campaign.breakthrough" || t === "synthesis.breakthrough") return "var(--green)";
  if (t === "finding.best_updated") return "var(--green)";
  if (t === "finding.certified")
    return e.payload?.certified === false ? "var(--text)" : "var(--green)";
  if (t === "campaign.budget_exhausted" || t === "synthesis.dead_end" || t === "session.failed")
    return "var(--red)";
  if (t === "synthesis.result_received" || t === "synthesis.ledger_updated") return "var(--green)";
  if (isMilestone(t)) return "var(--text)";
  return "var(--text3)";
}

export function isErrorEvent(t: string): boolean {
  return t.endsWith(".error") || t.endsWith(".timeout") || t.endsWith(".failed");
}

// Short, monospace-friendly hypothesis id for display.
export function shortId(id: string | null | undefined): string {
  if (!id) return "";
  return id.length > 8 ? id.slice(0, 8) : id;
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

export function isOrchestratorEvent(t: string): boolean {
  return t.startsWith("orchestrator.");
}

export interface OrchestratorView {
  kind: OrchestratorKind;
  /** plain-language headline — never a raw enum. */
  label: string;
  /** primary supporting line (hypothesis text / the orchestrator's own detail). */
  detail: string | null;
  /** tertiary line (e.g. the "why" behind a decision). */
  note: string | null;
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
    let label: string;
    switch (verdict) {
      case "confirmed":
        label = "Confirmed a hypothesis";
        break;
      case "refuted":
        label = "Refuted a hypothesis";
        break;
      case "inconclusive":
        label = "Marked a hypothesis inconclusive";
        break;
      default:
        label = "Reviewed a result";
    }
    const meta: string[] = [];
    const mv = num(p.metric_value);
    if (p.metric_name && mv != null) meta.push(`${p.metric_name} ${fmtMetric(mv)}`);
    const np = num(p.null_p);
    if (np != null) meta.push(fmtNullP(np));
    if (p.downgraded) meta.push("adjusted");
    const detail = typeof p.hypothesis_text === "string" ? p.hypothesis_text : null;
    const note =
      (typeof p.why === "string" && p.why) ||
      (typeof p.inconclusive_reason === "string" && p.inconclusive_reason) ||
      null;
    return { kind, label, detail, note, verdict, meta, dotColor: verdictColor(verdict) };
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
      detail: status ? `Evidence so far: ${status.replace(/_/g, " ")}` : null,
      note: null,
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
      detail: typeof p.text === "string" ? p.text : null,
      note: null,
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
    detail: typeof p.detail === "string" ? p.detail : null,
    note: null,
    verdict: null,
    meta: [],
    dotColor: "var(--text)",
  };
}

// A one-line summary of a run of orchestrator entries, for the group header.
export function orchestratorGroupSummary(views: OrchestratorView[]): string {
  const n = { literature: 0, reasoning: 0, hypothesis: 0, decision: 0 };
  for (const v of views) n[v.kind] += 1;
  const parts: string[] = [];
  if (n.literature) parts.push("reviewed the literature");
  if (n.hypothesis) parts.push(`${n.hypothesis} ${n.hypothesis > 1 ? "hypotheses" : "hypothesis"} written`);
  if (n.decision) parts.push(`${n.decision} ${n.decision > 1 ? "results reviewed" : "result reviewed"}`);
  if (n.reasoning && !parts.length) {
    // A pure-reasoning run reads best as its own first headline.
    return views.find((v) => v.kind === "reasoning")?.label ?? "Reasoning";
  }
  return parts.length ? parts.join(" · ") : "Orchestrator activity";
}
