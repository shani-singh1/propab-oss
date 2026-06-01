import type { PropabEvent, Verdict } from "../types";

// Coarse phase for an event type, used to group the stream and color it.
export type Phase =
  | "campaign"
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
    case "paper.ready":
      return "Paper ready";
    case "paper.section_completed":
      return `Paper section: ${p.section ?? ""}`.trim();
    default:
      return t.replace(/[._]/g, " ");
  }
}

export function verdictColor(v: Verdict | string | undefined): string {
  switch (v) {
    case "confirmed":
      return "text-confirmed";
    case "refuted":
      return "text-refuted";
    case "running":
    case "pending":
      return "text-running";
    default:
      return "text-inconclusive";
  }
}

export function verdictBorder(v: Verdict | string | undefined): string {
  switch (v) {
    case "confirmed":
      return "border-l-confirmed";
    case "refuted":
      return "border-l-refuted";
    case "pending":
      return "border-l-running";
    default:
      return "border-l-inconclusive";
  }
}

export function isErrorEvent(t: string): boolean {
  return t.endsWith(".error") || t.endsWith(".timeout") || t.endsWith(".failed");
}

// Short, monospace-friendly hypothesis id for display.
export function shortId(id: string | null | undefined): string {
  if (!id) return "";
  return id.length > 8 ? id.slice(0, 8) : id;
}
