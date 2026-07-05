import type { CampaignListItem, CampaignState, Verdict } from "../types";

// Tone drives the colored status dot. Maps to the reference palette:
//   live  -> ink, pulsing        (a campaign actively running)
//   pos   -> green               (concluded on a breakthrough)
//   neg   -> red                 (stopped, budget spent / failed)
//   idle  -> ink3                (paused, resumable)
//   queued-> ink4                (not started)
export type Tone = "live" | "pos" | "neg" | "idle" | "queued";

export interface StatusView {
  label: string;
  tone: Tone;
  /** true for statuses that still burn compute — the dot pulses. */
  active: boolean;
  /** true when a resume call is a legal next step (composer / resume). */
  resumable: boolean;
}

export function statusView(status: string | null | undefined): StatusView {
  switch (status) {
    case "active":
      return { label: "Running", tone: "live", active: true, resumable: false };
    case "paused":
      return { label: "Paused", tone: "idle", active: false, resumable: true };
    case "breakthrough":
      return { label: "Concluded", tone: "pos", active: false, resumable: true };
    case "budget_exhausted":
      return { label: "Budget spent", tone: "neg", active: false, resumable: true };
    case "failed":
      return { label: "Failed", tone: "neg", active: false, resumable: true };
    case "completed":
      return { label: "Completed", tone: "pos", active: false, resumable: true };
    case "queued":
      return { label: "Queued", tone: "queued", active: false, resumable: false };
    default:
      return { label: status || "Unknown", tone: "idle", active: false, resumable: false };
  }
}

// Tailwind color the tone dot / text should use.
export function toneColor(tone: Tone): string {
  switch (tone) {
    case "pos":
      return "var(--green)";
    case "neg":
      return "var(--red)";
    case "live":
      return "var(--text)";
    case "idle":
      return "var(--text3)";
    default:
      return "var(--text4)";
  }
}

export function verdictColor(v: Verdict | string | undefined | null): string {
  switch (v) {
    case "confirmed":
      return "var(--green)";
    case "refuted":
      return "var(--red)";
    case "pending":
    case "running":
      return "var(--text)";
    case "inconclusive":
      return "var(--text3)";
    default:
      return "var(--text4)";
  }
}

// Progress toward a breakthrough, in [0,1]. Prefers the metric-vs-threshold ratio;
// falls back to the confirmed/total ratio when no metric improvement is available.
export function campaignProgress(
  c: Pick<CampaignListItem, "improvement_pct" | "total_confirmed" | "total_hypotheses"> & {
    breakthrough_threshold_pct?: number;
  },
): number {
  const imp = c.improvement_pct;
  const thr = c.breakthrough_threshold_pct;
  if (imp != null && thr && thr > 0 && imp > 0) {
    return Math.max(0, Math.min(1, imp / thr));
  }
  if (c.total_hypotheses > 0) {
    return Math.max(0, Math.min(1, c.total_confirmed / c.total_hypotheses));
  }
  return 0;
}

// Number of experiments currently in-flight (frontier size), best-effort.
export function agentsInFlight(state: CampaignState | null): number {
  return state?.summary?.tree?.frontier_size ?? 0;
}
