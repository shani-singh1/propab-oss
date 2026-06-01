import { Link } from "react-router-dom";
import type { CampaignState } from "../../types";
import { Badge, ProgressBar, StatusDot, Stat } from "../ui";
import { budgetPct, fmtDuration, fmtMetric, fmtPct } from "../../lib/format";

export function OverviewBar({
  state,
  connected,
  paperReady,
}: {
  state: CampaignState;
  connected: boolean;
  paperReady: boolean;
}) {
  const s = state.summary;
  const pct = budgetPct(state.campaign.compute_seconds_used, state.campaign.compute_budget_seconds);
  const baselineKnown = s.baseline_metric > 1e-9;

  return (
    <div className="border-b border-border bg-surface px-6 py-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <StatusDot status={s.status} />
            <Badge tone={s.status === "active" ? "running" : s.status === "failed" ? "refuted" : "neutral"}>
              {s.status}
            </Badge>
            <span
              className={`text-[11px] ${connected ? "text-confirmed" : "text-text-muted"}`}
              title={connected ? "Live event stream connected" : "Stream disconnected"}
            >
              {connected ? "● live" : "○ offline"}
            </span>
          </div>
          <h1 className="mt-2 text-lg font-semibold leading-snug text-text-primary" title={state.campaign.question}>
            {state.campaign.question}
          </h1>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {paperReady && (
            <Link
              to={`/campaign/${state.campaign_id}/paper`}
              className="rounded-lg bg-brand px-3 py-1.5 text-sm font-medium text-white transition hover:bg-brand/90"
            >
              View paper →
            </Link>
          )}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-8 gap-y-3">
        <Stat label="Tested" value={s.total_hypotheses} />
        <Stat label="Confirmed" value={<span className="text-confirmed">{s.total_confirmed}</span>} />
        <Stat label="Tree depth" value={s.tree.max_depth} sub={`${s.tree.frontier_size} on frontier`} />
        <Stat
          label="Best"
          value={fmtMetric(s.best_metric)}
          sub={baselineKnown ? `vs ${fmtMetric(s.baseline_metric)} baseline` : "no baseline"}
        />
        {s.improvement_pct != null && (
          <Stat
            label="Improvement"
            value={
              <span className={s.improvement_pct > 0 ? "text-confirmed" : "text-text-secondary"}>
                {fmtPct(s.improvement_pct)}
              </span>
            }
            sub={`target +${s.breakthrough_threshold_pct}%`}
          />
        )}
        <div className="min-w-[180px] flex-1">
          <div className="mb-1 flex items-center justify-between text-[11px] text-text-muted">
            <span>Budget</span>
            <span>
              {fmtDuration(s.elapsed_sec)} / {fmtDuration(state.campaign.compute_budget_seconds)}
            </span>
          </div>
          <ProgressBar pct={pct} tone={pct > 90 ? "warning" : "brand"} />
        </div>
      </div>
    </div>
  );
}
