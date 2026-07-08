import { Link } from "react-router-dom";
import type { CampaignState } from "../../types";
import type { BreakthroughMeter, CampaignModel } from "../../lib/model";
import type { StatusView } from "../../lib/status";
import { toneColor } from "../../lib/status";
import { fmtDuration, fmtMetric, fmtPct } from "../../lib/format";
import { Dot } from "../primitives";

// ── Persistent Campaign HUD (design.md §A) ───────────────────────────────────
// A compact, always-visible vital-signs strip that replaces the thin header in
// Campaign.tsx. Left→right: status · question · Best-vs-baseline with a
// distance-to-breakthrough meter · hypotheses tested · confirmed · running
// workers (live) · LLM count · errors · budget burn-down · paper link.
// Everything reads from `campaign.summary` / `campaign.campaign` + `model.counts`
// via the existing tokens; both themes are first-class; motion-reduce respected.

// A monospace "N label" chip used across the vitals cluster.
function Vital({
  n,
  label,
  color,
  live,
}: {
  n: number | string;
  label: string;
  color?: string;
  live?: boolean;
}) {
  return (
    <span className="flex items-center gap-[5px] whitespace-nowrap" style={color ? { color } : undefined}>
      {live && <Dot color={color ?? "var(--text)"} pulse size={6} />}
      <span className="font-semibold text-ink-2" style={color ? { color } : undefined}>
        {n}
      </span>
      <span className="text-ink-3">{label}</span>
    </span>
  );
}

// The centerpiece: a slim track from baseline → breakthrough threshold with a
// marker at best_metric. Turns green + pulses once the threshold is crossed.
function BreakthroughMeterBar({ meter }: { meter: BreakthroughMeter }) {
  const { baseline, best, improvementPct, thresholdPct, progress, crossed, hasMetric } = meter;
  const pos = `${Math.max(0, Math.min(1, progress)) * 100}%`;
  const fill = crossed ? "var(--green)" : "var(--text2)";
  const improved = improvementPct != null && improvementPct > 0;

  return (
    <div className="min-w-[190px] max-w-[280px] flex-1">
      <div className="mb-[5px] flex items-baseline gap-[7px]">
        <span className="font-mono text-[9.5px] font-semibold uppercase tracking-[0.08em] text-ink-3">
          Best vs baseline
        </span>
        {hasMetric ? (
          <span className="font-mono text-[11px] font-semibold text-ink">{fmtMetric(best)}</span>
        ) : (
          <span className="font-mono text-[11px] text-ink-3">measuring…</span>
        )}
        {improvementPct != null && (
          <span
            className="font-mono text-[10.5px] font-semibold"
            style={{ color: improved ? "var(--green)" : "var(--text3)" }}
          >
            {fmtPct(improvementPct)}
          </span>
        )}
        {crossed && (
          <span className="font-mono text-[9px] font-bold uppercase tracking-[0.1em] text-pos animate-ppulse motion-reduce:animate-none">
            breakthrough
          </span>
        )}
      </div>

      {/* track: baseline (left) → threshold (right), marker at best */}
      <div
        className={`relative h-[6px] w-full rounded-full ${crossed ? "animate-ppulse motion-reduce:animate-none" : ""}`}
        style={{ background: crossed ? "var(--greenDim)" : "var(--chip)" }}
        role="meter"
        aria-label="Distance to breakthrough"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress * 100)}
      >
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-500"
          style={{ width: pos, background: fill }}
        />
        {/* threshold tick at the far right edge */}
        <span
          className="absolute inset-y-[-2px] right-0 w-[2px] rounded-full"
          style={{ background: crossed ? "var(--green)" : "var(--text4)" }}
          aria-hidden
        />
        {/* best-metric marker */}
        {hasMetric && (
          <span
            className="absolute top-1/2 h-[11px] w-[11px] -translate-x-1/2 -translate-y-1/2 rounded-full border-2 transition-[left] duration-500"
            style={{
              left: pos,
              background: crossed ? "var(--green)" : "var(--text)",
              borderColor: "var(--centerBg)",
            }}
            aria-hidden
          />
        )}
      </div>

      <div className="mt-[4px] flex items-center justify-between font-mono text-[9px] text-ink-4">
        <span>{hasMetric ? fmtMetric(baseline) : "baseline"}</span>
        <span>{thresholdPct > 0 ? `need +${thresholdPct}%` : "threshold"}</span>
      </div>
    </div>
  );
}

// Budget burn-down: elapsed vs remaining as a thin bar.
function BudgetBurnDown({ elapsed, remaining }: { elapsed: number; remaining: number }) {
  const total = elapsed + remaining;
  const pct = total > 0 ? Math.max(0, Math.min(1, elapsed / total)) : 0;
  const low = remaining > 0 && total > 0 && remaining / total < 0.15;
  return (
    <div className="min-w-[150px] max-w-[220px] flex-1">
      <div className="mb-[5px] flex items-baseline justify-between font-mono text-[9.5px]">
        <span className="font-semibold uppercase tracking-[0.08em] text-ink-3">Budget</span>
        <span className="text-ink-4">
          <span className="text-ink-2">{fmtDuration(elapsed)}</span>
          {remaining > 0 && (
            <>
              {" · "}
              <span style={low ? { color: "var(--red)" } : undefined}>
                {fmtDuration(remaining)} left
              </span>
            </>
          )}
        </span>
      </div>
      <div className="h-[6px] w-full overflow-hidden rounded-full bg-chip">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${pct * 100}%`, background: low ? "var(--red)" : "var(--text3)" }}
        />
      </div>
    </div>
  );
}

export default function CampaignHud({
  id,
  campaign,
  sv,
  counts,
  meter,
  active,
  paperReady,
}: {
  id: string | undefined;
  campaign: CampaignState;
  sv: StatusView;
  counts: CampaignModel["counts"];
  meter: BreakthroughMeter;
  active: boolean;
  paperReady: boolean;
}) {
  const c = campaign.campaign;
  const s = campaign.summary;
  const question = c?.question ?? s?.question ?? "";
  const hyps = s?.total_hypotheses ?? c?.hypothesis_tree?.frontier?.length ?? 0;
  const confirmed = s?.total_confirmed ?? 0;
  const elapsed = s?.elapsed_sec ?? c?.compute_seconds_used ?? 0;
  const remaining =
    s?.remaining_sec ??
    (c?.compute_budget_seconds != null && c?.compute_seconds_used != null
      ? Math.max(0, c.compute_budget_seconds - c.compute_seconds_used)
      : 0);

  return (
    <header className="shrink-0 border-b border-line px-[26px] pb-[13px] pt-[13px]">
      {/* line 1 — identity + vitals cluster */}
      <div className="flex items-center gap-[11px]">
        <Dot color={toneColor(sv.tone)} pulse={sv.active} size={8} />
        <span
          className="min-w-0 max-w-[46vw] truncate font-mono text-[14px] font-semibold leading-none tracking-[-0.01em] text-ink"
          title={question}
        >
          {question || (id ? id.slice(0, 8) : "campaign")}
        </span>
        <span className="shrink-0 text-[11.5px] font-medium leading-none text-ink-2">{sv.label}</span>

        <div className="ml-auto flex flex-wrap items-center justify-end gap-x-[14px] gap-y-[6px] font-mono text-[11px] font-medium leading-none">
          <Vital n={hyps} label="tested" />
          <Vital n={confirmed} label="confirmed" color={confirmed > 0 ? "var(--green)" : undefined} />
          {active && counts.workersRunning > 0 && (
            <Vital n={counts.workersRunning} label="running" color="var(--text)" live />
          )}
          <span className="hidden text-ink-3 sm:inline">
            <span className="font-semibold text-ink-2">{counts.llm}</span> LLM
          </span>
          {counts.errors > 0 && <Vital n={counts.errors} label="errors" color="var(--red)" />}
          {paperReady && id && (
            <Link
              to={`/campaign/${id}/paper`}
              className="text-ink-3 underline decoration-ink-4 underline-offset-2 hover:text-ink"
            >
              paper
            </Link>
          )}
        </div>
      </div>

      {/* line 2 — the discovery meter + budget burn-down */}
      <div className="mt-[11px] flex flex-wrap items-start gap-x-[26px] gap-y-[10px] pl-[19px]">
        <BreakthroughMeterBar meter={meter} />
        <BudgetBurnDown elapsed={elapsed} remaining={remaining} />
      </div>
    </header>
  );
}
