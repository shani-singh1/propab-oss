import type { CampaignState } from "../../types";
import { Badge } from "../ui";
import { fmtMetric, fmtPct } from "../../lib/format";

export function FindingsPanel({ state }: { state: CampaignState }) {
  const tree = state.campaign.hypothesis_tree;
  const confirmed = Object.values(tree?.nodes ?? {})
    .filter((n) => n.verdict === "confirmed")
    .sort((a, b) => b.confidence - a.confidence);
  const s = state.summary;
  const improved = s.improvement_pct != null && s.improvement_pct > 0;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <h2 className="text-sm font-semibold">Findings</h2>
        <Badge tone="confirmed">{confirmed.length} confirmed</Badge>
      </div>

      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center gap-4">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-text-muted">Best result</div>
            <div className="text-xl font-bold tabular-nums text-text-primary">
              {fmtMetric(s.best_metric)}
            </div>
          </div>
          {s.baseline_metric > 1e-9 && (
            <div>
              <div className="text-[11px] uppercase tracking-wide text-text-muted">vs baseline</div>
              <div className={`text-base font-semibold ${improved ? "text-confirmed" : "text-text-secondary"}`}>
                {fmtPct(s.improvement_pct)}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {confirmed.length === 0 ? (
          <div className="px-4 py-6 text-sm text-text-muted">
            No confirmed findings yet.
          </div>
        ) : (
          confirmed.map((n) => (
            <div key={n.id} className="border-b border-border/50 px-4 py-2.5">
              <div className="flex items-start gap-2">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-confirmed" />
                <div className="min-w-0 flex-1">
                  <p className="text-sm leading-snug text-text-primary">{n.text}</p>
                  <div className="mt-1 flex items-center gap-2 text-[11px] text-text-muted">
                    <span className="text-confirmed">confirmed</span>
                    <span>confidence {fmtMetric(n.confidence)}</span>
                    <span>gen {n.generation}</span>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
