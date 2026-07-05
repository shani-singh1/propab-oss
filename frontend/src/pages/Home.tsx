import { useNavigate } from "react-router-dom";
import { useCampaigns } from "../hooks/useCampaigns";
import { campaignProgress, statusView, toneColor } from "../lib/status";
import { fmtRelative, truncate } from "../lib/format";
import { Bar, Dot } from "../components/primitives";
import type { CampaignListItem } from "../types";

function Card({ c }: { c: CampaignListItem }) {
  const nav = useNavigate();
  const sv = statusView(c.status);
  return (
    <button
      onClick={() => nav(`/campaign/${c.id}`)}
      className="pp-row flex cursor-pointer flex-col gap-[11px] rounded-[10px] border border-edge bg-rail px-4 py-[15px] text-left"
    >
      <div className="flex w-full items-center gap-[9px]">
        <Dot color={toneColor(sv.tone)} pulse={sv.active} size={8} />
        <span className="truncate font-mono text-[13.5px] font-semibold leading-none text-ink">
          {truncate(c.question, 26)}
        </span>
        <span className="ml-auto whitespace-nowrap text-[11px] font-medium text-ink-2">
          {sv.label}
        </span>
      </div>
      <div className="min-h-[56px] text-[12.5px] leading-[1.5] text-ink-2">{c.question}</div>
      <Bar pct={campaignProgress(c)} height={4} />
      <div className="flex w-full items-center gap-2 font-mono text-[10.5px] font-medium leading-none text-ink-3">
        <span>{c.total_confirmed} confirmed</span>
        <span>·</span>
        <span>{c.total_hypotheses} hyp</span>
        <span className="ml-auto">{fmtRelative(c.completed_at ?? c.started_at)}</span>
      </div>
    </button>
  );
}

export default function Home() {
  const { campaigns, loading, error } = useCampaigns();
  const nav = useNavigate();

  return (
    <main className="flex min-w-0 flex-1 flex-col bg-center">
      <div className="flex shrink-0 items-center gap-[10px] border-b border-line px-[26px] py-4">
        <span className="text-[16px] font-semibold leading-none text-ink">Campaigns</span>
        <span className="font-mono text-[12px] font-medium leading-none text-ink-3">
          {campaigns.length} program{campaigns.length === 1 ? "" : "s"}
        </span>
        <button
          onClick={() => nav("/new")}
          className="ml-auto flex items-center gap-[7px] rounded-[7px] bg-ink px-[13px] py-2 text-[12.5px] font-semibold leading-none text-center"
          style={{ color: "var(--centerBg)" }}
        >
          + New campaign
        </button>
      </div>

      <div className="pp-scroll min-h-0 flex-1 overflow-y-auto px-[26px] pb-7 pt-5">
        {error && (
          <div className="max-w-[840px] rounded-[10px] border border-edge bg-rail px-4 py-3 text-[12.5px] text-ink-2">
            Couldn’t load campaigns — {error}
          </div>
        )}
        {!error && !loading && campaigns.length === 0 && (
          <div className="flex max-w-[840px] flex-col items-start gap-3 rounded-[10px] border border-edge bg-rail px-5 py-6">
            <div className="text-[14px] font-semibold text-ink">No campaigns yet</div>
            <div className="text-[12.5px] leading-relaxed text-ink-2">
              Start one by defining a research question — Propab reads the literature, forms rival
              hypotheses, and tests them.
            </div>
            <button
              onClick={() => nav("/new")}
              className="rounded-lg bg-ink px-4 py-2 text-[12.5px] font-semibold"
              style={{ color: "var(--centerBg)" }}
            >
              + New campaign
            </button>
          </div>
        )}
        <div className="grid max-w-[840px] grid-cols-1 gap-3 md:grid-cols-2">
          {campaigns.map((c) => (
            <Card key={c.id} c={c} />
          ))}
        </div>
      </div>
    </main>
  );
}
