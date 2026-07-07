import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useCampaignLive } from "../hooks/useCampaignLive";
import { useNow } from "../hooks/useNow";
import { useLiveStore } from "../store";
import { statusView, toneColor } from "../lib/status";
import { buildCampaignModel } from "../lib/model";
import { truncate } from "../lib/format";
import { Dot } from "../components/primitives";
import NarrativeStream from "../components/campaign/NarrativeStream";
import Composer from "../components/campaign/Composer";
import RightPanel from "../components/campaign/RightPanel";
import type { PropabEvent } from "../types";

// A short, human "what is it doing right now" line for the live indicator.
function liveLabel(events: PropabEvent[], workersRunning: number): string | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    const p = e.payload || {};
    if (e.event_type === "campaign.phase" || e.step === "campaign.phase") {
      if (typeof p.detail === "string") return p.detail;
      if (typeof p.phase === "string") return `Phase: ${String(p.phase).replace(/_/g, " ")}`;
    }
    if (e.event_type === "hypothesis.generated") return "Generating hypotheses";
    if (e.event_type === "round.started") return `Opening round ${p.round ?? ""}`.trim();
    if (i < events.length - 40) break; // only scan the recent tail
  }
  if (workersRunning > 0) return `Running ${workersRunning} experiment${workersRunning > 1 ? "s" : ""}`;
  return undefined;
}

export default function Campaign() {
  const { id } = useParams();
  useCampaignLive(id);
  const campaign = useLiveStore((s) => s.campaign);
  const events = useLiveStore((s) => s.events);
  const connected = useLiveStore((s) => s.connected);
  const error = useLiveStore((s) => s.error);

  const [rightOpen, setRightOpen] = useState(true);

  const c = campaign?.campaign;
  const status = campaign?.summary?.status ?? c?.status ?? "active";
  const sv = statusView(status);
  const question = c?.question ?? "";
  const beliefs = c?.belief_state?.active_beliefs ?? [];
  const confirmed = campaign?.summary?.total_confirmed ?? 0;

  const model = useMemo(() => buildCampaignModel(events), [events]);
  const active = sv.active && connected;
  const now = useNow(active);

  const paperReady = useMemo(() => events.some((e) => e.event_type === "paper.ready"), [events]);
  const label = active ? liveLabel(events, model.counts.workersRunning) : undefined;

  if (!campaign && error) {
    return (
      <main className="flex min-w-0 flex-1 items-center justify-center bg-center">
        <div className="max-w-[420px] px-6 text-center">
          <div className="mb-2 text-[14px] font-semibold text-ink">Couldn’t load campaign</div>
          <div className="text-[12.5px] leading-relaxed text-ink-3">{error}</div>
          <Link to="/" className="mt-4 inline-block text-[12.5px] text-ink-2 underline">
            ← Back to campaigns
          </Link>
        </div>
      </main>
    );
  }

  return (
    <>
      <main className="flex min-w-0 flex-1 flex-col bg-center">
        {/* header */}
        <div className="shrink-0 border-b border-line px-[26px] pb-3 pt-[14px]">
          <div className="flex items-center gap-[11px]">
            <Dot color={toneColor(sv.tone)} pulse={sv.active} size={8} />
            <span className="font-mono text-[15px] font-semibold leading-none tracking-[-0.01em] text-ink">
              {truncate(question, 34) || (id ? id.slice(0, 8) : "campaign")}
            </span>
            <span className="text-[11.5px] font-medium leading-none text-ink-2">{sv.label}</span>
            <div className="ml-auto flex items-center gap-[14px] font-mono text-[11px] font-medium leading-none text-ink-3">
              {active && model.counts.workersRunning > 0 && (
                <span className="text-ink-2">{model.counts.workersRunning} running</span>
              )}
              <span>{confirmed} confirmed</span>
              <span className="hidden sm:inline">{model.counts.llm} LLM</span>
              {model.counts.errors > 0 && (
                <span style={{ color: "var(--red)" }}>{model.counts.errors} errors</span>
              )}
              {paperReady && (
                <Link to={`/campaign/${id}/paper`} className="underline hover:text-ink">
                  paper
                </Link>
              )}
            </div>
          </div>
          <div className="mt-[9px] pl-[19px]">
            <span className="line-clamp-2 block text-[12.5px] leading-[1.45] text-ink-2">
              {question}
            </span>
          </div>
        </div>

        {/* narrative */}
        <div className="pp-scroll min-h-0 flex-1 overflow-y-auto px-[26px] pb-2 pt-5">
          <NarrativeStream
            narrative={model.narrative}
            start={c?.started_at ?? null}
            now={now}
            live={{ active, label }}
          />
        </div>

        {/* composer */}
        <div className="shrink-0 px-[26px] pb-4 pt-[10px]">
          {id && <Composer campaignId={id} status={sv} />}
        </div>
      </main>

      <RightPanel
        model={model}
        tree={c?.hypothesis_tree}
        question={question}
        beliefs={beliefs}
        events={events}
        now={now}
        active={active}
        open={rightOpen}
        onToggle={() => setRightOpen((o) => !o)}
      />
    </>
  );
}
