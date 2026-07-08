import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useCampaignLive } from "../hooks/useCampaignLive";
import { useNow } from "../hooks/useNow";
import { useLiveStore } from "../store";
import { statusView } from "../lib/status";
import { breakthroughMeter, buildCampaignModel, discoverySummary } from "../lib/model";
import NarrativeStream from "../components/campaign/NarrativeStream";
import { useFollowEdge, JumpToLivePill } from "../components/campaign/followEdge";
import CampaignHud from "../components/campaign/CampaignHud";
import DiscoveryHero from "../components/campaign/DiscoveryHero";
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

  const model = useMemo(() => buildCampaignModel(events), [events]);
  const meter = useMemo(() => breakthroughMeter(campaign?.summary), [campaign?.summary]);
  const discovery = useMemo(
    () => discoverySummary(c, campaign?.summary, events),
    [c, campaign?.summary, events],
  );
  const active = sv.active && connected;
  const now = useNow(active);

  const paperReady = useMemo(() => events.some((e) => e.event_type === "paper.ready"), [events]);
  const label = active ? liveLabel(events, model.counts.workersRunning) : undefined;

  // Follow-the-live-edge: the center scroll container auto-scrolls while pinned to
  // the bottom; scrolling up reveals a "jump to live (N new)" pill.
  const edge = useFollowEdge(events.length, model.narrative.length);

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
        {/* §A — persistent Campaign HUD (vital signs + distance-to-breakthrough meter) */}
        {campaign ? (
          <CampaignHud
            id={id}
            campaign={campaign}
            sv={sv}
            counts={model.counts}
            meter={meter}
            active={active}
            paperReady={paperReady}
          />
        ) : (
          <div className="shrink-0 border-b border-line px-[26px] pb-3 pt-[14px]">
            <span className="font-mono text-[14px] font-semibold text-ink">
              {id ? id.slice(0, 8) : "campaign"}
            </span>
            <span className="ml-[10px] text-[11.5px] text-ink-3">loading…</span>
          </div>
        )}

        {/* narrative — center column owns the follow-the-live-edge scroll */}
        <div className="relative flex min-h-0 flex-1 flex-col">
          <div
            ref={edge.ref}
            onScroll={edge.onScroll}
            className="pp-scroll min-h-0 flex-1 overflow-y-auto px-[26px] pb-2 pt-5"
          >
            {/* §B — Discovery Hero pinned atop the center column */}
            {campaign && (
              <div className="mx-auto max-w-[720px]">
                <DiscoveryHero discovery={discovery} />
              </div>
            )}
            <NarrativeStream
              narrative={model.narrative}
              start={c?.started_at ?? null}
              now={now}
              live={{ active, label }}
              discovery={discovery}
              campaignId={id}
            />
          </div>
          <JumpToLivePill show={!edge.pinned} newCount={edge.newCount} onClick={edge.jump} />
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
