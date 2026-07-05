import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useCampaignLive } from "../hooks/useCampaignLive";
import { useLiveStore } from "../store";
import { agentsInFlight, statusView, toneColor } from "../lib/status";
import { truncate } from "../lib/format";
import { Dot } from "../components/primitives";
import Timeline from "../components/campaign/Timeline";
import Composer from "../components/campaign/Composer";
import RightPanel from "../components/campaign/RightPanel";

export default function Campaign() {
  const { id } = useParams();
  useCampaignLive(id);
  const campaign = useLiveStore((s) => s.campaign);
  const events = useLiveStore((s) => s.events);
  const connected = useLiveStore((s) => s.connected);
  const error = useLiveStore((s) => s.error);

  const [filter, setFilter] = useState<"all" | "milestones">("all");
  const [rightOpen, setRightOpen] = useState(true);

  const scrollRef = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);
  useEffect(() => {
    const el = scrollRef.current;
    if (el && atBottomRef.current) el.scrollTop = el.scrollHeight;
  }, [events.length, filter]);

  const c = campaign?.campaign;
  const status = campaign?.summary?.status ?? c?.status ?? "active";
  const sv = statusView(status);
  const question = c?.question ?? "";
  const beliefs = c?.belief_state?.active_beliefs ?? [];
  const confirmed = campaign?.summary?.total_confirmed ?? 0;
  const agents = agentsInFlight(campaign);
  const paperReady = useMemo(() => events.some((e) => e.event_type === "paper.ready"), [events]);
  const lastSource = events.length ? events[events.length - 1].source : undefined;

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
            <div className="ml-auto flex items-center gap-1 rounded-lg bg-chip p-[3px]">
              {(["all", "milestones"] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className="rounded-md px-[11px] py-[5px] text-[11.5px] font-semibold leading-none"
                  style={
                    filter === f
                      ? { background: "var(--rightBg)", color: "var(--text)", boxShadow: "0 1px 2px rgba(0,0,0,.12)" }
                      : { background: "transparent", color: "var(--text3)" }
                  }
                >
                  {f === "all" ? "Activity" : "Milestones"}
                </button>
              ))}
            </div>
          </div>
          <div className="mt-[9px] flex items-center gap-[10px] pl-[19px]">
            <span className="line-clamp-2 min-w-0 flex-1 text-[12.5px] leading-[1.45] text-ink-2">
              {question}
            </span>
            <span className="shrink-0 self-start whitespace-nowrap font-mono text-[11px] font-medium leading-none text-ink-3">
              {agents} agents · {confirmed} confirmed
              {paperReady && (
                <>
                  {" · "}
                  <Link to={`/campaign/${id}/paper`} className="underline hover:text-ink">
                    paper
                  </Link>
                </>
              )}
            </span>
          </div>
        </div>

        {/* timeline */}
        <div
          ref={scrollRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
          }}
          className="pp-scroll min-h-0 flex-1 overflow-y-auto px-[26px] pb-2 pt-5"
        >
          <Timeline
            events={events}
            start={c?.started_at ?? null}
            filter={filter}
            live={{ active: sv.active && connected, source: lastSource }}
          />
        </div>

        {/* composer */}
        <div className="shrink-0 px-[26px] pb-4 pt-[10px]">
          {id && <Composer campaignId={id} status={sv} />}
        </div>
      </main>

      <RightPanel
        tree={c?.hypothesis_tree}
        question={question}
        beliefs={beliefs}
        events={events}
        open={rightOpen}
        onToggle={() => setRightOpen((o) => !o)}
      />
    </>
  );
}
