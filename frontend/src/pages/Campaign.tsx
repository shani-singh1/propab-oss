import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { useCampaignLive } from "../hooks/useCampaignLive";
import { useLiveStore } from "../store";
import { OverviewBar } from "../components/campaign/OverviewBar";
import { HypothesisTree } from "../components/campaign/HypothesisTree";
import { EventStream } from "../components/campaign/EventStream";
import { FindingsPanel } from "../components/campaign/FindingsPanel";
import { Spinner } from "../components/ui";

export default function Campaign() {
  const { id } = useParams<{ id: string }>();
  useCampaignLive(id);

  const campaign = useLiveStore((s) => s.campaign);
  const connected = useLiveStore((s) => s.connected);
  const error = useLiveStore((s) => s.error);
  const events = useLiveStore((s) => s.events);

  const paperReady = useMemo(
    () => events.some((e) => e.event_type === "paper.ready") ||
      Boolean(campaign?.event_counts_by_type?.["paper.ready"]),
    [events, campaign],
  );

  if (error && !campaign) {
    return (
      <div className="grid h-full place-items-center">
        <div className="text-center">
          <p className="text-refuted">Couldn’t load campaign.</p>
          <p className="mt-1 text-sm text-text-muted">{error}</p>
          <Link to="/" className="mt-3 inline-block text-brand hover:underline">
            ← Back to dashboard
          </Link>
        </div>
      </div>
    );
  }

  if (!campaign) {
    return (
      <div className="grid h-full place-items-center">
        <Spinner />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <OverviewBar state={campaign} connected={connected} paperReady={paperReady} />
      <div className="grid flex-1 grid-cols-1 gap-0 overflow-hidden lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        {/* Left: hypothesis tree */}
        <div className="min-h-0 border-r border-border bg-surface">
          <HypothesisTree tree={campaign.campaign.hypothesis_tree} />
        </div>
        {/* Right: events (top) + findings (bottom) */}
        <div className="grid min-h-0 grid-rows-[minmax(0,3fr)_minmax(0,2fr)]">
          <div className="min-h-0 border-b border-border bg-surface">
            <EventStream />
          </div>
          <div className="min-h-0 bg-surface">
            <FindingsPanel state={campaign} />
          </div>
        </div>
      </div>
    </div>
  );
}
