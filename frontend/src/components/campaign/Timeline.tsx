import { useMemo } from "react";
import type { PropabEvent } from "../../types";
import {
  eventDotColor,
  eventLabel,
  isMilestone,
  PHASE_LABEL,
  phaseOf,
} from "../../lib/events";
import { fmtOffset } from "../../lib/format";
import { Tag } from "../primitives";

// A short secondary detail line for events that carry useful payload.
function detailOf(e: PropabEvent): string | null {
  const p = e.payload || {};
  switch (e.event_type) {
    case "campaign.baseline_measured":
      return p.baseline_metric != null ? `baseline ${p.baseline_metric}` : null;
    case "hypothesis.generated":
      return p.count != null ? `${p.count} candidate hypotheses` : null;
    case "agent.completed":
      return p.confidence != null ? `confidence ${Number(p.confidence).toFixed(2)}` : null;
    case "tool.called":
    case "tool.result":
    case "tool.error":
      return p.tool ? String(p.tool) : null;
    case "synthesis.result_received":
    case "synthesis.ledger_updated":
      return p.statement ? String(p.statement) : p.summary ? String(p.summary) : null;
    case "paper.section_completed":
      return p.section ? String(p.section) : null;
    case "campaign.progress":
      return p.improvement_pct != null ? `improvement ${p.improvement_pct}%` : null;
    default:
      return typeof p.reason === "string" ? p.reason : typeof p.note === "string" ? p.note : null;
  }
}

// A short uppercase tag for a few emphasized event kinds.
function tagOf(e: PropabEvent): { text: string; color: string; bg: string } | null {
  const t = e.event_type;
  if (t.endsWith(".error") || t.endsWith(".failed") || t === "campaign.budget_exhausted")
    return { text: "conflict", color: "var(--red)", bg: "var(--redDim)" };
  if (t === "campaign.breakthrough" || t === "synthesis.breakthrough")
    return { text: "breakthrough", color: "var(--green)", bg: "var(--greenDim)" };
  if (t.startsWith("synthesis."))
    return { text: "belief update", color: "var(--text3)", bg: "transparent" };
  return null;
}

function Row({ e, start, last }: { e: PropabEvent; start: string | null; last: boolean }) {
  const detail = detailOf(e);
  const tag = tagOf(e);
  const source = e.source || PHASE_LABEL[phaseOf(e.event_type)].toLowerCase();
  return (
    <div className="relative flex gap-[14px] pb-[18px]">
      <div className="relative flex w-[14px] shrink-0 justify-center">
        {!last && (
          <div
            className="absolute bottom-[-18px] top-[6px] w-px"
            style={{ background: "var(--divider)" }}
          />
        )}
        <div
          className="relative mt-[3px] h-[9px] w-[9px] rounded-full"
          style={{ background: eventDotColor(e), border: "2px solid var(--centerBg)" }}
        />
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-[5px] flex items-center gap-[7px]">
          <span className="font-mono text-[11px] font-semibold leading-none text-ink-2">
            {source}
          </span>
          <span className="text-[10px] text-ink-4">·</span>
          <span className="font-mono text-[10.5px] leading-none text-ink-3">
            {fmtOffset(start, e.timestamp)}
          </span>
          {tag && (
            <span className="ml-[6px]">
              <Tag color={tag.color} bg={tag.bg}>
                {tag.text}
              </Tag>
            </span>
          )}
        </div>
        <div className="text-[13px] leading-[1.55] text-ink">{eventLabel(e)}</div>
        {detail && (
          <div className="mt-1 truncate font-mono text-[11.5px] leading-[1.4] text-ink-3">
            {detail}
          </div>
        )}
      </div>
    </div>
  );
}

export default function Timeline({
  events,
  start,
  filter,
  live,
}: {
  events: PropabEvent[];
  start: string | null;
  filter: "all" | "milestones";
  live: { active: boolean; source?: string } | null;
}) {
  const shown = useMemo(
    () => (filter === "milestones" ? events.filter((e) => isMilestone(e.event_type)) : events),
    [events, filter],
  );

  return (
    <div className="max-w-[680px]">
      {shown.length === 0 && (
        <div className="py-6 text-[12.5px] leading-relaxed text-ink-3">
          No activity yet. Events will stream in as the campaign runs.
        </div>
      )}
      {shown.map((e, i) => (
        <Row key={e.event_id} e={e} start={start} last={i === shown.length - 1 && !live?.active} />
      ))}

      {live?.active && (
        <div className="flex gap-[14px]">
          <div className="flex w-[14px] shrink-0 justify-center">
            <span className="mt-[3px] h-[9px] w-[9px] animate-ppulse rounded-full bg-ink" />
          </div>
          <div className="flex flex-1 items-center gap-2">
            <span className="text-[12.5px] leading-none text-ink-2">
              {live.source ? `${live.source} is running` : "campaign is running"}
            </span>
            <span className="flex gap-[3px]">
              <span className="h-[3px] w-[3px] animate-pdots rounded-full bg-ink-2" />
              <span className="h-[3px] w-[3px] animate-pdots rounded-full bg-ink-2 [animation-delay:0.2s]" />
              <span className="h-[3px] w-[3px] animate-pdots rounded-full bg-ink-2 [animation-delay:0.4s]" />
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
