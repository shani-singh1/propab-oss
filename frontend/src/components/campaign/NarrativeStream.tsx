import { useEffect, useRef, useState } from "react";
import type {
  DecisionItem,
  DiscoverySummary,
  GenerationItem,
  MechanicsItem,
  ReasoningItem,
  TimelineItem,
} from "../../lib/model";
import type { PropabEvent } from "../../types";
import { eventLabel } from "../../lib/events";
import { fmtOffset } from "../../lib/format";
import { StatusDot } from "./bits";
import { usePrefersReducedMotion } from "./followEdge";
import EventCard from "./EventCards";

// ── The flat narrative timeline (redesign §4) ────────────────────────────────
// ONE continuous document, read top-to-bottom like a lab notebook — no group
// cards, no "ORCHESTRATOR" headers, no sparklines/verdict-bars/stat-pills. Every
// row shares a single dim left timestamp gutter; the prose is capped to a
// readable measure and left-aligned. Hierarchy is by IMPORTANCE:
//   findings LOUD · reasoning plain prose · decisions inline (verdict = the only
//   color) · generation boundaries as hairlines · mechanics dim + collapsed.

// A shared readable measure for the prose column (guide §3/§5).
const MEASURE = "max-w-[680px]";

// The dim, monospace left gutter every row shares. NB: we align via inline style,
// not the `text-right` utility — the design tokens define a `right` color, so
// Tailwind's `text-right` also emits `color: var(--rightBg)`, which would clobber
// the `text-ink-4` dim ink and render the gutter near-invisible.
function Gutter({ label }: { label: string }) {
  return (
    <div
      className="w-[50px] shrink-0 select-none pt-[2px] font-mono text-[10.5px] leading-[1.6] text-ink-4"
      style={{ textAlign: "right" }}
    >
      {label}
    </div>
  );
}

// Tier 2 — a plain reasoning sentence, in the reading color at normal weight.
function ReasoningRow({ it, start }: { it: ReasoningItem; start: string | null }) {
  return (
    <div className="flex gap-[14px] py-[6px]">
      <Gutter label={fmtOffset(start, it.at)} />
      <div className={`min-w-0 flex-1 ${MEASURE}`}>
        <p className="text-[13.5px] font-normal leading-[1.62] text-ink">{it.text}</p>
        {it.quote && (
          <p className="mt-[5px] border-l-2 border-edge pl-[10px] text-[12.5px] leading-[1.5] text-ink-3">
            “{it.quote}”
          </p>
        )}
      </div>
    </div>
  );
}

// Tier 2 — a verdict, inline: a small colored dot + one line, a quiet quoted
// hypothesis, tiny metric/p chips, and a subtle "→ next move". The verdict color
// is the ONLY emphasis.
function DecisionRow({ it, start }: { it: DecisionItem; start: string | null }) {
  return (
    <div className="flex gap-[14px] py-[6px]">
      <Gutter label={fmtOffset(start, it.at)} />
      <div className={`min-w-0 flex-1 ${MEASURE}`}>
        <div className="flex items-start gap-[9px]">
          <span className="mt-[6px]">
            <StatusDot color={it.dotColor} size={7} />
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-[13px] leading-[1.5] text-ink">{it.label}</div>
            {it.claim && (
              <div className="mt-[3px] text-[12.5px] leading-[1.5] text-ink-3">“{it.claim}”</div>
            )}
            {(it.chips.length > 0 || it.next) && (
              <div className="mt-[5px] flex flex-wrap items-center gap-x-[12px] gap-y-[3px] font-mono text-[10.5px] leading-none text-ink-3">
                {it.chips.map((c, i) => (
                  <span key={i} className="tabular-nums">
                    {c}
                  </span>
                ))}
                {it.next && (
                  <span className="flex items-center gap-[4px]">
                    <span aria-hidden style={{ color: it.dotColor }}>
                      →
                    </span>
                    {it.next}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// A new-round boundary: a hairline divider with a tiny gutter label — NOT a card.
function GenerationRow({ it }: { it: GenerationItem }) {
  return (
    <div className="flex items-center gap-[14px] py-[15px]">
      <div
        className="w-[50px] shrink-0 font-mono text-[10px] uppercase tracking-[0.08em] text-ink-3"
        style={{ textAlign: "right" }}
      >
        {it.label}
      </div>
      <div className={`h-px flex-1 bg-line ${MEASURE}`} />
    </div>
  );
}

// Tier 3 — the folded mechanics run: one dim inline line, expandable to raw.
function MechanicsRow({ it, start }: { it: MechanicsItem; start: string | null }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex gap-[14px] py-[3px]">
      <Gutter label={fmtOffset(start, it.at)} />
      <div className={`min-w-0 flex-1 ${MEASURE}`}>
        <button
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="flex items-center gap-[7px] font-mono text-[11px] leading-[1.5] text-ink-3 transition-colors hover:text-ink"
        >
          <span aria-hidden className="text-ink-4">
            ·
          </span>
          <span>{it.summary}</span>
          <span aria-hidden className="text-ink-4">
            {open ? "▾" : "▸"}
          </span>
        </button>
        {open && <RawList events={it.events} start={start} />}
      </div>
    </div>
  );
}

// The raw event list behind a mechanics row — quiet, scrollable, capped.
function RawList({ events, start }: { events: PropabEvent[]; start: string | null }) {
  const [limit, setLimit] = useState(40);
  return (
    <ul className="pp-scroll mt-[7px] max-h-[220px] overflow-auto border-l border-line pl-[11px]">
      {events.slice(0, limit).map((e) => (
        <li
          key={e.event_id}
          className="flex gap-[9px] py-[2px] font-mono text-[10.5px] leading-[1.5]"
        >
          <span className="shrink-0 text-ink-4">{fmtOffset(start, e.timestamp)}</span>
          <span className="shrink-0 text-ink-4">{e.source || "·"}</span>
          <span className="min-w-0 truncate text-ink-3">{eventLabel(e)}</span>
        </li>
      ))}
      {events.length > limit && (
        <li>
          <button
            onClick={() => setLimit((l) => l + 200)}
            className="py-[4px] font-mono text-[10px] text-ink-4 hover:text-ink"
          >
            +{events.length - limit} more…
          </button>
        </li>
      )}
    </ul>
  );
}

// ── Live edge — the streaming "what it's doing right now" line ────────────────
// A subtle typewriter reveal on each new phase narration, with a caret while
// typing; motion-reduce shows the full line immediately with no caret.
function useTypewriter(text: string | undefined, enabled: boolean): { shown: string; typing: boolean } {
  const [shown, setShown] = useState(text ?? "");
  const [typing, setTyping] = useState(false);
  const prev = useRef<string | undefined>(text);
  useEffect(() => {
    if (!enabled || !text) {
      setShown(text ?? "");
      setTyping(false);
      prev.current = text;
      return;
    }
    if (text === prev.current) return;
    prev.current = text;
    setShown("");
    setTyping(true);
    let i = 0;
    const id = window.setInterval(() => {
      i += 1;
      setShown(text.slice(0, i));
      if (i >= text.length) {
        setTyping(false);
        window.clearInterval(id);
      }
    }, 16);
    return () => window.clearInterval(id);
  }, [text, enabled]);
  return { shown, typing };
}

function LiveEdge({ label, start }: { label?: string; start: string | null }) {
  const reduced = usePrefersReducedMotion();
  const { shown, typing } = useTypewriter(label, !reduced);
  return (
    <div className="flex gap-[14px] py-[8px]" aria-live="polite">
      <Gutter label={start ? "now" : ""} />
      <div className={`flex min-w-0 flex-1 items-center gap-[10px] ${MEASURE}`}>
        <span className="h-[8px] w-[8px] shrink-0 animate-ppulse rounded-full bg-ink motion-reduce:animate-none" />
        <span className="min-w-0 flex-1 text-[13px] leading-[1.5] text-ink-2">
          {shown || "campaign is running"}
          {typing && (
            <span className="ml-[1px] inline-block h-[12px] w-[6px] translate-y-[1px] animate-ppulse bg-ink-3 align-middle motion-reduce:animate-none" />
          )}
        </span>
        {!typing && (
          <span className="flex shrink-0 gap-[3px]">
            <span className="h-[3px] w-[3px] animate-pdots rounded-full bg-ink-2 motion-reduce:animate-none" />
            <span className="h-[3px] w-[3px] animate-pdots rounded-full bg-ink-2 [animation-delay:0.2s] motion-reduce:animate-none" />
            <span className="h-[3px] w-[3px] animate-pdots rounded-full bg-ink-2 [animation-delay:0.4s] motion-reduce:animate-none" />
          </span>
        )}
      </div>
    </div>
  );
}

// ── The stream ───────────────────────────────────────────────────────────────
export default function NarrativeStream({
  timeline,
  start,
  live,
  discovery = null,
  campaignId,
}: {
  timeline: TimelineItem[];
  start: string | null;
  live: { active: boolean; label?: string } | null;
  discovery?: DiscoverySummary | null;
  campaignId?: string;
}) {
  return (
    <div className="w-full pb-4">
      {timeline.length === 0 && (
        <div className="flex gap-[14px] py-6">
          <Gutter label="" />
          <div className={`flex-1 text-[13px] leading-[1.6] text-ink-3 ${MEASURE}`}>
            No activity yet. The orchestrator's narrative will stream in as the campaign
            builds its prior, measures the baseline, and opens its first round.
          </div>
        </div>
      )}

      {timeline.map((it) => {
        switch (it.tier) {
          case "reasoning":
            return <ReasoningRow key={it.key} it={it} start={start} />;
          case "decision":
            return <DecisionRow key={it.key} it={it} start={start} />;
          case "generation":
            return <GenerationRow key={it.key} it={it} />;
          case "mechanics":
            return <MechanicsRow key={it.key} it={it} start={start} />;
          case "finding":
            return (
              <div key={it.key} className="flex gap-[14px] pt-[4px]">
                <div className="w-[50px] shrink-0" />
                <div className="min-w-0 max-w-[720px] flex-1">
                  <EventCard
                    e={it.event}
                    start={start}
                    discovery={discovery}
                    campaignId={campaignId}
                  />
                </div>
              </div>
            );
        }
      })}

      {live?.active && <LiveEdge label={live.label} start={start} />}
    </div>
  );
}
