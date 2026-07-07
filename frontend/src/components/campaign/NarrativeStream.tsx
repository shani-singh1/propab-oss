import { useState } from "react";
import type { NarrativeItem, RoundView, Worker } from "../../lib/model";
import type { PropabEvent } from "../../types";
import { eventLabel } from "../../lib/events";
import { fmtElapsed, fmtOffset } from "../../lib/format";
import { StatusDot, WorkerStatusMeta } from "./bits";
import WorkerDetail from "./WorkerDetail";

// ── Round group card ─────────────────────────────────────────────────────────
// A completed round collapses to a one-line summary with a verdict tally; the
// live round is expanded by default. Expanding reveals the per-hypothesis
// verdicts (each openable into full worker detail), a compute stat line, and a
// lazy "raw events" reveal — progressive disclosure over hundreds of events.

function tally(r: RoundView) {
  return [
    { n: r.confirmed, label: "confirmed", color: "var(--green)" },
    { n: r.refuted, label: "refuted", color: "var(--red)" },
    { n: r.inconclusive, label: "inconclusive", color: "var(--text3)" },
  ].filter((x) => x.n > 0);
}

function VerdictChips({ r }: { r: RoundView }) {
  const t = tally(r);
  if (!t.length)
    return <span className="font-mono text-[11px] text-ink-3">no verdicts yet</span>;
  return (
    <span className="flex flex-wrap items-center gap-[8px]">
      {t.map((x) => (
        <span key={x.label} className="font-mono text-[11px] leading-none" style={{ color: x.color }}>
          {x.n} {x.label}
        </span>
      ))}
    </span>
  );
}

function WorkerVerdictRow({ w, now }: { w: Worker; now: number }) {
  const [open, setOpen] = useState(false);
  const meta = WorkerStatusMeta(w.status);
  return (
    <div className="rounded-[7px] border border-line">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="pp-row flex w-full items-start gap-[9px] rounded-[7px] px-[10px] py-[8px] text-left"
      >
        <span className="mt-[3px]">
          <StatusDot color={meta.color} live={meta.live} size={7} />
        </span>
        <span className="min-w-0 flex-1 truncate text-[12px] leading-[1.4] text-ink-2">
          {w.text || `Hypothesis ${w.shortId}`}
        </span>
        <span className="shrink-0 font-mono text-[10px] leading-none" style={{ color: meta.color }}>
          {meta.label}
        </span>
        <span className="mt-[1px] shrink-0 text-[10px] text-ink-4">{open ? "▾" : "▸"}</span>
      </button>
      {open && <WorkerDetail w={w} now={now} />}
    </div>
  );
}

function RawEvents({ events, start }: { events: PropabEvent[]; start: string | null }) {
  const [open, setOpen] = useState(false);
  const [limit, setLimit] = useState(60);
  if (!events.length) return null;
  return (
    <div className="mt-[10px]">
      <button
        onClick={() => setOpen((o) => !o)}
        className="font-mono text-[10.5px] text-ink-3 hover:text-ink"
      >
        {open ? "▾ hide" : "▸ show"} {events.length} raw events
      </button>
      {open && (
        <ul className="pp-scroll mt-[7px] max-h-[240px] overflow-auto border-l border-line pl-[11px]">
          {events.slice(0, limit).map((e) => (
            <li key={e.event_id} className="flex gap-[9px] py-[3px] font-mono text-[10.5px] leading-[1.5]">
              <span className="shrink-0 text-ink-4">{fmtOffset(start, e.timestamp)}</span>
              <span className="shrink-0 text-ink-3">{e.source || "·"}</span>
              <span className="min-w-0 truncate text-ink-2">{eventLabel(e)}</span>
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
      )}
    </div>
  );
}

function RoundGroup({
  round,
  start,
  now,
  defaultOpen,
}: {
  round: RoundView;
  start: string | null;
  now: number;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const running = round.status === "running";
  const title = round.isSetup ? "Setup" : `Round ${round.number}`;
  const dur = fmtElapsed(round.startedAt, running ? now : round.endedAt);

  return (
    <div className="mb-[14px] overflow-hidden rounded-[10px] border border-edge bg-rail/40">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="pp-row flex w-full items-center gap-[11px] px-[15px] py-[12px] text-left"
      >
        <StatusDot color={running ? "var(--text)" : "var(--text3)"} live={running} size={9} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-[9px]">
            <span className="text-[13px] font-semibold leading-none text-ink">{title}</span>
            {running && (
              <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.1em] text-ink-3">
                live
              </span>
            )}
            {round.hypothesesGenerated > 0 && (
              <span className="font-mono text-[10.5px] leading-none text-ink-3">
                {round.hypothesesGenerated} hypotheses
              </span>
            )}
          </div>
          <div className="mt-[6px] flex items-center gap-[10px]">
            <VerdictChips r={round} />
          </div>
        </div>
        <span className="shrink-0 font-mono text-[10.5px] text-ink-4">{dur}</span>
        <span className="shrink-0 text-[11px] text-ink-4">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div className="border-t border-line px-[15px] py-[13px]">
          {/* compute stat line */}
          <div className="mb-[12px] flex flex-wrap gap-[6px]">
            {[
              ["LLM calls", round.llmCalls],
              ["tool calls", round.toolCalls],
              ["code runs", round.codeRuns],
              ...(round.errors ? ([["errors", round.errors]] as [string, number][]) : []),
              ...(round.marginalReturn != null
                ? ([["marginal", round.marginalReturn.toFixed(2)]] as [string, string][])
                : []),
            ].map(([label, n]) => (
              <span
                key={label}
                className="rounded bg-chip px-[8px] py-[4px] font-mono text-[10px] leading-none text-ink-3"
                style={label === "errors" ? { color: "var(--red)" } : undefined}
              >
                <span className="font-semibold text-ink-2">{n}</span> {label}
              </span>
            ))}
          </div>

          {/* per-hypothesis verdicts */}
          {round.workers.length > 0 && (
            <div className="flex flex-col gap-[7px]">
              {round.workers.map((w) => (
                <WorkerVerdictRow key={w.hypothesisId} w={w} now={now} />
              ))}
            </div>
          )}

          {round.workers.length === 0 && (
            <div className="text-[12px] leading-relaxed text-ink-3">
              {round.isSetup
                ? "Preparing the campaign — building the literature prior and measuring the baseline."
                : "Dispatching hypotheses…"}
            </div>
          )}

          <RawEvents events={round.events} start={start} />
        </div>
      )}
    </div>
  );
}

// ── Standalone lifecycle milestone ───────────────────────────────────────────

function milestoneMeta(e: PropabEvent): { color: string; emphatic: boolean } {
  const t = e.event_type;
  if (t === "campaign.breakthrough" || (t === "campaign.completed" && !e.payload?.stop_reason))
    return { color: "var(--green)", emphatic: true };
  if (t === "campaign.budget_exhausted" || t === "session.failed")
    return { color: "var(--red)", emphatic: true };
  if (t === "paper.ready") return { color: "var(--green)", emphatic: false };
  return { color: "var(--text)", emphatic: false };
}

function MilestoneRow({ e, start }: { e: PropabEvent; start: string | null }) {
  const m = milestoneMeta(e);
  return (
    <div
      className="mb-[14px] flex items-center gap-[11px] rounded-[10px] px-[15px] py-[12px]"
      style={{
        border: `1px solid ${m.emphatic ? m.color : "var(--border)"}`,
        background: m.emphatic ? "var(--chip)" : "transparent",
      }}
    >
      <StatusDot color={m.color} size={9} />
      <span className="flex-1 text-[13px] font-semibold leading-none text-ink">
        {eventLabel(e)}
      </span>
      <span className="font-mono text-[10.5px] text-ink-4">{fmtOffset(start, e.timestamp)}</span>
    </div>
  );
}

// ── The stream ───────────────────────────────────────────────────────────────

export default function NarrativeStream({
  narrative,
  start,
  now,
  live,
}: {
  narrative: NarrativeItem[];
  start: string | null;
  now: number;
  live: { active: boolean; label?: string } | null;
}) {
  // Index of the last round item — that one is live-expanded by default.
  let lastRoundIdx = -1;
  narrative.forEach((it, i) => {
    if (it.kind === "round" && it.round.status === "running") lastRoundIdx = i;
  });

  return (
    <div className="mx-auto max-w-[720px]">
      {narrative.length === 0 && (
        <div className="py-6 text-[12.5px] leading-relaxed text-ink-3">
          No activity yet. The orchestrator's narrative will stream in as the
          campaign builds its prior, measures the baseline, and opens its first
          round.
        </div>
      )}

      {narrative.map((it, i) =>
        it.kind === "milestone" ? (
          <MilestoneRow key={it.event.event_id} e={it.event} start={start} />
        ) : (
          <RoundGroup
            key={it.round.key}
            round={it.round}
            start={start}
            now={now}
            defaultOpen={it.round.status === "running" ? i === lastRoundIdx : it.round.isSetup && narrative.length === 1}
          />
        ),
      )}

      {live?.active && (
        <div className="flex items-center gap-[10px] px-[4px] py-[2px]">
          <span className="h-[9px] w-[9px] animate-ppulse rounded-full bg-ink motion-reduce:animate-none" />
          <span className="text-[12.5px] leading-none text-ink-2">
            {live.label || "campaign is running"}
          </span>
          <span className="flex gap-[3px]">
            <span className="h-[3px] w-[3px] animate-pdots rounded-full bg-ink-2 motion-reduce:animate-none" />
            <span className="h-[3px] w-[3px] animate-pdots rounded-full bg-ink-2 [animation-delay:0.2s] motion-reduce:animate-none" />
            <span className="h-[3px] w-[3px] animate-pdots rounded-full bg-ink-2 [animation-delay:0.4s] motion-reduce:animate-none" />
          </span>
        </div>
      )}
    </div>
  );
}
