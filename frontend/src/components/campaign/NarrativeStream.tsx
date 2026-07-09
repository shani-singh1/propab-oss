import { useEffect, useMemo, useRef, useState } from "react";
import type { DiscoverySummary, NarrativeItem, OrchestratorGroup, RoundStat, RoundView, Worker } from "../../lib/model";
import { roundPhaseNotes, roundStats } from "../../lib/model";
import type { PropabEvent } from "../../types";
import { eventLabel, orchestratorGroupSummary, orchestratorView, type OrchestratorView } from "../../lib/events";
import { fmtElapsed, fmtMetric, fmtOffset } from "../../lib/format";
import { StatusDot, WorkerStatusMeta } from "./bits";
import { usePrefersReducedMotion } from "./followEdge";
import WorkerDetail from "./WorkerDetail";
import EventCard from "./EventCards";

// ── Round group card ─────────────────────────────────────────────────────────
// A completed round collapses to a one-line summary — now enriched with a
// proportional verdict bar, an activity sparkline, and a best-metric delta — and
// the live round is expanded by default. Expanding reveals the orchestrator's
// phase voice for the round, the per-hypothesis verdicts (each openable into full
// worker detail), a compute stat line, and a lazy "raw events" reveal.

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

// A stacked, proportional verdict bar. Running (not-yet-decided) hypotheses show
// as a faint segment so the live round reads as "in progress", not empty.
function VerdictBar({ r }: { r: RoundView }) {
  const running = r.workers.filter((w) => w.status === "running").length;
  const failed = r.workers.filter((w) => w.status === "failed").length;
  const segs = [
    { n: r.confirmed, color: "var(--green)", label: "confirmed" },
    { n: r.refuted, color: "var(--red)", label: "refuted" },
    { n: r.inconclusive, color: "var(--text3)", label: "inconclusive" },
    { n: failed, color: "var(--red)", label: "failed" },
    { n: running, color: "var(--text4)", label: "running", live: true },
  ].filter((s) => s.n > 0);
  const total = segs.reduce((a, s) => a + s.n, 0);
  if (!total) return null;
  return (
    <div
      className="mt-[8px] flex h-[5px] w-full overflow-hidden rounded-full bg-chip"
      role="img"
      aria-label={segs.map((s) => `${s.n} ${s.label}`).join(", ")}
    >
      {segs.map((s) => (
        <span
          key={s.label}
          className={s.live ? "animate-ppulse motion-reduce:animate-none" : ""}
          style={{ width: `${(s.n / total) * 100}%`, background: s.color }}
        />
      ))}
    </div>
  );
}

// A compact activity sparkline (event volume across the round's own time span).
function Sparkline({ data, live }: { data: number[]; live: boolean }) {
  const max = Math.max(1, ...data);
  if (data.every((d) => d === 0)) return null;
  return (
    <span
      className="flex h-[18px] w-[62px] items-end gap-[1px] text-ink-3"
      aria-hidden
      title="activity over the round"
    >
      {data.map((d, i) => {
        const last = live && i === data.length - 1 && d > 0;
        return (
          <span
            key={i}
            className={`flex-1 rounded-[1px] ${last ? "animate-ppulse motion-reduce:animate-none" : ""}`}
            style={{
              height: `${d === 0 ? 0 : Math.max(11, (d / max) * 100)}%`,
              minHeight: d > 0 ? 1 : 0,
              background: last ? "var(--text)" : "currentColor",
            }}
          />
        );
      })}
    </span>
  );
}

// The per-round best-metric delta — calm, never an alarm. Improvement in the
// campaign's optimization direction reads green; a non-improving round is muted.
function BestDelta({ stat, discovery }: { stat: RoundStat; discovery: DiscoverySummary | null }) {
  if (stat.delta == null || Math.abs(stat.delta) < 1e-9) return null;
  const higher = discovery?.direction !== "lower_is_better";
  const improving = higher ? stat.delta > 0 : stat.delta < 0;
  const name = discovery?.metricName ?? "metric";
  return (
    <span
      className="flex items-center gap-[4px] font-mono text-[10.5px] leading-none"
      style={{ color: improving ? "var(--green)" : "var(--text3)" }}
      title={`best ${name} moved by ${stat.delta > 0 ? "+" : ""}${fmtMetric(stat.delta)} this round`}
    >
      <span aria-hidden>{improving ? "▲" : "▼"}</span>
      <span className="tabular-nums">
        {stat.delta > 0 ? "+" : "−"}
        {fmtMetric(Math.abs(stat.delta))}
      </span>
      <span className="text-ink-4">{name}</span>
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

// The orchestrator's phase voice for a round — folded into the story as calm,
// muted lines rather than loud rows.
function PhaseNotes({ notes }: { notes: string[] }) {
  if (!notes.length) return null;
  return (
    <div className="mb-[12px] flex flex-col gap-[5px] border-l-2 border-edge pl-[11px]">
      {notes.map((n, i) => (
        <div key={i} className="text-[11.5px] leading-[1.45] text-ink-3">
          {n}
        </div>
      ))}
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
  stat,
  start,
  now,
  discovery,
  defaultOpen,
}: {
  round: RoundView;
  stat: RoundStat | undefined;
  start: string | null;
  now: number;
  discovery: DiscoverySummary | null;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const running = round.status === "running";
  const title = round.isSetup ? "Setup" : `Round ${round.number}`;
  const dur = fmtElapsed(round.startedAt, running ? now : round.endedAt);
  const notes = useMemo(() => roundPhaseNotes(round), [round]);

  return (
    <div className="mb-[14px] animate-ptick overflow-hidden rounded-[10px] border border-edge bg-rail/40 motion-reduce:animate-none">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="pp-row flex w-full items-start gap-[11px] px-[15px] py-[12px] text-left"
      >
        <span className="mt-[3px]">
          <StatusDot color={running ? "var(--text)" : "var(--text3)"} live={running} size={9} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-[9px] gap-y-[4px]">
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
          <div className="mt-[6px] flex flex-wrap items-center gap-x-[12px] gap-y-[4px]">
            <VerdictChips r={round} />
            {stat && <BestDelta stat={stat} discovery={discovery} />}
          </div>
          <VerdictBar r={round} />
        </div>
        {stat && <Sparkline data={stat.spark} live={running} />}
        <span className="mt-[1px] shrink-0 font-mono text-[10.5px] text-ink-4">{dur}</span>
        <span className="mt-[1px] shrink-0 text-[11px] text-ink-4">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div className="border-t border-line px-[15px] py-[13px]">
          <PhaseNotes notes={notes} />

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

// ── Streaming orchestrator voice ─────────────────────────────────────────────
// The live "what it's doing right now" line at the active edge. A subtle
// typewriter reveal on each new phase narration, with a caret while typing;
// motion-reduce shows the full line immediately with no caret.

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

function LiveEdge({ label }: { label?: string }) {
  const reduced = usePrefersReducedMotion();
  const { shown, typing } = useTypewriter(label, !reduced);
  return (
    <div className="flex items-center gap-[10px] px-[4px] py-[2px]" aria-live="polite">
      <span className="h-[9px] w-[9px] animate-ppulse rounded-full bg-ink motion-reduce:animate-none" />
      <span className="min-w-0 flex-1 text-[12.5px] leading-[1.4] text-ink-2">
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
  );
}

// ── Orchestrator activity lane (redesign §4) ─────────────────────────────────
// The mid-panel's headline content: the orchestrator narrating the campaign in
// plain language. A consecutive run of orchestrator.* events collapses into one
// block (open by default so its thinking is never hidden), each entry a row
// color-coded by kind — literature review, reasoning, a hypothesis written, or a
// verdict decision. Worker chatter lives in the right panel; this lane stays the
// orchestrator's voice.

function OrchestratorRow({ v, e, start }: { v: OrchestratorView; e: PropabEvent; start: string | null }) {
  return (
    <div className="flex items-start gap-[9px] border-t border-line px-[15px] py-[9px] first:border-t-0">
      <span className="mt-[4px]">
        <StatusDot color={v.dotColor} size={6} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-[8px] gap-y-[4px]">
          <span className="text-[12px] font-semibold leading-[1.35] text-ink">{v.label}</span>
          {v.meta.map((m, i) => (
            <span
              key={i}
              className="rounded bg-chip px-[6px] py-[2px] font-mono text-[9.5px] leading-none text-ink-3"
            >
              {m}
            </span>
          ))}
        </div>
        {v.detail && (
          <div className="mt-[3px] text-[11.5px] leading-[1.45] text-ink-2">{v.detail}</div>
        )}
        {v.note && <div className="mt-[3px] text-[11px] leading-[1.4] text-ink-3">{v.note}</div>}
      </div>
      <span className="shrink-0 font-mono text-[10px] leading-[1.6] text-ink-4">
        {fmtOffset(start, e.timestamp)}
      </span>
    </div>
  );
}

function OrchestratorGroupCard({ group, start }: { group: OrchestratorGroup; start: string | null }) {
  const [open, setOpen] = useState(true);
  const views = useMemo(() => group.events.map(orchestratorView), [group]);
  const summary = useMemo(() => orchestratorGroupSummary(views), [views]);

  return (
    <div className="mb-[14px] animate-ptick overflow-hidden rounded-[10px] border border-edge bg-rail/30 motion-reduce:animate-none">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="pp-row flex w-full items-center gap-[10px] px-[15px] py-[11px] text-left"
      >
        <StatusDot color="var(--text)" size={8} />
        <span className="shrink-0 font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-3">
          Orchestrator
        </span>
        <span className="min-w-0 flex-1 truncate text-[12px] leading-none text-ink-2">{summary}</span>
        <span className="shrink-0 font-mono text-[10px] text-ink-4">{group.events.length}</span>
        <span className="shrink-0 text-[11px] text-ink-4">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="flex flex-col border-t border-line">
          {views.map((v, i) => (
            <OrchestratorRow key={group.events[i].event_id} v={v} e={group.events[i]} start={start} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── The stream ───────────────────────────────────────────────────────────────
// Standalone milestone events are dispatched through the special-event card
// registry (EventCards) — a breakthrough, a confirmed finding, a paper, and the
// baseline each get a purpose-built card; everything else falls back to a plain
// milestone row.

export default function NarrativeStream({
  narrative,
  start,
  now,
  live,
  discovery = null,
  campaignId,
}: {
  narrative: NarrativeItem[];
  start: string | null;
  now: number;
  live: { active: boolean; label?: string } | null;
  discovery?: DiscoverySummary | null;
  campaignId?: string;
}) {
  // Per-round derived stats (verdict split feeds the bar; sparkline + best-metric
  // delta threaded across rounds in narrative order).
  const stats = useMemo(() => {
    const rounds = narrative
      .filter((it): it is Extract<NarrativeItem, { kind: "round" }> => it.kind === "round")
      .map((it) => it.round);
    return roundStats(rounds, {
      baseline: discovery?.meter.hasMetric ? discovery.meter.baseline : null,
      direction: discovery?.direction ?? "higher_is_better",
      metricName: discovery?.metricName ?? null,
    });
  }, [narrative, discovery]);

  // Index of the last running round — that one is live-expanded by default.
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

      {narrative.map((it, i) => {
        if (it.kind === "milestone")
          return (
            <EventCard
              key={it.event.event_id}
              e={it.event}
              start={start}
              discovery={discovery}
              campaignId={campaignId}
            />
          );
        if (it.kind === "orchestrator")
          return <OrchestratorGroupCard key={it.group.key} group={it.group} start={start} />;
        return (
          <RoundGroup
            key={it.round.key}
            round={it.round}
            stat={stats.get(it.round.key)}
            start={start}
            now={now}
            discovery={discovery}
            defaultOpen={it.round.status === "running" ? i === lastRoundIdx : it.round.isSetup && narrative.length === 1}
          />
        );
      })}

      {live?.active && <LiveEdge label={live.label} />}
    </div>
  );
}
