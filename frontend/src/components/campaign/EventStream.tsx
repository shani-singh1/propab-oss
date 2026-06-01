import { useMemo, useState } from "react";
import { useLiveStore } from "../../store";
import type { PropabEvent } from "../../types";
import { eventLabel, isErrorEvent, phaseOf, shortId, type Phase } from "../../lib/events";
import { fmtClock } from "../../lib/format";

type Filter = "all" | "experiment" | "tool" | "code" | "llm" | "errors";

const PHASE_DOT: Record<Phase, string> = {
  campaign: "bg-brand",
  literature: "bg-brand-dim",
  hypothesis: "bg-brand",
  experiment: "bg-running",
  tool: "bg-text-secondary",
  code: "bg-warning",
  llm: "bg-brand-dim",
  synthesis: "bg-confirmed",
  paper: "bg-confirmed",
  other: "bg-inconclusive",
};

export function EventStream() {
  const events = useLiveStore((s) => s.events);
  const [filter, setFilter] = useState<Filter>("all");
  const [q, setQ] = useState("");
  const [openId, setOpenId] = useState<string | null>(null);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const out: PropabEvent[] = [];
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      const ph = phaseOf(e.event_type);
      if (filter === "errors" && !isErrorEvent(e.event_type)) continue;
      if (filter !== "all" && filter !== "errors" && ph !== filter) continue;
      if (needle) {
        const hay = `${e.event_type} ${e.step} ${JSON.stringify(e.payload)}`.toLowerCase();
        if (!hay.includes(needle)) continue;
      }
      out.push(e);
      if (out.length >= 400) break;
    }
    return out;
  }, [events, filter, q]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold">Event stream</h2>
          <span className="text-[11px] text-text-muted">{events.length} events</span>
        </div>
        <div className="flex items-center gap-1">
          {(["all", "experiment", "tool", "code", "llm", "errors"] as Filter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`rounded-md px-2 py-1 text-[11px] capitalize transition ${
                filter === f ? "bg-brand/15 text-brand" : "text-text-muted hover:bg-raised"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>
      <div className="border-b border-border px-4 py-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search events…"
          className="w-full rounded-md border border-border bg-bg px-2.5 py-1.5 text-xs outline-none focus:border-brand"
        />
      </div>
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {filtered.length === 0 ? (
          <div className="px-4 py-6 text-sm text-text-muted">No matching events.</div>
        ) : (
          filtered.map((e) => (
            <Row
              key={e.event_id}
              e={e}
              open={openId === e.event_id}
              onToggle={() => setOpenId(openId === e.event_id ? null : e.event_id)}
            />
          ))
        )}
      </div>
    </div>
  );
}

function Row({ e, open, onToggle }: { e: PropabEvent; open: boolean; onToggle: () => void }) {
  const ph = phaseOf(e.event_type);
  const err = isErrorEvent(e.event_type);
  return (
    <div className="animate-fadeInUp border-b border-border/50">
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-2.5 px-4 py-1.5 text-left transition hover:bg-raised/50"
      >
        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${err ? "bg-refuted" : PHASE_DOT[ph]}`} />
        <span className="w-14 shrink-0 font-mono text-[10px] text-text-muted">{fmtClock(e.timestamp)}</span>
        <span className={`min-w-0 flex-1 truncate text-sm ${err ? "text-refuted" : "text-text-primary"}`}>
          {eventLabel(e)}
        </span>
        {e.hypothesis_id && (
          <span className="shrink-0 font-mono text-[10px] text-text-muted">{shortId(e.hypothesis_id)}</span>
        )}
      </button>
      {open && <Payload e={e} />}
    </div>
  );
}

function Payload({ e }: { e: PropabEvent }) {
  const p = e.payload || {};
  // Special-case the high-signal payloads for readability.
  const blocks: { label: string; text: string }[] = [];
  if (e.event_type === "llm.prompt" && p.prompt) blocks.push({ label: "prompt", text: String(p.prompt) });
  if (e.event_type === "llm.response" && (p.response ?? p.text))
    blocks.push({ label: "response", text: String(p.response ?? p.text) });
  if (e.event_type.startsWith("code.") && (p.code || p.stdout))
    blocks.push({ label: p.code ? "code" : "stdout", text: String(p.code ?? p.stdout) });

  return (
    <div className="animate-fadeInUp bg-bg/60 px-4 pb-3 pt-1">
      {blocks.map((b) => (
        <div key={b.label} className="mb-2">
          <div className="mb-1 text-[10px] uppercase tracking-wide text-text-muted">{b.label}</div>
          <pre className="max-h-56 overflow-auto scrollbar-thin whitespace-pre-wrap break-words rounded-md border border-border bg-surface p-2 font-mono text-[11px] leading-relaxed text-text-secondary">
            {b.text}
          </pre>
        </div>
      ))}
      {blocks.length === 0 && (
        <pre className="max-h-56 overflow-auto scrollbar-thin whitespace-pre-wrap break-words rounded-md border border-border bg-surface p-2 font-mono text-[11px] leading-relaxed text-text-secondary">
          {JSON.stringify(p, null, 2)}
        </pre>
      )}
      <div className="mt-1 font-mono text-[10px] text-text-muted">
        {e.event_type} · {e.step} · {e.source}
      </div>
    </div>
  );
}
