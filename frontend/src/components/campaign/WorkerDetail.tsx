import type { Worker } from "../../lib/model";
import { fmtElapsed } from "../../lib/format";
import { StatusDot, WorkerStatusMeta } from "./bits";

// Detailed view of one sub-agent: the hypothesis under test, live status/timing,
// the step trace, activity counts, and the latest code it generated. Rendered
// inline (expanded row) in the Workers/Tasks panels and reused by the narrative.
export default function WorkerDetail({ w, now }: { w: Worker; now: number }) {
  const meta = WorkerStatusMeta(w.status);
  const elapsed = fmtElapsed(w.startedAt, w.status === "running" ? now : w.endedAt);

  return (
    <div className="border-t border-line bg-chip/30 px-[16px] py-[13px]">
      {/* status line */}
      <div className="mb-[10px] flex flex-wrap items-center gap-x-[10px] gap-y-1 font-mono text-[10.5px] leading-none">
        <span className="flex items-center gap-[6px]" style={{ color: meta.color }}>
          <StatusDot color={meta.color} live={meta.live} size={7} />
          {meta.label}
        </span>
        {w.confidence != null && (
          <span className="text-ink-3">confidence {w.confidence.toFixed(2)}</span>
        )}
        <span className="text-ink-3">{elapsed}</span>
        {w.round != null && <span className="text-ink-4">round {w.round}</span>}
        <span className="ml-auto text-ink-4">{w.shortId}</span>
      </div>

      {/* activity counts */}
      <div className="mb-[12px] flex gap-[7px]">
        {[
          ["steps", w.steps.length],
          ["LLM", w.llmCalls],
          ["tools", w.toolCalls],
          ["runs", w.codeRuns],
          ...(w.errors ? ([["errors", w.errors]] as [string, number][]) : []),
        ].map(([label, n]) => (
          <span
            key={label}
            className="rounded bg-chip px-[7px] py-[3px] font-mono text-[10px] leading-none text-ink-3"
            style={label === "errors" ? { color: "var(--red)" } : undefined}
          >
            <span className="font-semibold text-ink-2">{n}</span> {label}
          </span>
        ))}
      </div>

      {/* step trace */}
      {w.steps.length > 0 && (
        <div className="mb-[12px]">
          <div className="mb-[7px] font-mono text-[9.5px] font-semibold uppercase tracking-[0.12em] text-ink-3">
            Trace
          </div>
          <ol className="flex flex-col gap-[6px]">
            {w.steps.slice(-8).map((s) => (
              <li key={s.event.event_id} className="flex gap-[8px] text-[11.5px] leading-[1.4]">
                <span className="mt-[5px] h-[5px] w-[5px] shrink-0 rounded-full bg-ink-4" />
                <div className="min-w-0">
                  <span className="text-ink-2">{s.label}</span>
                  {s.detail && (
                    <span className="ml-[6px] text-ink-3">— {truncate(s.detail, 80)}</span>
                  )}
                </div>
              </li>
            ))}
          </ol>
          {w.steps.length > 8 && (
            <div className="mt-[6px] pl-[13px] font-mono text-[10px] text-ink-4">
              +{w.steps.length - 8} earlier steps
            </div>
          )}
        </div>
      )}

      {/* latest code */}
      {w.currentCode && (
        <div>
          <div className="mb-[7px] font-mono text-[9.5px] font-semibold uppercase tracking-[0.12em] text-ink-3">
            Latest code
          </div>
          <pre className="pp-scroll max-h-[180px] overflow-auto rounded-[7px] border border-line bg-desk/60 p-[10px] font-mono text-[10.5px] leading-[1.5] text-ink-2">
            <code>{w.currentCode.length > 2400 ? w.currentCode.slice(0, 2400) + "\n…" : w.currentCode}</code>
          </pre>
        </div>
      )}

      {w.steps.length === 0 && !w.currentCode && (
        <div className="text-[11.5px] leading-relaxed text-ink-3">
          Spinning up — no steps recorded yet.
        </div>
      )}
    </div>
  );
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
