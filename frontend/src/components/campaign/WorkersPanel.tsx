import { useState } from "react";
import type { Worker } from "../../lib/model";
import { fmtElapsed } from "../../lib/format";
import { StatusDot, WorkerStatusMeta } from "./bits";
import WorkerDetail from "./WorkerDetail";

function WorkerRow({ w, now }: { w: Worker; now: number }) {
  const [open, setOpen] = useState(false);
  const meta = WorkerStatusMeta(w.status);
  const elapsed = fmtElapsed(w.startedAt, w.status === "running" ? now : w.endedAt);
  const lastStep = w.steps[w.steps.length - 1];

  return (
    <div className="border-b border-line last:border-0">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="pp-row flex w-full items-start gap-[10px] px-[16px] py-[11px] text-left"
      >
        <span className="mt-[3px]">
          <StatusDot color={meta.color} live={meta.live} size={8} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-[8px]">
            <span className="truncate text-[12.5px] font-medium leading-[1.35] text-ink">
              {w.text || `Hypothesis ${w.shortId}`}
            </span>
          </div>
          <div className="mt-[4px] flex items-center gap-[8px] font-mono text-[10px] leading-none text-ink-3">
            <span style={{ color: meta.color }}>{meta.label}</span>
            <span className="text-ink-4">·</span>
            <span>{elapsed}</span>
            {w.status === "running" && lastStep && (
              <>
                <span className="text-ink-4">·</span>
                <span className="truncate text-ink-3">{lastStep.label}</span>
              </>
            )}
            {w.confidence != null && w.status !== "running" && (
              <>
                <span className="text-ink-4">·</span>
                <span>{w.confidence.toFixed(2)}</span>
              </>
            )}
          </div>
        </div>
        <span className="mt-[2px] shrink-0 text-[11px] text-ink-4">{open ? "▾" : "▸"}</span>
      </button>
      {open && <WorkerDetail w={w} now={now} />}
    </div>
  );
}

export default function WorkersPanel({ workers, now }: { workers: Worker[]; now: number }) {
  const [limit, setLimit] = useState(50);

  if (!workers.length) {
    return (
      <div className="px-[16px] py-6 text-[12px] leading-relaxed text-ink-3">
        No sub-agents yet. Workers appear here once the orchestrator dispatches
        hypotheses for experimentation.
      </div>
    );
  }

  const running = workers.filter((w) => w.status === "running");
  const shown = workers.slice(0, limit);

  return (
    <div>
      {running.length > 0 && (
        <div className="flex items-center gap-[8px] border-b border-line px-[16px] py-[9px]">
          <StatusDot color="var(--text)" live size={7} />
          <span className="font-mono text-[10.5px] font-semibold leading-none text-ink-2">
            {running.length} running
          </span>
          <span className="font-mono text-[10.5px] leading-none text-ink-4">
            · {workers.length - running.length} finished
          </span>
        </div>
      )}
      {shown.map((w) => (
        <WorkerRow key={w.hypothesisId} w={w} now={now} />
      ))}
      {workers.length > limit && (
        <button
          onClick={() => setLimit((l) => l + 100)}
          className="w-full px-[16px] py-[11px] text-left font-mono text-[11px] text-ink-3 hover:text-ink"
        >
          Show {Math.min(100, workers.length - limit)} more of {workers.length}…
        </button>
      )}
    </div>
  );
}
