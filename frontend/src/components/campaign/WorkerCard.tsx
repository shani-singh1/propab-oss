import type { Worker } from "../../lib/model";
import { fmtElapsed } from "../../lib/format";
import { StatusDot, workerAccent } from "./bits";
import WorkerDetail from "./WorkerDetail";

// A worker is a compact ONE-LINE row (redesign §5): status dot · hypothesis
// (truncated) · round · elapsed. Click to open the drawer with that worker's
// real experiment — the full trace, activity, latest code. All the worker's
// llm.*/tool.* chatter lives HERE in the rail, never in the center narrative.
export default function WorkerCard({
  w,
  now,
  open,
  onToggle,
}: {
  w: Worker;
  now: number;
  open: boolean;
  onToggle: () => void;
}) {
  const a = workerAccent(w.status);
  const running = w.status === "running";
  const elapsed = fmtElapsed(w.startedAt, running ? now : w.endedAt);

  return (
    <div className="px-[8px]">
      <button
        onClick={onToggle}
        aria-expanded={open}
        className="pp-row flex w-full items-center gap-[9px] rounded-[8px] px-[8px] py-[7px] text-left"
      >
        <StatusDot color={a.edge} live={a.breathe} size={6} />
        <span className="min-w-0 flex-1 truncate text-[12px] leading-[1.4] text-ink-2">
          {w.text || `Hypothesis ${w.shortId}`}
        </span>
        {!running && (
          <span
            className="shrink-0 font-mono text-[9.5px] uppercase tracking-[0.06em]"
            style={{ color: a.labelColor }}
          >
            {a.label}
          </span>
        )}
        {w.round != null && (
          <span className="shrink-0 font-mono text-[9.5px] text-ink-4">R{w.round}</span>
        )}
        <span className="shrink-0 font-mono text-[10px] tabular-nums text-ink-4">{elapsed}</span>
        <span className="shrink-0 text-[10px] text-ink-4">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="mb-[6px] mt-[2px] overflow-hidden rounded-[8px] border border-edge">
          <WorkerDetail w={w} now={now} />
        </div>
      )}
    </div>
  );
}
