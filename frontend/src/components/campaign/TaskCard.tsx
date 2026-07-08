import type { InFlightTask, Worker } from "../../lib/model";
import { fmtElapsed } from "../../lib/format";
import { KindBadge, kindMeta, TimerPill } from "./bits";
import WorkerDetail from "./WorkerDetail";

// A live background-task card: kind badge, title, a breathing live-duration pill,
// a detail line, and — for worker experiments — an openable full detail view.
// The whole card carries a faint in-flight throb on its edge.
export default function TaskCard({
  task,
  worker,
  now,
  open,
  onToggle,
}: {
  task: InFlightTask;
  worker: Worker | undefined;
  now: number;
  open: boolean;
  onToggle: () => void;
}) {
  const meta = kindMeta(task.kind);
  const expandable = task.kind === "experiment" && !!worker;

  return (
    <div className="px-[12px] py-[6px]">
      <div className="overflow-hidden rounded-[11px] border border-edge bg-right">
        <div className="flex">
          <span
            aria-hidden
            className="w-[3px] shrink-0 animate-ppulse motion-reduce:animate-none"
            style={{ background: meta.color }}
          />
          <div className="min-w-0 flex-1">
            <button
              onClick={() => expandable && onToggle()}
              aria-expanded={expandable ? open : undefined}
              className={`flex w-full items-start gap-[10px] px-[12px] py-[11px] text-left ${expandable ? "pp-row" : "cursor-default"}`}
            >
              <KindBadge kind={task.kind} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-[8px]">
                  <span className="min-w-0 flex-1 truncate text-[12.5px] font-medium leading-[1.35] text-ink">
                    {task.title}
                  </span>
                  <TimerPill text={fmtElapsed(task.startedAt, now)} live />
                  {expandable && (
                    <span className="shrink-0 text-[11px] text-ink-4">{open ? "▾" : "▸"}</span>
                  )}
                </div>
                <div className="mt-[5px] flex items-center gap-[7px] font-mono text-[10px] leading-none text-ink-3">
                  <span style={{ color: meta.color }}>{meta.label}</span>
                  {task.shortId && (
                    <>
                      <span className="text-ink-4">·</span>
                      <span>{task.shortId}</span>
                    </>
                  )}
                  {task.detail && (
                    <>
                      <span className="text-ink-4">·</span>
                      <span className="truncate text-ink-3">{task.detail}</span>
                    </>
                  )}
                </div>
              </div>
            </button>
            {open && worker && <WorkerDetail w={worker} now={now} />}
          </div>
        </div>
      </div>
    </div>
  );
}
