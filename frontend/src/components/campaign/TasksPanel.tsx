import { useState } from "react";
import type { InFlightTask, Worker } from "../../lib/model";
import { fmtElapsed } from "../../lib/format";
import { KindBadge, kindMeta } from "./bits";
import WorkerDetail from "./WorkerDetail";

// The transparency-into-the-machine view: everything running RIGHT NOW —
// worker experiments, sandbox code, LLM calls, tool calls — each openable.
function TaskRow({
  task,
  worker,
  now,
}: {
  task: InFlightTask;
  worker: Worker | undefined;
  now: number;
}) {
  const [open, setOpen] = useState(false);
  const expandable = task.kind === "experiment" && !!worker;
  const meta = kindMeta(task.kind);

  return (
    <div className="border-b border-line last:border-0">
      <button
        onClick={() => expandable && setOpen((o) => !o)}
        aria-expanded={expandable ? open : undefined}
        className={`flex w-full items-start gap-[10px] px-[16px] py-[11px] text-left ${expandable ? "pp-row" : "cursor-default"}`}
      >
        <KindBadge kind={task.kind} />
        <div className="min-w-0 flex-1">
          <div className="truncate text-[12.5px] font-medium leading-[1.35] text-ink">
            {task.title}
          </div>
          <div className="mt-[4px] flex items-center gap-[7px] font-mono text-[10px] leading-none text-ink-3">
            <span style={{ color: meta.color }}>{meta.label}</span>
            <span className="text-ink-4">·</span>
            <span className="animate-ppulse motion-reduce:animate-none">
              {fmtElapsed(task.startedAt, now)}
            </span>
            {task.detail && (
              <>
                <span className="text-ink-4">·</span>
                <span className="truncate text-ink-3">{task.detail}</span>
              </>
            )}
          </div>
        </div>
        {expandable && <span className="mt-[2px] shrink-0 text-[11px] text-ink-4">{open ? "▾" : "▸"}</span>}
      </button>
      {open && worker && <WorkerDetail w={worker} now={now} />}
    </div>
  );
}

export default function TasksPanel({
  tasks,
  workers,
  now,
  active,
}: {
  tasks: InFlightTask[];
  workers: Worker[];
  now: number;
  active: boolean;
}) {
  const byHyp = new Map(workers.map((w) => [w.hypothesisId, w]));

  if (!tasks.length) {
    return (
      <div className="px-[16px] py-6 text-[12px] leading-relaxed text-ink-3">
        {active
          ? "Nothing executing this instant. Background tasks appear here the moment a worker, sandbox run, or LLM call is in flight."
          : "No active tasks — the campaign has concluded."}
      </div>
    );
  }

  const groups: { kind: string; label: string; items: InFlightTask[] }[] = [
    { kind: "experiment", label: "Experiments", items: tasks.filter((t) => t.kind === "experiment") },
    { kind: "code", label: "Sandbox code", items: tasks.filter((t) => t.kind === "code") },
    { kind: "llm", label: "LLM calls", items: tasks.filter((t) => t.kind === "llm") },
    { kind: "tool", label: "Tool calls", items: tasks.filter((t) => t.kind === "tool") },
  ].filter((g) => g.items.length);

  return (
    <div>
      {groups.map((g) => (
        <div key={g.kind}>
          <div className="flex items-center gap-[8px] border-b border-line bg-chip/20 px-[16px] py-[7px]">
            <span className="font-mono text-[9.5px] font-semibold uppercase tracking-[0.12em] text-ink-3">
              {g.label}
            </span>
            <span className="font-mono text-[9.5px] text-ink-4">{g.items.length}</span>
          </div>
          {g.items.map((t) => (
            <TaskRow key={t.id} task={t} worker={t.hypothesisId ? byHyp.get(t.hypothesisId) : undefined} now={now} />
          ))}
        </div>
      ))}
    </div>
  );
}
