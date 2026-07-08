import { useMemo, useState } from "react";
import type { InFlightKind, InFlightTask, Worker } from "../../lib/model";
import { useScrollRef, useVirtual } from "../../lib/useVirtual";
import TaskCard from "./TaskCard";

type Row =
  | { type: "header"; key: string; label: string; count: number }
  | { type: "task"; key: string; task: InFlightTask };

const GROUPS: { kind: InFlightKind; label: string }[] = [
  { kind: "experiment", label: "Experiments" },
  { kind: "code", label: "Sandbox code" },
  { kind: "llm", label: "LLM calls" },
  { kind: "tool", label: "Tool calls" },
];

// The "watch the machine think" tab: everything running RIGHT NOW as live cards,
// grouped by kind, virtualized so a burst of hundreds of in-flight calls stays
// smooth. An empty state that reads as "idle, healthy," not "nothing here."
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
  const [openIds, setOpenIds] = useState<Set<string>>(new Set());
  const [scrollRef, getScroll] = useScrollRef<HTMLDivElement>();
  const byHyp = useMemo(() => new Map(workers.map((w) => [w.hypothesisId, w])), [workers]);

  const rows = useMemo<Row[]>(() => {
    const out: Row[] = [];
    for (const g of GROUPS) {
      const items = tasks.filter((t) => t.kind === g.kind);
      if (!items.length) continue;
      out.push({ type: "header", key: `h-${g.kind}`, label: g.label, count: items.length });
      for (const t of items) out.push({ type: "task", key: t.id, task: t });
    }
    return out;
  }, [tasks]);

  const { virtualItems, totalSize } = useVirtual({
    count: rows.length,
    getScrollElement: getScroll,
    estimateHeight: 82,
    resetKey: rows.length,
  });

  if (!tasks.length) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-[10px] px-[24px] py-10 text-center">
        <span
          className={`h-[10px] w-[10px] rounded-full ${active ? "animate-ppulse motion-reduce:animate-none" : ""}`}
          style={{ background: active ? "var(--green)" : "var(--text4)" }}
        />
        <div className="text-[12px] leading-relaxed text-ink-3">
          {active
            ? "Idle and healthy — no work in flight this instant. Experiments, sandbox runs, and LLM calls surface here the moment they start."
            : "No active tasks — the campaign has concluded."}
        </div>
      </div>
    );
  }

  const toggleOpen = (id: string) =>
    setOpenIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  return (
    <div ref={scrollRef} className="pp-scroll relative h-full min-h-0 overflow-auto">
      <div style={{ height: totalSize, position: "relative" }}>
        {virtualItems.map((vi) => {
          const row = rows[vi.index];
          return (
            <div
              key={row.key}
              ref={vi.measureRef}
              style={{ position: "absolute", top: vi.start, left: 0, right: 0 }}
            >
              {row.type === "header" ? (
                <div className="flex items-center gap-[8px] px-[16px] pb-[4px] pt-[12px]">
                  <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-3">
                    {row.label}
                  </span>
                  <span className="font-mono text-[10px] text-ink-4">{row.count}</span>
                </div>
              ) : (
                <TaskCard
                  task={row.task}
                  worker={row.task.hypothesisId ? byHyp.get(row.task.hypothesisId) : undefined}
                  now={now}
                  open={openIds.has(row.task.id)}
                  onToggle={() => toggleOpen(row.task.id)}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
