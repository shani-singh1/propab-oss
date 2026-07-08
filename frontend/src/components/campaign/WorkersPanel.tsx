import { useMemo, useState } from "react";
import type { Worker, WorkerStatus } from "../../lib/model";
import { useScrollRef, useVirtual } from "../../lib/useVirtual";
import WorkerCard from "./WorkerCard";
import { StatusDot, workerAccent } from "./bits";

type Row =
  | { type: "header"; key: string; label: string; count: number; live?: boolean }
  | { type: "worker"; key: string; worker: Worker };

const STATUS_ORDER: WorkerStatus[] = ["running", "confirmed", "refuted", "inconclusive", "failed"];

// The sticky status-chip + free-text filter bar. Chips act as an inclusive
// multi-select; empty selection means "all".
function FilterBar({
  counts,
  active,
  onToggleStatus,
  query,
  onQuery,
  total,
  shown,
}: {
  counts: Record<WorkerStatus, number>;
  active: Set<WorkerStatus>;
  onToggleStatus: (s: WorkerStatus) => void;
  query: string;
  onQuery: (v: string) => void;
  total: number;
  shown: number;
}) {
  return (
    <div className="z-10 shrink-0 border-b border-line bg-right px-[12px] py-[9px]">
      <div className="flex items-center gap-[7px]">
        <input
          value={query}
          onChange={(e) => onQuery(e.target.value)}
          placeholder="Filter hypotheses…"
          className="min-w-0 flex-1 rounded-[7px] border border-edge bg-chip px-[9px] py-[6px] text-[11.5px] leading-none text-ink outline-none placeholder:text-ink-3 focus:border-ink-3"
        />
        {(query || active.size > 0) && (
          <span className="shrink-0 font-mono text-[10px] text-ink-3">
            {shown}/{total}
          </span>
        )}
      </div>
      <div className="pp-scroll mt-[8px] flex items-center gap-[6px] overflow-x-auto">
        {STATUS_ORDER.map((s) => {
          const n = counts[s] ?? 0;
          if (!n) return null;
          const on = active.has(s);
          const a = workerAccent(s);
          return (
            <button
              key={s}
              onClick={() => onToggleStatus(s)}
              aria-pressed={on}
              className="flex shrink-0 items-center gap-[5px] rounded-full border px-[9px] py-[4px] text-[10.5px] font-medium leading-none transition-colors"
              style={{
                borderColor: on ? a.edge : "var(--border)",
                background: on ? a.tint === "transparent" ? "var(--chip)" : a.tint : "transparent",
                color: on ? "var(--text)" : "var(--text3)",
              }}
            >
              <StatusDot color={a.edge} size={6} live={s === "running" && on} />
              {a.label}
              <span className="font-mono text-ink-3">{n}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default function WorkersPanel({ workers, now }: { workers: Worker[]; now: number }) {
  const [statusFilter, setStatusFilter] = useState<Set<WorkerStatus>>(new Set());
  const [query, setQuery] = useState("");
  const [openIds, setOpenIds] = useState<Set<string>>(new Set());
  const [scrollRef, getScroll] = useScrollRef<HTMLDivElement>();

  const counts = useMemo(() => {
    const c = { running: 0, confirmed: 0, refuted: 0, inconclusive: 0, failed: 0 } as Record<
      WorkerStatus,
      number
    >;
    for (const w of workers) c[w.status] += 1;
    return c;
  }, [workers]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return workers.filter((w) => {
      if (statusFilter.size > 0 && !statusFilter.has(w.status)) return false;
      if (q && !(w.text || w.shortId).toLowerCase().includes(q)) return false;
      return true;
    });
  }, [workers, statusFilter, query]);

  // Flatten into rows with Running / Finished section headers (workers already
  // arrive running-first, then most-recent). Headers virtualize alongside cards.
  const rows = useMemo<Row[]>(() => {
    const out: Row[] = [];
    const running = filtered.filter((w) => w.status === "running");
    const finished = filtered.filter((w) => w.status !== "running");
    if (running.length) {
      out.push({ type: "header", key: "h-running", label: "Running", count: running.length, live: true });
      for (const w of running) out.push({ type: "worker", key: w.hypothesisId, worker: w });
    }
    if (finished.length) {
      out.push({ type: "header", key: "h-finished", label: "Finished", count: finished.length });
      for (const w of finished) out.push({ type: "worker", key: w.hypothesisId, worker: w });
    }
    return out;
  }, [filtered]);

  const resetKey = `${statusFilter.size}:${query}:${rows.length}`;
  const { virtualItems, totalSize } = useVirtual({
    count: rows.length,
    getScrollElement: getScroll,
    estimateHeight: 150,
    resetKey,
  });

  if (!workers.length) {
    return (
      <div className="px-[16px] py-6 text-[12px] leading-relaxed text-ink-3">
        No sub-agents yet. Workers appear here once the orchestrator dispatches
        hypotheses for experimentation.
      </div>
    );
  }

  const toggleStatus = (s: WorkerStatus) =>
    setStatusFilter((prev) => {
      const next = new Set(prev);
      next.has(s) ? next.delete(s) : next.add(s);
      return next;
    });
  const toggleOpen = (id: string) =>
    setOpenIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  return (
    <div className="flex h-full min-h-0 flex-col">
      <FilterBar
        counts={counts}
        active={statusFilter}
        onToggleStatus={toggleStatus}
        query={query}
        onQuery={setQuery}
        total={workers.length}
        shown={filtered.length}
      />
      <div ref={scrollRef} className="pp-scroll relative min-h-0 flex-1 overflow-auto">
        {filtered.length === 0 ? (
          <div className="px-[16px] py-6 text-[12px] leading-relaxed text-ink-3">
            No workers match this filter.
          </div>
        ) : (
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
                    <div className="flex items-center gap-[8px] px-[16px] pb-[5px] pt-[12px]">
                      {row.live ? (
                        <StatusDot color="var(--text)" live size={7} />
                      ) : (
                        <span className="h-[7px] w-[7px]" />
                      )}
                      <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-3">
                        {row.label}
                      </span>
                      <span className="font-mono text-[10px] text-ink-4">{row.count}</span>
                    </div>
                  ) : (
                    <WorkerCard
                      w={row.worker}
                      now={now}
                      open={openIds.has(row.worker.hypothesisId)}
                      onToggle={() => toggleOpen(row.worker.hypothesisId)}
                    />
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
