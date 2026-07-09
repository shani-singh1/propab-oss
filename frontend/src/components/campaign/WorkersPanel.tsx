import { useMemo, useState } from "react";
import type { Worker, WorkerStatus } from "../../lib/model";
import type { HypothesisTree } from "../../types";
import { useScrollRef, useVirtual } from "../../lib/useVirtual";
import { layoutGraph } from "../../lib/treeGraph";
import { verdictColor } from "../../lib/status";
import WorkerCard from "./WorkerCard";
import { StatusDot, workerAccent } from "./bits";

// A compact, non-interactive tree mini-map (redesign §5): nodes colored by
// verdict so the user sees the shape of the search at a glance, right beside the
// workers. The full zoom/pan tree lives in its own tab.
function MiniTree({ tree, question }: { tree: HypothesisTree | undefined; question: string }) {
  const layout = useMemo(
    () => layoutGraph(tree, question, { toggled: new Set() }),
    [tree, question],
  );
  const nodeCount = layout.nodes.length - 1; // minus synthetic root
  if (nodeCount <= 0) return null;
  const rFor = (depth: number, isRoot: boolean) => (isRoot ? 6 : depth === 1 ? 5 : 4);

  return (
    <div className="shrink-0 border-t border-line px-[12px] pb-[11px] pt-[9px]">
      <div className="mb-[7px] flex items-center gap-[8px]">
        <span className="font-mono text-[9.5px] font-semibold uppercase tracking-[0.12em] text-ink-3">
          Search tree
        </span>
        <span className="font-mono text-[9.5px] text-ink-4">
          {nodeCount} node{nodeCount === 1 ? "" : "s"}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${Math.max(1, layout.width)} ${Math.max(1, layout.height)}`}
        preserveAspectRatio="xMidYMid meet"
        width="100%"
        height={124}
        style={{ display: "block" }}
        aria-label="Hypothesis search tree, nodes colored by verdict"
      >
        {layout.edges.map((e) => (
          <line
            key={`${e.from}->${e.to}`}
            x1={e.x1}
            y1={e.y1}
            x2={e.x2}
            y2={e.y2}
            stroke="var(--divider)"
            strokeWidth={1.4}
          />
        ))}
        {layout.nodes.map((n) => {
          const stroke = n.isRoot ? "var(--text)" : verdictColor(n.verdict);
          const fill =
            n.verdict === "confirmed"
              ? "var(--greenDim)"
              : n.verdict === "refuted"
                ? "var(--redDim)"
                : n.isRoot
                  ? "var(--text)"
                  : "var(--rightBg)";
          return (
            <circle
              key={n.id}
              cx={n.x}
              cy={n.y}
              r={rFor(n.depth, n.isRoot)}
              fill={fill}
              stroke={stroke}
              strokeWidth={2}
              className={n.isFrontier ? "animate-ppulse motion-reduce:animate-none" : ""}
            />
          );
        })}
      </svg>
    </div>
  );
}

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

export default function WorkersPanel({
  workers,
  now,
  tree,
  question,
}: {
  workers: Worker[];
  now: number;
  tree: HypothesisTree | undefined;
  question: string;
}) {
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
    estimateHeight: 40,
    resetKey,
  });

  if (!workers.length) {
    return (
      <div className="flex h-full min-h-0 flex-col">
        <div className="flex-1 px-[16px] py-6 text-[12px] leading-relaxed text-ink-3">
          No sub-agents yet. Workers appear here once the orchestrator dispatches
          hypotheses for experimentation.
        </div>
        <MiniTree tree={tree} question={question} />
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
      <MiniTree tree={tree} question={question} />
    </div>
  );
}
