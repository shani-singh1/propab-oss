import { useMemo } from "react";
import type { PropabEvent, CampaignSummary } from "../../types";
import { computeStats, type ComputeBreakdownRow } from "../../lib/model";
import { fmtDuration } from "../../lib/format";

// Compute / cost tab (§D): LLM / tool / code / error volume, a per-purpose and
// per-tool breakdown, the compute-budget burn-down, and — once the backend emits
// them (design.md §3.1–3.2) — per-call latency percentiles and token totals read
// from `llm.response` `duration_ms` / `tokens_in` / `tokens_out`.

// Compact millisecond duration, e.g. "840ms", "3.2s".
function fmtMs(ms: number | null): string {
  if (ms == null || !isFinite(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)}s`;
}

// Compact token count, e.g. "812", "12.4k", "1.2M".
function fmtTokens(n: number): string {
  if (!isFinite(n)) return "—";
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}

function StatTile({ label, value, color }: { label: string; value: number | string; color?: string }) {
  return (
    <div className="rounded-[9px] border border-edge bg-right px-[11px] py-[10px]">
      <div className="font-mono text-[18px] font-semibold leading-none" style={{ color: color ?? "var(--text)" }}>
        {value}
      </div>
      <div className="mt-[6px] font-mono text-[9.5px] uppercase tracking-[0.1em] text-ink-3">
        {label}
      </div>
    </div>
  );
}

function Breakdown({ title, rows, total }: { title: string; rows: ComputeBreakdownRow[]; total: number }) {
  if (!rows.length) return null;
  const max = Math.max(1, ...rows.map((r) => r.count));
  return (
    <div className="mb-[16px]">
      <div className="mb-[8px] flex items-baseline gap-[8px]">
        <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-3">
          {title}
        </span>
        <span className="font-mono text-[10px] text-ink-4">{total}</span>
      </div>
      <div className="flex flex-col gap-[7px]">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-[9px]">
            <span className="w-[92px] shrink-0 truncate text-[11px] text-ink-2">{r.label}</span>
            <div className="h-[6px] flex-1 overflow-hidden rounded-full bg-chip">
              <div
                className="h-full rounded-full"
                style={{ width: `${(r.count / max) * 100}%`, background: "var(--text3)" }}
              />
            </div>
            <span className="w-[26px] shrink-0 text-right font-mono text-[10px] text-ink-3">
              {r.count}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ComputePanel({
  events,
  summary,
}: {
  events: PropabEvent[];
  summary: CampaignSummary | undefined;
}) {
  const stats = useMemo(() => computeStats(events, summary), [events, summary]);
  const b = stats.budget;

  return (
    <div className="pp-scroll h-full overflow-auto px-[14px] pb-6 pt-[14px]">
      <div className="mb-[16px] grid grid-cols-4 gap-[7px]">
        <StatTile label="LLM" value={stats.llm} />
        <StatTile label="Tools" value={stats.tool} />
        <StatTile label="Code" value={stats.code} />
        <StatTile label="Errors" value={stats.errors} color={stats.errors ? "var(--red)" : undefined} />
      </div>

      {/* budget burn-down */}
      {b.hasBudget && (
        <div className="mb-[18px]">
          <div className="mb-[8px] flex items-baseline justify-between">
            <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-3">
              Compute budget
            </span>
            <span className="font-mono text-[10px] text-ink-3">
              {fmtDuration(b.usedSec)} / {fmtDuration(b.totalSec)}
            </span>
          </div>
          <div className="h-[8px] w-full overflow-hidden rounded-full bg-chip">
            <div
              className="h-full rounded-full transition-[width] duration-500"
              style={{
                width: `${b.pct * 100}%`,
                background: b.pct > 0.85 ? "var(--red)" : "var(--text)",
              }}
            />
          </div>
          <div className="mt-[6px] font-mono text-[10px] text-ink-4">
            {fmtDuration(b.remainingSec)} remaining · {Math.round(b.pct * 100)}% spent
          </div>
        </div>
      )}

      {/* per-call latency (design.md §3.1) — once llm.response carries duration_ms */}
      {stats.hasLatency && (
        <div className="mb-[16px]">
          <div className="mb-[8px] flex items-baseline gap-[8px]">
            <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-3">
              LLM latency
            </span>
            <span className="font-mono text-[10px] text-ink-4">{stats.latency.count} calls</span>
          </div>
          <div className="grid grid-cols-3 gap-[7px]">
            <StatTile label="p50" value={fmtMs(stats.latency.p50Ms)} />
            <StatTile label="p95" value={fmtMs(stats.latency.p95Ms)} />
            <StatTile label="max" value={fmtMs(stats.latency.maxMs)} />
          </div>
        </div>
      )}

      {/* token totals (design.md §3.2) — once llm.response carries tokens_in/out */}
      {stats.hasTokens && (
        <div className="mb-[16px]">
          <div className="mb-[8px] flex items-baseline gap-[8px]">
            <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-3">
              Tokens
            </span>
            <span className="font-mono text-[10px] text-ink-4">{stats.tokens.count} calls</span>
          </div>
          <div className="grid grid-cols-3 gap-[7px]">
            <StatTile label="in" value={fmtTokens(stats.tokens.in)} />
            <StatTile label="out" value={fmtTokens(stats.tokens.out)} />
            <StatTile label="total" value={fmtTokens(stats.tokens.total)} />
          </div>
        </div>
      )}

      <Breakdown title="LLM by purpose" rows={stats.llmByPurpose} total={stats.llm} />
      <Breakdown title="Tools by name" rows={stats.toolByName} total={stats.tool} />
      {stats.errors > 0 && (
        <Breakdown title="Errors by type" rows={stats.errorsByType} total={stats.errors} />
      )}

      {/* honest gap only for what remains un-emitted */}
      {(!stats.hasLatency || !stats.hasTokens) && (
        <div className="mt-[6px] rounded-[9px] border border-dashed border-edge px-[11px] py-[10px]">
          <div className="font-mono text-[9.5px] uppercase tracking-[0.1em] text-ink-3">
            {!stats.hasLatency && !stats.hasTokens ? "Latency · tokens" : !stats.hasLatency ? "Latency" : "Tokens"}
          </div>
          <div className="mt-[5px] text-[11px] leading-[1.5] text-ink-3">
            Awaiting per-call{" "}
            {!stats.hasLatency && <span className="text-ink-2">duration_ms</span>}
            {!stats.hasLatency && !stats.hasTokens && " / "}
            {!stats.hasTokens && <span className="text-ink-2">tokens_in/out</span>} on{" "}
            <span className="text-ink-2">llm.response</span>. They render here the moment the
            backend emits them.
          </div>
        </div>
      )}
    </div>
  );
}
