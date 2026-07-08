import type { CSSProperties } from "react";
import type { InFlightKind, Worker, WorkerStatus, WorkerStep } from "../../lib/model";

// Small shared presentational atoms for the workers / tasks / narrative views.

export function WorkerStatusMeta(status: WorkerStatus): { label: string; color: string; live: boolean } {
  switch (status) {
    case "running":
      return { label: "Running", color: "var(--text)", live: true };
    case "confirmed":
      return { label: "Confirmed", color: "var(--green)", live: false };
    case "refuted":
      return { label: "Refuted", color: "var(--red)", live: false };
    case "inconclusive":
      return { label: "Inconclusive", color: "var(--text3)", live: false };
    case "failed":
      return { label: "Failed", color: "var(--red)", live: false };
    default:
      return { label: "Unknown", color: "var(--text3)", live: false };
  }
}

// A colored, optionally pulsing status dot.
export function StatusDot({
  color,
  live = false,
  size = 8,
}: {
  color: string;
  live?: boolean;
  size?: number;
}) {
  return (
    <span
      className={`inline-block shrink-0 rounded-full ${live ? "animate-ppulse motion-reduce:animate-none" : ""}`}
      style={{ width: size, height: size, background: color }}
    />
  );
}

const KIND_META: Record<InFlightKind, { glyph: string; label: string; color: string }> = {
  experiment: { glyph: "⬡", label: "Experiment", color: "var(--text)" },
  code: { glyph: "{ }", label: "Sandbox", color: "var(--text2)" },
  llm: { glyph: "◇", label: "LLM", color: "var(--text2)" },
  tool: { glyph: "⚙", label: "Tool", color: "var(--text2)" },
};

export function kindMeta(kind: InFlightKind) {
  return KIND_META[kind];
}

// A monospace glyph badge for a background-task kind.
export function KindBadge({ kind }: { kind: InFlightKind }) {
  const m = KIND_META[kind];
  return (
    <span
      className="flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-[6px] bg-chip font-mono text-[10px] font-semibold"
      style={{ color: m.color }}
      aria-hidden
    >
      {m.glyph}
    </span>
  );
}

// ── Worker-card treatment ────────────────────────────────────────────────────
// A status maps to a *calm* card accent: a left edge + faint surface tint. Running
// breathes (a soft edge pulse, motion-reduce safe); confirmed carries a green edge;
// refuted is treated as information-gained (muted, never a red alarm); failed is
// the only genuinely muted-negative state.

export interface WorkerAccent {
  /** left-edge accent color */
  edge: string;
  /** faint card surface tint (already alpha-baked or transparent) */
  tint: string;
  /** the accent breathes while the worker is live */
  breathe: boolean;
  label: string;
  labelColor: string;
}

export function workerAccent(status: WorkerStatus): WorkerAccent {
  switch (status) {
    case "running":
      return { edge: "var(--text)", tint: "transparent", breathe: true, label: "Running", labelColor: "var(--text2)" };
    case "confirmed":
      return { edge: "var(--green)", tint: "var(--greenDim)", breathe: false, label: "Confirmed", labelColor: "var(--green)" };
    case "refuted":
      // Information gained — a closed branch. Muted, not alarm.
      return { edge: "var(--text3)", tint: "transparent", breathe: false, label: "Refuted", labelColor: "var(--text3)" };
    case "inconclusive":
      return { edge: "var(--text4)", tint: "transparent", breathe: false, label: "Inconclusive", labelColor: "var(--text3)" };
    case "failed":
      return { edge: "var(--red)", tint: "var(--redDim)", breathe: false, label: "Failed", labelColor: "var(--red)" };
    default:
      return { edge: "var(--text4)", tint: "transparent", breathe: false, label: "Unknown", labelColor: "var(--text3)" };
  }
}

// A small labelled activity chip (e.g. "12 LLM"). `tone` recolors for errors.
export function StatChip({
  n,
  label,
  tone = "default",
  title,
}: {
  n: number;
  label: string;
  tone?: "default" | "error";
  title?: string;
}) {
  const error = tone === "error" && n > 0;
  return (
    <span
      title={title}
      className="inline-flex items-center gap-[4px] rounded-[5px] bg-chip px-[6px] py-[3px] font-mono text-[10px] leading-none"
      style={error ? { color: "var(--red)" } : undefined}
    >
      <span className="font-semibold" style={error ? undefined : { color: "var(--text2)" }}>
        {n}
      </span>
      <span style={error ? undefined : { color: "var(--text3)" }}>{label}</span>
    </span>
  );
}

const STEP_KIND_COLOR: Record<WorkerStep["kind"], string> = {
  tool: "var(--text2)",
  code: "var(--text)",
  think: "var(--text3)",
  stop: "var(--green)",
  other: "var(--text4)",
};

// A tiny per-step sparkline: one thin bar per step, colored by step kind, tallest
// for code/tool activity. Reads at a glance as "how much has this worker done."
export function StepSparkline({ steps, live }: { steps: WorkerStep[]; live?: boolean }) {
  if (!steps.length) return null;
  const recent = steps.slice(-28);
  const w = 3;
  const gap = 2;
  const width = recent.length * (w + gap);
  const height = 16;
  const heightFor = (k: WorkerStep["kind"]) =>
    k === "code" ? 15 : k === "tool" ? 12 : k === "stop" ? 9 : k === "think" ? 7 : 5;
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="shrink-0"
      aria-hidden
      style={{ display: "block" }}
    >
      {recent.map((s, i) => {
        const h = heightFor(s.kind);
        const last = i === recent.length - 1;
        return (
          <rect
            key={s.event.event_id}
            x={i * (w + gap)}
            y={height - h}
            width={w}
            height={h}
            rx={1.2}
            fill={STEP_KIND_COLOR[s.kind]}
            className={last && live ? "animate-ppulse motion-reduce:animate-none" : ""}
          />
        );
      })}
    </svg>
  );
}

// The compact activity strip shown on every worker card.
export function ActivityStrip({ w }: { w: Worker }) {
  return (
    <div className="flex flex-wrap items-center gap-[5px]">
      <StepSparkline steps={w.steps} live={w.status === "running"} />
      <StatChip n={w.llmCalls} label="LLM" title="LLM calls" />
      <StatChip n={w.toolCalls} label="tools" title="Tool calls" />
      <StatChip n={w.codeRuns} label="runs" title="Sandbox code runs" />
      {w.errors > 0 && <StatChip n={w.errors} label="err" tone="error" title="Errors" />}
    </div>
  );
}

// A pill for the running elapsed timer that softly breathes while live.
export function TimerPill({ text, live, style }: { text: string; live?: boolean; style?: CSSProperties }) {
  return (
    <span
      className={`shrink-0 rounded-[5px] bg-chip px-[6px] py-[3px] font-mono text-[10px] leading-none text-ink-2 ${live ? "animate-ppulse motion-reduce:animate-none" : ""}`}
      style={style}
    >
      {text}
    </span>
  );
}
