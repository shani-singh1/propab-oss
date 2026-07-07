import type { InFlightKind, WorkerStatus } from "../../lib/model";

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
