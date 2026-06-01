import type { ReactNode } from "react";

export function StatusDot({ status }: { status: string }) {
  const s = (status || "").toLowerCase();
  const color =
    s === "active" || s === "running"
      ? "bg-running"
      : s === "completed" || s === "breakthrough"
        ? "bg-confirmed"
        : s === "failed"
          ? "bg-refuted"
          : s === "budget_exhausted"
            ? "bg-warning"
            : "bg-inconclusive";
  const pulse = s === "active" || s === "running" ? "animate-pulseSoft" : "";
  return (
    <span className="relative inline-flex h-2.5 w-2.5">
      <span className={`inline-flex h-2.5 w-2.5 rounded-full ${color} ${pulse}`} />
    </span>
  );
}

export function Badge({
  children,
  tone = "neutral",
  className = "",
}: {
  children: ReactNode;
  tone?: "neutral" | "confirmed" | "refuted" | "running" | "warning" | "brand";
  className?: string;
}) {
  const tones: Record<string, string> = {
    neutral: "bg-raised text-text-secondary border-border",
    confirmed: "bg-confirmed/10 text-confirmed border-confirmed/30",
    refuted: "bg-refuted/10 text-refuted border-refuted/30",
    running: "bg-running/10 text-running border-running/30",
    warning: "bg-warning/10 text-warning border-warning/30",
    brand: "bg-brand/15 text-brand border-brand/30",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-xl border border-border bg-surface ${className}`}>{children}</div>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="flex flex-col">
      <span className="text-[11px] uppercase tracking-wide text-text-muted">{label}</span>
      <span className="text-base font-semibold text-text-primary tabular-nums">{value}</span>
      {sub != null && <span className="text-xs text-text-secondary">{sub}</span>}
    </div>
  );
}

export function Spinner() {
  return (
    <div className="flex items-center gap-2 text-text-secondary">
      <span className="h-3 w-3 animate-pulseSoft rounded-full bg-brand" />
      <span className="text-sm">Loading…</span>
    </div>
  );
}

export function ProgressBar({ pct, tone = "brand" }: { pct: number; tone?: "brand" | "warning" }) {
  const color = tone === "warning" ? "bg-warning" : "bg-brand";
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-raised">
      <div className={`h-full ${color} transition-all duration-500`} style={{ width: `${pct}%` }} />
    </div>
  );
}
