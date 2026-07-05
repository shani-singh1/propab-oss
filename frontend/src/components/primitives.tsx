import type { CSSProperties } from "react";

// A small status dot; color is a CSS-var string (e.g. var(--green)).
export function Dot({
  color,
  pulse = false,
  size = 8,
  className = "",
  style,
}: {
  color: string;
  pulse?: boolean;
  size?: number;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <span
      className={`inline-block shrink-0 rounded-full ${pulse ? "animate-ppulse" : ""} ${className}`}
      style={{ width: size, height: size, background: color, ...style }}
    />
  );
}

// A thin progress bar. `pct` is 0..1. Fill color is a CSS-var string.
export function Bar({
  pct,
  color = "var(--text3)",
  height = 4,
  className = "",
}: {
  pct: number;
  color?: string;
  height?: number;
  className?: string;
}) {
  return (
    <div
      className={`w-full overflow-hidden rounded-full bg-chip ${className}`}
      style={{ height }}
    >
      <div
        className="h-full rounded-full transition-[width] duration-500"
        style={{ width: `${Math.max(0, Math.min(1, pct)) * 100}%`, background: color }}
      />
    </div>
  );
}

// Small uppercase mono tag used for phase / conflict / belief-update chips.
export function Tag({
  children,
  color = "var(--text3)",
  bg = "transparent",
}: {
  children: React.ReactNode;
  color?: string;
  bg?: string;
}) {
  return (
    <span
      className="rounded font-mono text-[9.5px] font-semibold uppercase tracking-[0.06em]"
      style={{ color, background: bg, padding: bg === "transparent" ? 0 : "2px 6px" }}
    >
      {children}
    </span>
  );
}
