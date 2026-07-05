import type { BeliefObject, ConfidenceLevel } from "../../types";
import { Bar } from "../primitives";

// The backend tracks categorical confidence (strong/weak/unclear) plus
// supporting/contradicting node lists — not numeric probabilities. We render
// exactly that, honestly, in the reference's belief-list style.
const CONF: Record<ConfidenceLevel, { pct: number; color: string; label: string }> = {
  strong: { pct: 1, color: "var(--green)", label: "Strong" },
  weak: { pct: 0.34, color: "var(--red)", label: "Weak" },
  unclear: { pct: 0.52, color: "var(--text3)", label: "Unclear" },
};

function statusChip(status: string): { text: string; color: string } | null {
  switch (status) {
    case "strengthened":
      return { text: "↑ strengthened", color: "var(--green)" };
    case "weakened":
      return { text: "↓ weakened", color: "var(--red)" };
    case "abandoned":
      return { text: "abandoned", color: "var(--text4)" };
    default:
      return null;
  }
}

export default function BeliefsView({ beliefs }: { beliefs: BeliefObject[] }) {
  if (!beliefs.length) {
    return (
      <div className="px-[18px] py-6 text-[12px] leading-relaxed text-ink-3">
        No active beliefs yet. They form once the campaign synthesizes its first results.
      </div>
    );
  }
  return (
    <div className="px-[18px] pb-4 pt-1.5">
      {beliefs.map((b, i) => {
        const conf = CONF[b.confidence] ?? CONF.unclear;
        const chip = statusChip(b.status);
        const abandoned = b.status === "abandoned";
        return (
          <div key={i} className="border-b border-line py-[13px] last:border-0">
            <div className="mb-[9px] flex items-baseline gap-[9px]">
              <span className="shrink-0 font-mono text-[11.5px] font-semibold leading-none text-ink-2">
                B{i + 1}
              </span>
              <span
                className="text-[12.5px] leading-[1.4] text-ink"
                style={abandoned ? { color: "var(--text3)", textDecoration: "line-through" } : undefined}
              >
                {b.statement}
              </span>
            </div>
            <div className="flex items-center gap-[10px] pl-[22px]">
              <Bar pct={conf.pct} color={conf.color} height={6} className="flex-1" />
              <span className="font-mono text-[11px] font-semibold leading-none" style={{ color: conf.color }}>
                {conf.label}
              </span>
            </div>
            <div className="mt-[7px] flex items-center gap-[10px] pl-[22px] font-mono text-[10px] leading-none text-ink-3">
              <span style={{ color: "var(--green)" }}>+{b.supporting_nodes.length} support</span>
              <span style={{ color: "var(--red)" }}>−{b.contradicting_nodes.length} against</span>
              {chip && (
                <span className="ml-auto" style={{ color: chip.color }}>
                  {chip.text}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
