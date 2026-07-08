import { useMemo } from "react";
import type { BeliefObject, BeliefStatus, ConfidenceLevel } from "../../types";

// Beliefs as an *evolution* view (§D): each belief shows its status transition
// (active/strengthened/weakened/abandoned), supporting-vs-contradicting evidence
// as a tiny diverging bar, exhaustion_rounds, and confidence — ordered so the
// most-recently-*changed* beliefs surface first.

const CONF: Record<ConfidenceLevel, { color: string; label: string }> = {
  strong: { color: "var(--green)", label: "Strong" },
  weak: { color: "var(--red)", label: "Weak" },
  unclear: { color: "var(--text3)", label: "Unclear" },
};

const STATUS: Record<BeliefStatus, { text: string; color: string; rank: number }> = {
  strengthened: { text: "↑ strengthened", color: "var(--green)", rank: 0 },
  weakened: { text: "↓ weakened", color: "var(--red)", rank: 1 },
  abandoned: { text: "✕ abandoned", color: "var(--text4)", rank: 2 },
  active: { text: "• active", color: "var(--text3)", rank: 3 },
};

// A small two-sided diverging bar: contradicting (red) left of center, supporting
// (green) right of center, scaled to the largest evidence count on screen.
function DivergingBar({ support, against, max }: { support: number; against: number; max: number }) {
  const unit = max > 0 ? 50 / max : 0; // % per evidence node, each side capped at 50%
  return (
    <div className="relative h-[6px] w-full overflow-hidden rounded-full bg-chip">
      <span className="absolute inset-y-0 left-1/2 w-px bg-edge" aria-hidden />
      <span
        className="absolute inset-y-0 rounded-l-full"
        style={{ right: "50%", width: `${against * unit}%`, background: "var(--red)" }}
      />
      <span
        className="absolute inset-y-0 rounded-r-full"
        style={{ left: "50%", width: `${support * unit}%`, background: "var(--green)" }}
      />
    </div>
  );
}

export default function BeliefsView({ beliefs }: { beliefs: BeliefObject[] }) {
  const ordered = useMemo(() => {
    // Most-recently-changed proxy: changed statuses first (no timestamps exist in
    // the contract), then by total evidence magnitude.
    return beliefs
      .map((b, i) => ({ b, i }))
      .sort((a, b) => {
        const ra = STATUS[a.b.status]?.rank ?? 3;
        const rb = STATUS[b.b.status]?.rank ?? 3;
        if (ra !== rb) return ra - rb;
        const ea = a.b.supporting_nodes.length + a.b.contradicting_nodes.length;
        const eb = b.b.supporting_nodes.length + b.b.contradicting_nodes.length;
        return eb - ea;
      });
  }, [beliefs]);

  const maxEvidence = useMemo(
    () =>
      Math.max(
        1,
        ...beliefs.map((b) => Math.max(b.supporting_nodes.length, b.contradicting_nodes.length)),
      ),
    [beliefs],
  );

  if (!beliefs.length) {
    return (
      <div className="px-[18px] py-6 text-[12px] leading-relaxed text-ink-3">
        No active beliefs yet. They form once the campaign synthesizes its first results.
      </div>
    );
  }

  return (
    <div className="px-[14px] pb-4 pt-2">
      {ordered.map(({ b, i }) => {
        const conf = CONF[b.confidence] ?? CONF.unclear;
        const status = STATUS[b.status] ?? STATUS.active;
        const abandoned = b.status === "abandoned";
        return (
          <div
            key={i}
            className="mb-[8px] rounded-[10px] border border-edge bg-right px-[12px] py-[11px] last:mb-0"
          >
            <div className="mb-[9px] flex items-start gap-[9px]">
              <span className="mt-[1px] shrink-0 font-mono text-[11px] font-semibold leading-none text-ink-2">
                B{i + 1}
              </span>
              <span
                className="flex-1 text-[12.5px] leading-[1.4] text-ink"
                style={abandoned ? { color: "var(--text3)", textDecoration: "line-through" } : undefined}
              >
                {b.statement}
              </span>
              <span
                className="shrink-0 font-mono text-[9.5px] font-semibold leading-none"
                style={{ color: status.color }}
              >
                {status.text}
              </span>
            </div>

            <div className="flex items-center gap-[10px] pl-[24px]">
              <DivergingBar
                support={b.supporting_nodes.length}
                against={b.contradicting_nodes.length}
                max={maxEvidence}
              />
              <span
                className="shrink-0 font-mono text-[10px] font-semibold leading-none"
                style={{ color: conf.color }}
              >
                {conf.label}
              </span>
            </div>

            <div className="mt-[7px] flex items-center gap-[10px] pl-[24px] font-mono text-[10px] leading-none">
              <span style={{ color: "var(--green)" }}>+{b.supporting_nodes.length} support</span>
              <span style={{ color: "var(--red)" }}>−{b.contradicting_nodes.length} against</span>
              {b.exhaustion_rounds > 0 && (
                <span className="ml-auto text-ink-4">
                  {b.exhaustion_rounds} round{b.exhaustion_rounds === 1 ? "" : "s"} stable
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
