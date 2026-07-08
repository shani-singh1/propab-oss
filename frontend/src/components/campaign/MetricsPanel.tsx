import { useMemo, useState } from "react";
import type { RoundView } from "../../lib/model";
import type { CampaignSummary } from "../../types";
import { metricTrajectory } from "../../lib/model";
import { fmtMetric } from "../../lib/format";

// Metrics tab (§D): the best-metric trajectory over rounds vs. the baseline and
// the breakthrough threshold, as a small step chart. Follows the dataviz skill's
// conventions — one green data series (identity by the single hue; the title
// names it, so no legend box is required), recessive baseline/threshold reference
// lines with direct labels, thin marks, 8px markers, a per-point hover tooltip,
// and text in ink tokens (never the series color).

const VW = 300;
const VH = 176;
const PAD = { top: 16, right: 16, bottom: 26, left: 40 };

export default function MetricsPanel({
  rounds,
  summary,
  direction,
}: {
  rounds: RoundView[];
  summary: CampaignSummary | undefined;
  direction: "higher_is_better" | "lower_is_better";
}) {
  const traj = useMemo(
    () => metricTrajectory(rounds, summary, direction),
    [rounds, summary, direction],
  );
  const [hover, setHover] = useState<number | null>(null);

  if (!traj.hasData) {
    return (
      <div className="px-[16px] py-6 text-[12px] leading-relaxed text-ink-3">
        No metric trajectory yet. Once a baseline is measured and rounds report a
        best objective value, the best-so-far curve renders here against the
        baseline and the breakthrough threshold.
      </div>
    );
  }

  const plotW = VW - PAD.left - PAD.right;
  const plotH = VH - PAD.top - PAD.bottom;

  const ys = [
    ...traj.points.map((p) => p.best),
    traj.baseline ?? 0,
    ...(traj.thresholdValue != null ? [traj.thresholdValue] : []),
  ];
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const yPad = (yMax - yMin) * 0.12;
  yMin -= yPad;
  yMax += yPad;

  const n = traj.points.length;
  const xFor = (i: number) => PAD.left + (n <= 1 ? plotW / 2 : (plotW * i) / (n - 1));
  const yFor = (v: number) => PAD.top + plotH * (1 - (v - yMin) / (yMax - yMin));

  // Step path: horizontal carry from previous round then step to new best.
  const stepPath = traj.points
    .map((p, i) => {
      const x = xFor(i);
      const y = yFor(p.best);
      if (i === 0) return `M ${x} ${y}`;
      const prevY = yFor(traj.points[i - 1].best);
      return `L ${x} ${prevY} L ${x} ${y}`;
    })
    .join(" ");

  const baselineY = traj.baseline != null ? yFor(traj.baseline) : null;
  const thresholdY = traj.thresholdValue != null ? yFor(traj.thresholdValue) : null;
  const last = traj.points[n - 1];
  const crossed =
    traj.thresholdValue != null &&
    (direction === "higher_is_better"
      ? last.best >= traj.thresholdValue
      : last.best <= traj.thresholdValue);

  return (
    <div className="px-[14px] pb-6 pt-[14px]">
      <div className="mb-[10px] flex items-baseline gap-[8px]">
        <span className="text-[12.5px] font-semibold text-ink">Best metric over rounds</span>
        <span className="font-mono text-[10px] text-ink-3">
          {direction === "higher_is_better" ? "higher is better" : "lower is better"}
        </span>
      </div>

      <div className="relative">
        <svg viewBox={`0 0 ${VW} ${VH}`} width="100%" style={{ display: "block", overflow: "visible" }}>
          {/* recessive gridlines */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = PAD.top + plotH * t;
            return (
              <line
                key={t}
                x1={PAD.left}
                y1={y}
                x2={VW - PAD.right}
                y2={y}
                stroke="var(--divider)"
                strokeWidth={1}
              />
            );
          })}
          {/* y-axis min/max labels */}
          <text x={PAD.left - 6} y={yFor(yMax) + 3} textAnchor="end" fontSize={8.5} fontFamily="'JetBrains Mono', monospace" fill="var(--text3)">
            {fmtMetric(yMax)}
          </text>
          <text x={PAD.left - 6} y={yFor(yMin) + 3} textAnchor="end" fontSize={8.5} fontFamily="'JetBrains Mono', monospace" fill="var(--text3)">
            {fmtMetric(yMin)}
          </text>

          {/* baseline reference */}
          {baselineY != null && (
            <>
              <line x1={PAD.left} y1={baselineY} x2={VW - PAD.right} y2={baselineY} stroke="var(--text3)" strokeWidth={1.5} strokeDasharray="4 3" />
              <text x={VW - PAD.right} y={baselineY - 4} textAnchor="end" fontSize={8.5} fontFamily="'JetBrains Mono', monospace" fill="var(--text3)">
                baseline
              </text>
            </>
          )}
          {/* threshold reference */}
          {thresholdY != null && (
            <>
              <line x1={PAD.left} y1={thresholdY} x2={VW - PAD.right} y2={thresholdY} stroke={crossed ? "var(--green)" : "var(--text2)"} strokeWidth={1.5} strokeDasharray="2 3" />
              <text x={VW - PAD.right} y={thresholdY - 4} textAnchor="end" fontSize={8.5} fontFamily="'JetBrains Mono', monospace" fill={crossed ? "var(--green)" : "var(--text2)"}>
                breakthrough
              </text>
            </>
          )}

          {/* the best-metric step line */}
          <path d={stepPath} fill="none" stroke="var(--green)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />

          {/* markers + hit targets */}
          {traj.points.map((p, i) => (
            <g key={p.round} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <circle cx={xFor(i)} cy={yFor(p.best)} r={hover === i ? 5 : 4} fill="var(--green)" stroke="var(--rightBg)" strokeWidth={1.5} />
              <circle cx={xFor(i)} cy={yFor(p.best)} r={10} fill="transparent" />
              <text x={xFor(i)} y={VH - 8} textAnchor="middle" fontSize={8.5} fontFamily="'JetBrains Mono', monospace" fill="var(--text3)">
                R{p.round}
              </text>
            </g>
          ))}

          {/* direct label on the last (current best) point */}
          <text
            x={Math.min(xFor(n - 1), VW - PAD.right - 2)}
            y={yFor(last.best) - 8}
            textAnchor="end"
            fontSize={9.5}
            fontWeight={600}
            fontFamily="'JetBrains Mono', monospace"
            fill="var(--text)"
          >
            {fmtMetric(last.best)}
          </text>
        </svg>

        {hover != null && (
          <div className="pointer-events-none absolute left-[8px] top-0 rounded-[7px] border border-edge bg-right px-[9px] py-[6px] font-mono text-[10px] leading-[1.5] text-ink-2 shadow-win">
            <div className="text-ink">Round {traj.points[hover].round}</div>
            <div>best {fmtMetric(traj.points[hover].best)}</div>
            {traj.points[hover].roundBest != null && (
              <div className="text-ink-3">this round {fmtMetric(traj.points[hover].roundBest!)}</div>
            )}
          </div>
        )}
      </div>

      {/* summary line */}
      <div className="mt-[12px] flex items-center gap-[10px] font-mono text-[10px] text-ink-3">
        {traj.baseline != null && (
          <span>
            baseline <span className="text-ink-2">{fmtMetric(traj.baseline)}</span>
          </span>
        )}
        <span style={{ color: "var(--green)" }}>
          best {fmtMetric(last.best)}
        </span>
        {traj.thresholdValue != null && (
          <span className="ml-auto" style={crossed ? { color: "var(--green)" } : undefined}>
            {crossed ? "threshold crossed" : `need ${fmtMetric(traj.thresholdValue)}`}
          </span>
        )}
      </div>
    </div>
  );
}
