import { useMemo } from "react";
import type { HypothesisTree } from "../../types";
import { layoutTree } from "../../lib/tree";

// SVG hypothesis tree, laid out top-down from a synthetic RQ root.
export default function HypothesisTreeView({
  tree,
  question,
}: {
  tree: HypothesisTree | undefined;
  question: string;
}) {
  const laid = useMemo(() => layoutTree(tree, question), [tree, question]);
  const empty = (laid.nodes?.length ?? 0) <= 1;

  return (
    <div>
      <div className="relative w-full" style={{ minHeight: 260 }}>
        <svg
          viewBox={`0 0 ${laid.width} ${laid.height}`}
          width="100%"
          preserveAspectRatio="xMidYMin meet"
          style={{ display: "block", overflow: "visible" }}
        >
          {laid.edges.map((e, i) => (
            <line
              key={i}
              x1={e.x1}
              y1={e.y1}
              x2={e.x2}
              y2={e.y2}
              stroke="var(--border)"
              strokeWidth={1.5}
            />
          ))}
          {laid.nodes.map((n) => {
            const fill = n.filled
              ? "var(--text)"
              : n.verdict === "confirmed"
                ? "var(--greenDim)"
                : n.verdict === "refuted"
                  ? "var(--redDim)"
                  : "var(--rightBg)";
            return (
              <g key={n.id} className={n.pulse ? "animate-ppulse" : ""}>
                <circle cx={n.x} cy={n.y} r={n.r} fill={fill} stroke={n.color} strokeWidth={2} />
                <text
                  x={n.x}
                  y={n.y}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontFamily="'JetBrains Mono', ui-monospace, monospace"
                  fontSize={n.r >= 17 ? 11 : 9}
                  fontWeight={600}
                  fill={n.filled ? "var(--rightBg)" : "var(--text)"}
                >
                  {n.label}
                </text>
                {n.cap && (
                  <text
                    x={n.x}
                    y={n.y + n.r + 12}
                    textAnchor="middle"
                    fontSize={9.5}
                    fontWeight={500}
                    fill="var(--text3)"
                  >
                    {n.cap}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
        {empty && (
          <div className="absolute inset-x-0 bottom-4 text-center text-[11.5px] text-ink-3">
            No hypotheses yet
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-[14px] border-t border-line px-[18px] py-[14px] text-[11px] text-ink-2">
        {[
          ["Supported", "var(--green)"],
          ["Testing", "var(--text)"],
          ["Refuted", "var(--red)"],
          ["Queued", "var(--text4)"],
        ].map(([label, color]) => (
          <span key={label} className="flex items-center gap-[6px]">
            <span className="h-2 w-2 rounded-full" style={{ background: color }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
