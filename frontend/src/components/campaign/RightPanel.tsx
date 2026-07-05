import { useState } from "react";
import type { BeliefObject, HypothesisTree, PropabEvent } from "../../types";
import HypothesisTreeView from "./HypothesisTreeView";
import BeliefsView from "./BeliefsView";
import Ticker from "./Ticker";

function TabButton({
  active,
  onClick,
  label,
  count,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  count: number;
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-[7px] py-[15px] text-[13px] font-semibold leading-none"
      style={{
        color: active ? "var(--text)" : "var(--text3)",
        boxShadow: active ? "inset 0 -2px 0 var(--text)" : "none",
      }}
    >
      {label}
      <span className="rounded bg-chip px-[5px] py-0.5 font-mono text-[10px] font-medium text-ink-3">
        {count}
      </span>
    </button>
  );
}

export default function RightPanel({
  tree,
  question,
  beliefs,
  events,
  open,
  onToggle,
}: {
  tree: HypothesisTree | undefined;
  question: string;
  beliefs: BeliefObject[];
  events: PropabEvent[];
  open: boolean;
  onToggle: () => void;
}) {
  const [tab, setTab] = useState<"tree" | "beliefs">("tree");
  const nodeCount = tree ? Object.keys(tree.nodes || {}).length : 0;

  if (!open) {
    return (
      <aside
        onClick={onToggle}
        className="flex w-[46px] shrink-0 cursor-pointer flex-col items-center gap-4 border-l border-line bg-right py-[15px]"
      >
        <span className="text-[16px] text-ink-3">‹</span>
        <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3 [writing-mode:vertical-rl]">
          Investigation map
        </span>
        <span className="mt-auto h-[7px] w-[7px] animate-ppulse rounded-full bg-pos" />
      </aside>
    );
  }

  return (
    <aside className="flex w-[344px] shrink-0 flex-col border-l border-line bg-right">
      <div className="flex shrink-0 items-center gap-5 border-b border-line px-[18px]">
        <TabButton active={tab === "tree"} onClick={() => setTab("tree")} label="Hypothesis tree" count={nodeCount} />
        <TabButton active={tab === "beliefs"} onClick={() => setTab("beliefs")} label="Beliefs" count={beliefs.length} />
        <button
          onClick={onToggle}
          title="Collapse panel"
          className="ml-auto px-1 py-2 text-[16px] text-ink-3 hover:text-ink"
        >
          ›
        </button>
      </div>

      <div className="pp-scroll min-h-0 flex-1 overflow-auto">
        {tab === "tree" ? (
          <HypothesisTreeView tree={tree} question={question} />
        ) : (
          <BeliefsView beliefs={beliefs} />
        )}
      </div>

      <Ticker events={events} />
    </aside>
  );
}
