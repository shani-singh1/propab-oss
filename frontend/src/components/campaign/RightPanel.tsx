import { useEffect, useState } from "react";
import type { BeliefObject, HypothesisTree, PropabEvent } from "../../types";
import type { CampaignModel } from "../../lib/model";
import HypothesisTreeView from "./HypothesisTreeView";
import BeliefsView from "./BeliefsView";
import WorkersPanel from "./WorkersPanel";
import TasksPanel from "./TasksPanel";
import Ticker from "./Ticker";
import { StatusDot } from "./bits";

type Tab = "workers" | "tasks" | "tree" | "beliefs";

function TabButton({
  active,
  onClick,
  label,
  count,
  live,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  count: number;
  live?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      className="flex shrink-0 items-center gap-[6px] py-[14px] text-[12px] font-semibold leading-none"
      style={{
        color: active ? "var(--text)" : "var(--text3)",
        boxShadow: active ? "inset 0 -2px 0 var(--text)" : "none",
      }}
    >
      {live && <StatusDot color="var(--text)" live size={6} />}
      {label}
      <span className="rounded bg-chip px-[5px] py-0.5 font-mono text-[9.5px] font-medium text-ink-3">
        {count}
      </span>
    </button>
  );
}

export default function RightPanel({
  model,
  tree,
  question,
  beliefs,
  events,
  now,
  active,
  open,
  onToggle,
}: {
  model: CampaignModel;
  tree: HypothesisTree | undefined;
  question: string;
  beliefs: BeliefObject[];
  events: PropabEvent[];
  now: number;
  active: boolean;
  open: boolean;
  onToggle: () => void;
}) {
  const [tab, setTab] = useState<Tab>("workers");
  const nodeCount = tree ? Object.keys(tree.nodes || {}).length : 0;
  const runningTasks = model.inFlight.length;

  // Default to the most relevant tab: while live, show what's running.
  useEffect(() => {
    if (!active && model.workers.length === 0 && nodeCount > 0) setTab("tree");
    // only on first meaningful data
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  if (!open) {
    return (
      <aside
        onClick={onToggle}
        className="flex w-[46px] shrink-0 cursor-pointer flex-col items-center gap-4 border-l border-line bg-right py-[15px]"
      >
        <span className="text-[16px] text-ink-3">‹</span>
        <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3 [writing-mode:vertical-rl]">
          Workers &amp; tasks
        </span>
        {active && runningTasks > 0 && (
          <span className="mt-auto flex flex-col items-center gap-1">
            <span className="font-mono text-[11px] text-ink-2">{runningTasks}</span>
            <StatusDot color="var(--text)" live size={7} />
          </span>
        )}
      </aside>
    );
  }

  return (
    <aside className="flex w-[352px] shrink-0 flex-col border-l border-line bg-right">
      <div className="flex shrink-0 items-center gap-[16px] border-b border-line px-[16px]">
        <div className="pp-scroll flex min-w-0 flex-1 items-center gap-[16px] overflow-x-auto">
          <TabButton active={tab === "workers"} onClick={() => setTab("workers")} label="Workers" count={model.workers.length} live={active && model.counts.workersRunning > 0} />
          <TabButton active={tab === "tasks"} onClick={() => setTab("tasks")} label="Tasks" count={runningTasks} live={active && runningTasks > 0} />
          <TabButton active={tab === "tree"} onClick={() => setTab("tree")} label="Tree" count={nodeCount} />
          <TabButton active={tab === "beliefs"} onClick={() => setTab("beliefs")} label="Beliefs" count={beliefs.length} />
        </div>
        <button
          onClick={onToggle}
          title="Collapse panel"
          className="shrink-0 px-1 py-2 text-[16px] text-ink-3 hover:text-ink"
          aria-label="Collapse panel"
        >
          ›
        </button>
      </div>

      <div className="pp-scroll min-h-0 flex-1 overflow-auto">
        {tab === "workers" && <WorkersPanel workers={model.workers} now={now} />}
        {tab === "tasks" && (
          <TasksPanel tasks={model.inFlight} workers={model.workers} now={now} active={active} />
        )}
        {tab === "tree" && <HypothesisTreeView tree={tree} question={question} />}
        {tab === "beliefs" && <BeliefsView beliefs={beliefs} />}
      </div>

      <Ticker events={events} />
    </aside>
  );
}
