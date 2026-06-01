import { useMemo, useState } from "react";
import type { HypothesisNode, HypothesisTree as Tree } from "../../types";
import { Badge } from "../ui";
import { verdictBorder, verdictColor } from "../../lib/events";
import { fmtMetric } from "../../lib/format";

type Filter = "all" | "confirmed" | "frontier";

export function HypothesisTree({ tree }: { tree: Tree | undefined }) {
  const [filter, setFilter] = useState<Filter>("all");
  const [openId, setOpenId] = useState<string | null>(null);

  const { roots, childrenOf, frontier } = useMemo(() => {
    const nodes = tree?.nodes ?? {};
    const frontier = new Set(tree?.frontier ?? []);
    const childrenOf: Record<string, HypothesisNode[]> = {};
    const roots: HypothesisNode[] = [];
    for (const n of Object.values(nodes)) {
      if (n.parent_id && nodes[n.parent_id]) {
        (childrenOf[n.parent_id] ||= []).push(n);
      } else {
        roots.push(n);
      }
    }
    const byGen = (a: HypothesisNode, b: HypothesisNode) =>
      a.generation - b.generation || a.id.localeCompare(b.id);
    roots.sort(byGen);
    Object.values(childrenOf).forEach((arr) => arr.sort(byGen));
    return { roots, childrenOf, frontier };
  }, [tree]);

  const counts = tree?.nodes ? Object.keys(tree.nodes).length : 0;

  const show = (n: HypothesisNode): boolean => {
    if (filter === "confirmed") return n.verdict === "confirmed";
    if (filter === "frontier") return frontier.has(n.id);
    return true;
  };

  const renderNode = (n: HypothesisNode, depth: number): JSX.Element | null => {
    const kids = childrenOf[n.id] ?? [];
    const visibleKids = kids.map((k) => renderNode(k, depth + 1)).filter(Boolean);
    if (!show(n) && visibleKids.length === 0) return null;
    const onFrontier = frontier.has(n.id);
    const open = openId === n.id;
    return (
      <div key={n.id}>
        <button
          onClick={() => setOpenId(open ? null : n.id)}
          className={`group flex w-full items-start gap-2 border-l-2 py-1.5 pr-2 text-left transition hover:bg-raised/50 ${verdictBorder(
            n.verdict,
          )} ${onFrontier ? "bg-running/5" : ""}`}
          style={{ paddingLeft: 8 + depth * 16 }}
        >
          <span className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${dotColor(n.verdict)}`} />
          <span className="min-w-0 flex-1">
            <span className="block truncate text-sm text-text-primary">{n.text || "(untitled)"}</span>
            <span className="mt-0.5 flex flex-wrap items-center gap-2 text-[11px] text-text-muted">
              <span className={verdictColor(n.verdict)}>{n.verdict}</span>
              {n.confidence > 0 && <span>conf {fmtMetric(n.confidence)}</span>}
              <span>gen {n.generation}</span>
              {onFrontier && <span className="text-running">frontier</span>}
            </span>
          </span>
        </button>
        {open && (
          <div
            className="animate-fadeInUp border-l-2 border-transparent bg-bg/60 px-3 py-2 text-xs text-text-secondary"
            style={{ marginLeft: 8 + depth * 16 }}
          >
            <p className="whitespace-pre-wrap leading-relaxed text-text-primary">{n.text}</p>
            {n.evidence_summary && (
              <p className="mt-2 max-h-40 overflow-y-auto scrollbar-thin whitespace-pre-wrap break-words font-mono text-[11px] text-text-secondary">
                {n.evidence_summary}
              </p>
            )}
          </div>
        )}
        {visibleKids}
      </div>
    );
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold">Hypothesis tree</h2>
          <Badge tone="neutral">{counts} nodes</Badge>
        </div>
        <div className="flex gap-1">
          {(["all", "confirmed", "frontier"] as Filter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`rounded-md px-2 py-1 text-[11px] capitalize transition ${
                filter === f ? "bg-brand/15 text-brand" : "text-text-muted hover:bg-raised"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto scrollbar-thin py-1">
        {counts === 0 ? (
          <div className="px-4 py-6 text-sm text-text-muted">
            No hypotheses yet — waiting for seed generation…
          </div>
        ) : (
          roots.map((r) => renderNode(r, 0))
        )}
      </div>
    </div>
  );
}

function dotColor(v: string): string {
  switch (v) {
    case "confirmed":
      return "bg-confirmed";
    case "refuted":
      return "bg-refuted";
    case "pending":
      return "bg-running animate-pulseSoft";
    default:
      return "bg-inconclusive";
  }
}
