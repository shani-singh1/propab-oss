import type { HypothesisTree, Verdict } from "../types";

// Layout engine for the *interactive* hypothesis tree (§D). Unlike lib/tree.ts
// (which lays out the old static top-down thumbnail), this produces an absolute
// tidy layout keyed on the raw tree so the view can zoom/pan, collapse subtrees,
// and highlight the path to root — and it stays legible at 100+ nodes because
// collapsed subtrees contribute a single node, not a hairball.

export const RQ_ID = "__rq__";

export interface GraphNode {
  id: string;
  x: number;
  y: number;
  depth: number;
  verdict: Verdict | "root";
  isRoot: boolean;
  isFrontier: boolean;
  isConfirmed: boolean;
  isExhausted: boolean;
  /** the node has children currently hidden under a collapsed toggle. */
  collapsed: boolean;
  /** number of descendants hidden when collapsed (for the "+N" badge). */
  hiddenCount: number;
  hasChildren: boolean;
  label: string;
  text: string;
  confidence: number;
  evidenceSummary: string | null;
  expansionType: string | null;
  parentId: string | null;
}

export interface GraphEdge {
  from: string;
  to: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  verdict: Verdict | "root";
}

export interface GraphLayout {
  nodes: GraphNode[];
  edges: GraphEdge[];
  byId: Record<string, GraphNode>;
  width: number;
  height: number;
  maxDepth: number;
  /** ids of every node hidden inside a collapsed subtree (for counts). */
  hiddenIds: Set<string>;
}

const X_GAP = 64;
const Y_GAP = 88;
const PAD = 40;

// Nodes at/deeper than this auto-collapse (unless the user expands them), and
// exhausted subtrees collapse too — the two big sources of hairball.
export const AUTO_COLLAPSE_DEPTH = 4;

export interface LayoutOptions {
  /** ids the user explicitly toggled — flips that node's default collapsed state. */
  toggled: Set<string>;
}

export function layoutGraph(
  tree: HypothesisTree | undefined,
  question: string,
  opts: LayoutOptions,
): GraphLayout {
  const nodes = tree?.nodes ?? {};
  const ids = Object.keys(nodes);
  const frontier = new Set(tree?.frontier ?? []);
  const confirmed = new Set(tree?.confirmed ?? []);
  const exhausted = new Set(tree?.exhausted ?? []);

  // children map + real roots (parent missing/null)
  const children: Record<string, string[]> = { [RQ_ID]: [] };
  const roots: string[] = [];
  for (const id of ids) {
    const p = nodes[id].parent_id;
    if (p && nodes[p]) (children[p] ||= []).push(id);
    else roots.push(id);
  }
  children[RQ_ID] = roots;

  const depthOf: Record<string, number> = { [RQ_ID]: 0 };
  const parentOf: Record<string, string | null> = { [RQ_ID]: null };

  // Default-collapsed: a node whose subtree hides by default (deep or exhausted).
  const defaultCollapsed = (id: string, depth: number): boolean => {
    if (id === RQ_ID) return false;
    if (exhausted.has(id) && (children[id]?.length ?? 0) > 0) return true;
    if (depth >= AUTO_COLLAPSE_DEPTH && (children[id]?.length ?? 0) > 0) return true;
    return false;
  };
  const isCollapsed = (id: string, depth: number): boolean => {
    const dflt = defaultCollapsed(id, depth);
    return opts.toggled.has(id) ? !dflt : dflt;
  };

  // Count all descendants of a node (for the +N badge).
  const descendantCount = (id: string): number => {
    let n = 0;
    const stack = [...(children[id] ?? [])];
    while (stack.length) {
      const c = stack.pop()!;
      n += 1;
      for (const cc of children[c] ?? []) stack.push(cc);
    }
    return n;
  };

  // Tidy layout: post-order assign x to leaves sequentially, parents = mean(child x).
  const xOf: Record<string, number> = {};
  let leafCursor = 0;
  const hiddenIds = new Set<string>();
  const visible: string[] = [];

  const place = (id: string, depth: number) => {
    depthOf[id] = depth;
    visible.push(id);
    const kids = children[id] ?? [];
    const collapsed = isCollapsed(id, depth);
    if (collapsed || kids.length === 0) {
      xOf[id] = leafCursor * X_GAP;
      leafCursor += 1;
      if (collapsed) for (const c of kids) markHidden(c);
      return;
    }
    for (const c of kids) {
      parentOf[c] = id;
      place(c, depth + 1);
    }
    const first = xOf[kids[0]];
    const last = xOf[kids[kids.length - 1]];
    xOf[id] = (first + last) / 2;
  };
  const markHidden = (id: string) => {
    hiddenIds.add(id);
    for (const c of children[id] ?? []) markHidden(c);
  };

  parentOf[RQ_ID] = null;
  place(RQ_ID, 0);

  // Build output nodes.
  const outNodes: GraphNode[] = [];
  const byId: Record<string, GraphNode> = {};
  let hCount = 0;
  const labelOf: Record<string, string> = {};
  for (const id of visible) {
    if (id === RQ_ID) continue;
    if (depthOf[id] === 1) labelOf[id] = `H${(hCount += 1)}`;
  }

  let maxDepth = 0;
  for (const id of visible) {
    const depth = depthOf[id];
    maxDepth = Math.max(maxDepth, depth);
    const x = PAD + xOf[id];
    const y = PAD + depth * Y_GAP;
    if (id === RQ_ID) {
      const g: GraphNode = {
        id,
        x,
        y,
        depth,
        verdict: "root",
        isRoot: true,
        isFrontier: false,
        isConfirmed: false,
        isExhausted: false,
        collapsed: false,
        hiddenCount: 0,
        hasChildren: roots.length > 0,
        label: "RQ",
        text: question || "Research question",
        confidence: 0,
        evidenceSummary: null,
        expansionType: null,
        parentId: null,
      };
      outNodes.push(g);
      byId[id] = g;
      continue;
    }
    const n = nodes[id];
    const collapsed = isCollapsed(id, depth);
    const g: GraphNode = {
      id,
      x,
      y,
      depth,
      verdict: n.verdict,
      isRoot: false,
      isFrontier: frontier.has(id),
      isConfirmed: confirmed.has(id),
      isExhausted: exhausted.has(id),
      collapsed,
      hiddenCount: collapsed ? descendantCount(id) : 0,
      hasChildren: (children[id]?.length ?? 0) > 0,
      label: labelOf[id] || `e${depth}`,
      text: n.text,
      confidence: n.confidence,
      evidenceSummary: n.evidence_summary ?? null,
      expansionType: n.expansion_type ?? null,
      parentId: parentOf[id] ?? null,
    };
    outNodes.push(g);
    byId[id] = g;
  }

  // Edges among visible nodes.
  const edges: GraphEdge[] = [];
  for (const g of outNodes) {
    const pid = g.parentId;
    if (pid && byId[pid]) {
      const p = byId[pid];
      edges.push({ from: pid, to: g.id, x1: p.x, y1: p.y, x2: g.x, y2: g.y, verdict: g.verdict });
    }
  }

  const width = PAD * 2 + Math.max(0, (leafCursor - 1) * X_GAP);
  const height = PAD * 2 + maxDepth * Y_GAP;

  return { nodes: outNodes, edges, byId, width, height, maxDepth, hiddenIds };
}

// Ids on the path from a node up to the synthetic root (inclusive of both).
export function pathToRoot(layout: GraphLayout, id: string): Set<string> {
  const out = new Set<string>();
  let cur: string | null = id;
  while (cur) {
    out.add(cur);
    cur = layout.byId[cur]?.parentId ?? null;
  }
  return out;
}
