import type { HypothesisTree, Verdict } from "../types";
import { verdictColor } from "./status";

export interface LaidNode {
  id: string;
  x: number;
  y: number;
  r: number;
  label: string;
  cap: string;
  color: string;
  verdict: Verdict | "root";
  /** filled circle (root) vs. outlined (hypothesis). */
  filled: boolean;
  pulse: boolean;
}

export interface LaidEdge {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface LaidTree {
  width: number;
  height: number;
  nodes: LaidNode[];
  edges: LaidEdge[];
  /** id -> short label (H1, H2 …) for cross-referencing in the timeline. */
  labels: Record<string, string>;
}

const LEVEL_GAP = 96;
const TOP_PAD = 40;
const MIN_SPACING = 88;
const SIDE_PAD = 24;

// Compute a tidy top-down layout for the hypothesis tree, rooted at a synthetic
// "RQ" node that stands in for the campaign question.
export function layoutTree(tree: HypothesisTree | undefined, _question: string): LaidTree {
  const nodes = tree?.nodes ?? {};
  const ids = Object.keys(nodes);

  // children map + real roots (parent missing or null)
  const children: Record<string, string[]> = {};
  const roots: string[] = [];
  for (const id of ids) {
    const p = nodes[id].parent_id;
    if (p && nodes[p]) {
      (children[p] ||= []).push(id);
    } else {
      roots.push(id);
    }
  }

  // BFS levels from the synthetic RQ root (level 0). Real roots are level 1.
  const level: Record<string, number> = {};
  const order: string[] = [];
  let queue = roots.map((id) => ({ id, lvl: 1 }));
  const seen = new Set<string>();
  while (queue.length) {
    const next: typeof queue = [];
    for (const { id, lvl } of queue) {
      if (seen.has(id)) continue;
      seen.add(id);
      level[id] = lvl;
      order.push(id);
      for (const c of children[id] || []) next.push({ id: c, lvl: lvl + 1 });
    }
    queue = next;
  }

  // group ids per level
  const byLevel: Record<number, string[]> = { 0: ["__rq__"] };
  for (const id of order) (byLevel[level[id]] ||= []).push(id);
  const levels = Object.keys(byLevel)
    .map(Number)
    .sort((a, b) => a - b);

  const maxPerLevel = Math.max(1, ...levels.map((l) => byLevel[l].length));
  const width = Math.max(308, maxPerLevel * MIN_SPACING);
  const height = TOP_PAD + levels.length * LEVEL_GAP - (LEVEL_GAP - 48);

  // short H-labels for real level-1 hypotheses
  const labels: Record<string, string> = {};
  let hCount = 0;
  for (const id of byLevel[1] || []) labels[id] = `H${(hCount += 1)}`;

  const pos: Record<string, { x: number; y: number }> = {};
  const laid: LaidNode[] = [];

  for (const l of levels) {
    const row = byLevel[l];
    const y = TOP_PAD + l * LEVEL_GAP;
    const usable = width - SIDE_PAD * 2;
    row.forEach((id, i) => {
      const x = SIDE_PAD + (usable * (i + 1)) / (row.length + 1);
      pos[id] = { x, y };
      if (id === "__rq__") {
        laid.push({
          id,
          x,
          y,
          r: 20,
          label: "RQ",
          cap: "",
          color: "var(--text)",
          verdict: "root",
          filled: true,
          pulse: false,
        });
        return;
      }
      const n = nodes[id];
      const r = l === 1 ? 17 : 12;
      const lbl = labels[id] || "e";
      laid.push({
        id,
        x,
        y,
        r,
        label: lbl,
        cap: l <= 1 ? truncate(n.text, 12) : "",
        color: verdictColor(n.verdict),
        verdict: n.verdict,
        filled: false,
        pulse: n.verdict === "pending",
      });
    });
  }

  // edges: RQ -> real roots, and every parent -> child
  const edges: LaidEdge[] = [];
  for (const rid of roots) {
    if (pos["__rq__"] && pos[rid]) {
      edges.push({ x1: pos["__rq__"].x, y1: pos["__rq__"].y, x2: pos[rid].x, y2: pos[rid].y });
    }
  }
  for (const pId of Object.keys(children)) {
    for (const c of children[pId]) {
      if (pos[pId] && pos[c]) {
        edges.push({ x1: pos[pId].x, y1: pos[pId].y, x2: pos[c].x, y2: pos[c].y });
      }
    }
  }

  return { width, height: Math.max(height, 200), nodes: laid, edges, labels };
}

function truncate(s: string, n: number): string {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
