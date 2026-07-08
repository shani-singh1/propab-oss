import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { HypothesisTree } from "../../types";
import { layoutGraph, pathToRoot, type GraphNode } from "../../lib/treeGraph";
import { verdictColor } from "../../lib/status";

// Interactive hypothesis tree (§D): zoom/pan, click a node for a detail popover,
// verdict coloring, hover-to-highlight the path to root, collapsible deep/exhausted
// subtrees, and auto-fit — so 100+ nodes never become a hairball.

interface Transform {
  scale: number;
  tx: number;
  ty: number;
}

function radiusFor(n: GraphNode): number {
  if (n.isRoot) return 18;
  if (n.depth === 1) return 14;
  if (n.depth === 2) return 11;
  return 9;
}

function nodeFill(n: GraphNode): string {
  if (n.isRoot) return "var(--text)";
  if (n.verdict === "confirmed") return "var(--greenDim)";
  if (n.verdict === "refuted") return "var(--redDim)";
  return "var(--rightBg)";
}

export default function HypothesisTreeView({
  tree,
  question,
}: {
  tree: HypothesisTree | undefined;
  question: string;
}) {
  const [toggled, setToggled] = useState<Set<string>>(new Set());
  const [transform, setTransform] = useState<Transform>({ scale: 1, tx: 0, ty: 0 });
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [size, setSize] = useState({ w: 320, h: 380 });
  const containerRef = useRef<HTMLDivElement | null>(null);
  const userMoved = useRef(false);
  const drag = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);

  const layout = useMemo(() => layoutGraph(tree, question, { toggled }), [tree, question, toggled]);
  const nodeCount = layout.nodes.length - 1; // minus synthetic root
  const empty = nodeCount <= 0;

  // Measure the container.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setSize({ w: el.clientWidth, h: el.clientHeight });
    });
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  const fit = useCallback(() => {
    const pad = 28;
    const gw = Math.max(1, layout.width);
    const gh = Math.max(1, layout.height);
    const scale = Math.min((size.w - pad * 2) / gw, (size.h - pad * 2) / gh, 1.4);
    const s = Math.max(0.15, scale);
    const tx = (size.w - gw * s) / 2;
    const ty = pad;
    setTransform({ scale: s, tx, ty });
  }, [layout.width, layout.height, size.w, size.h]);

  // Auto-fit until the user manually pans/zooms; re-fit when structure changes.
  useEffect(() => {
    if (!userMoved.current) fit();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fit, layout.width, layout.height, size.w, size.h]);

  const zoomBy = useCallback((factor: number, cx: number, cy: number) => {
    userMoved.current = true;
    setTransform((t) => {
      const scale = Math.max(0.15, Math.min(3, t.scale * factor));
      const k = scale / t.scale;
      // keep the point (cx,cy) fixed while zooming
      return { scale, tx: cx - (cx - t.tx) * k, ty: cy - (cy - t.ty) * k };
    });
  }, []);

  // Native, non-passive wheel listener so preventDefault actually suppresses the
  // ancestor scroll (React attaches onWheel passively).
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      zoomBy(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX - rect.left, e.clientY - rect.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomBy]);

  const onPointerDown = (e: React.PointerEvent) => {
    if ((e.target as Element).closest("[data-node]")) return; // node clicks handled separately
    userMoved.current = true;
    drag.current = { x: e.clientX, y: e.clientY, tx: transform.tx, ty: transform.ty };
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    setTransform((t) => ({
      ...t,
      tx: drag.current!.tx + (e.clientX - drag.current!.x),
      ty: drag.current!.ty + (e.clientY - drag.current!.y),
    }));
  };
  const onPointerUp = (e: React.PointerEvent) => {
    drag.current = null;
    try {
      (e.currentTarget as Element).releasePointerCapture(e.pointerId);
    } catch {
      /* no-op */
    }
  };

  const toggleCollapse = (id: string) =>
    setToggled((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  const highlight = useMemo(
    () => (hoverId ? pathToRoot(layout, hoverId) : null),
    [hoverId, layout],
  );

  const selected = selectedId ? layout.byId[selectedId] : null;

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* controls */}
      <div className="flex shrink-0 items-center gap-[8px] border-b border-line px-[14px] py-[8px]">
        <span className="font-mono text-[10px] text-ink-3">
          {nodeCount} node{nodeCount === 1 ? "" : "s"}
          {layout.hiddenIds.size > 0 && (
            <span className="text-ink-4"> · {layout.hiddenIds.size} collapsed</span>
          )}
        </span>
        <div className="ml-auto flex items-center gap-[4px]">
          {[
            ["−", () => zoomBy(1 / 1.2, size.w / 2, size.h / 2)],
            ["+", () => zoomBy(1.2, size.w / 2, size.h / 2)],
          ].map(([g, fn]) => (
            <button
              key={g as string}
              onClick={fn as () => void}
              className="flex h-[22px] w-[22px] items-center justify-center rounded-[6px] border border-edge font-mono text-[12px] text-ink-2 hover:bg-chip"
            >
              {g as string}
            </button>
          ))}
          <button
            onClick={() => {
              userMoved.current = false;
              fit();
            }}
            className="rounded-[6px] border border-edge px-[8px] py-[3px] font-mono text-[10px] text-ink-2 hover:bg-chip"
          >
            Fit
          </button>
        </div>
      </div>

      {/* canvas */}
      <div
        ref={containerRef}
        className="relative min-h-0 flex-1 touch-none overflow-hidden"
        style={{ cursor: drag.current ? "grabbing" : "grab" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        {empty ? (
          <div className="absolute inset-0 flex items-center justify-center text-[11.5px] text-ink-3">
            No hypotheses yet
          </div>
        ) : (
          <svg width={size.w} height={size.h} style={{ display: "block" }}>
            <g transform={`translate(${transform.tx},${transform.ty}) scale(${transform.scale})`}>
              {layout.edges.map((e) => {
                const on = highlight ? highlight.has(e.to) && highlight.has(e.from) : true;
                return (
                  <line
                    key={`${e.from}->${e.to}`}
                    x1={e.x1}
                    y1={e.y1}
                    x2={e.x2}
                    y2={e.y2}
                    stroke={on ? "var(--border)" : "var(--divider)"}
                    strokeWidth={on && highlight ? 2 : 1.4}
                    opacity={highlight && !on ? 0.4 : 1}
                  />
                );
              })}
              {layout.nodes.map((n) => {
                const r = radiusFor(n);
                const stroke = n.isRoot ? "var(--text)" : verdictColor(n.verdict);
                const dim = highlight && !highlight.has(n.id);
                const isSel = n.id === selectedId;
                return (
                  <g
                    key={n.id}
                    data-node
                    style={{ cursor: "pointer" }}
                    opacity={dim ? 0.35 : 1}
                    onMouseEnter={() => setHoverId(n.id)}
                    onMouseLeave={() => setHoverId(null)}
                    onClick={() => setSelectedId((p) => (p === n.id ? null : n.id))}
                    onDoubleClick={() => n.hasChildren && !n.isRoot && toggleCollapse(n.id)}
                  >
                    {isSel && (
                      <circle cx={n.x} cy={n.y} r={r + 4} fill="none" stroke={stroke} strokeWidth={1.5} opacity={0.5} />
                    )}
                    <circle
                      cx={n.x}
                      cy={n.y}
                      r={r}
                      fill={nodeFill(n)}
                      stroke={stroke}
                      strokeWidth={2}
                      className={n.isFrontier ? "animate-ppulse motion-reduce:animate-none" : ""}
                    />
                    <text
                      x={n.x}
                      y={n.y}
                      textAnchor="middle"
                      dominantBaseline="central"
                      fontFamily="'JetBrains Mono', ui-monospace, monospace"
                      fontSize={n.isRoot || n.depth === 1 ? 10 : 8.5}
                      fontWeight={600}
                      fill={n.isRoot ? "var(--rightBg)" : "var(--text)"}
                      style={{ pointerEvents: "none" }}
                    >
                      {n.label}
                    </text>
                    {n.collapsed && n.hiddenCount > 0 && (
                      <>
                        <circle cx={n.x + r - 1} cy={n.y - r + 1} r={7} fill="var(--text)" />
                        <text
                          x={n.x + r - 1}
                          y={n.y - r + 1}
                          textAnchor="middle"
                          dominantBaseline="central"
                          fontFamily="'JetBrains Mono', ui-monospace, monospace"
                          fontSize={7.5}
                          fontWeight={700}
                          fill="var(--rightBg)"
                          style={{ pointerEvents: "none" }}
                        >
                          +{n.hiddenCount}
                        </text>
                      </>
                    )}
                  </g>
                );
              })}
            </g>
          </svg>
        )}

        {/* detail popover */}
        {selected && (
          <NodePopover
            node={selected}
            screen={{
              x: transform.tx + selected.x * transform.scale,
              y: transform.ty + selected.y * transform.scale,
            }}
            container={size}
            onClose={() => setSelectedId(null)}
            onToggleCollapse={() => toggleCollapse(selected.id)}
          />
        )}
      </div>

      {/* legend */}
      <div className="flex shrink-0 flex-wrap gap-x-[14px] gap-y-[6px] border-t border-line px-[16px] py-[11px] text-[10.5px] text-ink-2">
        {[
          ["Confirmed", "var(--green)"],
          ["Testing", "var(--text)"],
          ["Refuted", "var(--red)"],
          ["Inconclusive", "var(--text3)"],
          ["Queued", "var(--text4)"],
        ].map(([label, color]) => (
          <span key={label} className="flex items-center gap-[6px]">
            <span className="h-2 w-2 rounded-full" style={{ background: color as string }} />
            {label}
          </span>
        ))}
        <span className="ml-auto font-mono text-[9.5px] text-ink-4">drag · scroll · dbl-click to fold</span>
      </div>
    </div>
  );
}

function NodePopover({
  node,
  screen,
  container,
  onClose,
  onToggleCollapse,
}: {
  node: GraphNode;
  screen: { x: number; y: number };
  container: { w: number; h: number };
  onClose: () => void;
  onToggleCollapse: () => void;
}) {
  const W = 236;
  // Clamp within the container; prefer to the right of the node, flip if needed.
  let left = screen.x + 16;
  if (left + W > container.w - 8) left = Math.max(8, screen.x - W - 16);
  let top = Math.max(8, Math.min(screen.y - 10, container.h - 180));
  const color = node.isRoot ? "var(--text)" : verdictColor(node.verdict);

  return (
    <div
      className="absolute z-20 rounded-[10px] border border-edge bg-right p-[12px] shadow-win"
      style={{ left, top, width: W }}
      onPointerDown={(e) => e.stopPropagation()}
    >
      <div className="mb-[7px] flex items-center gap-[7px]">
        <span className="h-[8px] w-[8px] rounded-full" style={{ background: color }} />
        <span className="font-mono text-[11px] font-semibold text-ink">{node.label}</span>
        <span className="font-mono text-[9.5px] uppercase tracking-[0.08em]" style={{ color }}>
          {node.isRoot ? "root" : node.verdict}
        </span>
        <button
          onClick={onClose}
          className="ml-auto text-[13px] leading-none text-ink-3 hover:text-ink"
          aria-label="Close"
        >
          ×
        </button>
      </div>
      <div className="mb-[8px] text-[12px] leading-[1.45] text-ink">{node.text}</div>
      {!node.isRoot && (
        <div className="flex flex-col gap-[5px] font-mono text-[10px] leading-none text-ink-3">
          {node.confidence > 0 && (
            <div>
              confidence <span className="text-ink-2">{node.confidence.toFixed(2)}</span>
            </div>
          )}
          {node.expansionType && (
            <div>
              via <span className="text-ink-2">{node.expansionType}</span>
            </div>
          )}
          {node.evidenceSummary && (
            <div className="whitespace-normal leading-[1.4] text-ink-3">{node.evidenceSummary}</div>
          )}
          {node.isExhausted && <div className="text-ink-4">exhausted branch</div>}
        </div>
      )}
      {node.hasChildren && !node.isRoot && (
        <button
          onClick={onToggleCollapse}
          className="mt-[9px] w-full rounded-[6px] border border-edge py-[5px] font-mono text-[10px] text-ink-2 hover:bg-chip"
        >
          {node.collapsed ? `Expand ${node.hiddenCount} hidden` : "Collapse subtree"}
        </button>
      )}
    </div>
  );
}
