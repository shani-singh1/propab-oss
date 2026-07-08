import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// A tiny, dependency-free list virtualizer (windowing) tuned for the right-panel
// worker/task card lists. It renders only the cards intersecting the viewport
// (plus an overscan margin), so 500–2000+ cards stay smooth. Cards have variable
// height (a collapsed card vs. one expanded into its full detail), so heights are
// *measured* after render via ResizeObserver and cached by index; unmeasured rows
// fall back to `estimateHeight`. Offsets are a memoized running scan over the
// measured/estimated sizes — O(n) per recompute, which is fine for a few thousand.

export interface VirtualItem {
  index: number;
  /** px offset from the top of the scrolled content */
  start: number;
  /** current measured (or estimated) height */
  size: number;
  /** attach to the rendered row's outer element so its height is measured */
  measureRef: (el: HTMLElement | null) => void;
}

export interface UseVirtualOptions {
  count: number;
  /** returns the scroll container element (the element with overflow-y:auto) */
  getScrollElement: () => HTMLElement | null;
  estimateHeight?: number;
  overscan?: number;
  /** a value that, when it changes, invalidates all measured heights (e.g. filter) */
  resetKey?: unknown;
}

export interface UseVirtualResult {
  virtualItems: VirtualItem[];
  totalSize: number;
  /** scroll the container so item `index` is visible near the top */
  scrollToIndex: (index: number) => void;
}

export function useVirtual({
  count,
  getScrollElement,
  estimateHeight = 96,
  overscan = 6,
  resetKey,
}: UseVirtualOptions): UseVirtualResult {
  const [measured, setMeasured] = useState<Record<number, number>>({});
  const [scrollTop, setScrollTop] = useState(0);
  const [viewport, setViewport] = useState(0);
  const elsRef = useRef<Map<number, HTMLElement>>(new Map());
  const roRef = useRef<ResizeObserver | null>(null);

  // Reset measurements when the underlying list identity changes (filter/sort).
  useEffect(() => {
    setMeasured({});
  }, [resetKey]);

  // Track container scroll + size.
  useEffect(() => {
    const el = getScrollElement();
    if (!el) return;
    const onScroll = () => setScrollTop(el.scrollTop);
    const measureViewport = () => setViewport(el.clientHeight);
    measureViewport();
    setScrollTop(el.scrollTop);
    el.addEventListener("scroll", onScroll, { passive: true });
    const ro = new ResizeObserver(measureViewport);
    ro.observe(el);
    return () => {
      el.removeEventListener("scroll", onScroll);
      ro.disconnect();
    };
  }, [getScrollElement]);

  // One ResizeObserver watches every mounted row; updates the measured cache.
  useEffect(() => {
    const ro = new ResizeObserver((entries) => {
      let dirty = false;
      const next: Record<number, number> = {};
      for (const entry of entries) {
        const idx = Number((entry.target as HTMLElement).dataset.vindex);
        if (Number.isNaN(idx)) continue;
        const h = Math.round(entry.contentRect.height);
        if (h > 0) {
          next[idx] = h;
          dirty = true;
        }
      }
      // Only trigger a state update when a height ACTUALLY changed — otherwise the
      // new object identity re-renders → re-attaches refs → re-observes → the RO
      // fires again, an infinite loop. Returning `prev` unchanged lets React bail.
      if (dirty)
        setMeasured((prev) => {
          let changed = false;
          for (const k in next) {
            if (prev[Number(k)] !== next[Number(k)]) {
              changed = true;
              break;
            }
          }
          return changed ? { ...prev, ...next } : prev;
        });
    });
    roRef.current = ro;
    return () => ro.disconnect();
  }, []);

  // A STABLE ref callback per index (cached), so React doesn't treat it as a new
  // ref every render and detach/reattach (which would re-observe and re-loop).
  const refCache = useRef<Map<number, (el: HTMLElement | null) => void>>(new Map());
  const measureRefFor = useCallback((index: number) => {
    const cache = refCache.current;
    let cb = cache.get(index);
    if (!cb) {
      cb = (el: HTMLElement | null) => {
        if (el) {
          el.dataset.vindex = String(index);
          elsRef.current.set(index, el);
          roRef.current?.observe(el);
          const h = Math.round(el.getBoundingClientRect().height);
          if (h > 0) setMeasured((p) => (p[index] === h ? p : { ...p, [index]: h }));
        } else {
          const prev = elsRef.current.get(index);
          if (prev && roRef.current) roRef.current.unobserve(prev);
          elsRef.current.delete(index);
        }
      };
      cache.set(index, cb);
    }
    return cb;
  }, []);

  // Cumulative offsets over the whole list (measured height or estimate).
  const offsets = useMemo(() => {
    const arr = new Float64Array(count + 1);
    for (let i = 0; i < count; i++) {
      arr[i + 1] = arr[i] + (measured[i] ?? estimateHeight);
    }
    return arr;
  }, [count, measured, estimateHeight]);

  const totalSize = offsets[count] ?? 0;

  const { startIndex, endIndex } = useMemo(() => {
    if (count === 0) return { startIndex: 0, endIndex: -1 };
    const top = Math.max(0, scrollTop);
    const bottom = top + (viewport || 600);
    // binary search for first item whose end is past the top edge
    let lo = 0;
    let hi = count - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (offsets[mid + 1] <= top) lo = mid + 1;
      else hi = mid;
    }
    let start = lo;
    let end = start;
    while (end < count - 1 && offsets[end] < bottom) end++;
    start = Math.max(0, start - overscan);
    end = Math.min(count - 1, end + overscan);
    return { startIndex: start, endIndex: end };
  }, [count, scrollTop, viewport, offsets, overscan]);

  const virtualItems = useMemo(() => {
    const items: VirtualItem[] = [];
    for (let i = startIndex; i <= endIndex; i++) {
      items.push({
        index: i,
        start: offsets[i],
        size: measured[i] ?? estimateHeight,
        measureRef: measureRefFor(i),
      });
    }
    return items;
  }, [startIndex, endIndex, offsets, measured, estimateHeight, measureRefFor]);

  const scrollToIndex = useCallback(
    (index: number) => {
      const el = getScrollElement();
      if (!el) return;
      el.scrollTo({ top: offsets[Math.max(0, Math.min(count, index))] ?? 0, behavior: "smooth" });
    },
    [getScrollElement, offsets, count],
  );

  return { virtualItems, totalSize, scrollToIndex };
}

// Convenience: a ref-based accessor for the scroll element, so callers don't
// re-create `getScrollElement` on every render.
export function useScrollRef<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const get = useCallback(() => ref.current, []);
  return [ref, get] as const;
}
