import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

// Center-column live-edge behavior (design.md §C "follow-the-live-edge").
// Owns the scroll container: auto-scrolls while the reader is pinned to the
// bottom, and — when they scroll up to read history — stops following and
// surfaces a "jump to live (N new)" pill counting the narrative items that
// arrived meanwhile. No dependency; a soft-smooth jump that respects
// motion-reduce.

// Shared reduced-motion signal (reactive to the OS setting changing live).
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    if (typeof matchMedia === "undefined") return;
    const mq = matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener?.("change", onChange);
    return () => mq.removeEventListener?.("change", onChange);
  }, []);
  return reduced;
}

const AT_BOTTOM_PX = 90;

export interface FollowEdge {
  ref: React.RefObject<HTMLDivElement>;
  onScroll: () => void;
  pinned: boolean;
  newCount: number;
  jump: () => void;
}

// `growth` should increase with any content change (e.g. events.length) so the
// view keeps following the live edge while pinned; `items` should count the
// coarse narrative units (rounds + milestones) so the pill can say "N new".
export function useFollowEdge(growth: number, items: number): FollowEdge {
  const ref = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const [pinned, setPinned] = useState(true);
  const [newCount, setNewCount] = useState(0);
  const prevItems = useRef(items);
  const reduced = usePrefersReducedMotion();

  const setPin = useCallback((v: boolean) => {
    pinnedRef.current = v;
    setPinned(v);
  }, []);

  const onScroll = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = dist <= AT_BOTTOM_PX;
    if (atBottom !== pinnedRef.current) setPin(atBottom);
    if (atBottom) setNewCount(0);
  }, [setPin]);

  // Follow the live edge on any growth while pinned.
  useLayoutEffect(() => {
    const el = ref.current;
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [growth]);

  // Count new narrative items while the reader is away from the edge.
  useEffect(() => {
    const added = items - prevItems.current;
    prevItems.current = items;
    if (added > 0 && !pinnedRef.current) setNewCount((n) => n + added);
  }, [items]);

  const jump = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: reduced ? "auto" : "smooth" });
    setPin(true);
    setNewCount(0);
  }, [reduced, setPin]);

  return { ref, onScroll, pinned, newCount, jump };
}

// The floating "↓ jump to live" pill — shown only when the reader has scrolled
// away from the edge. Announces new arrivals politely for screen readers.
export function JumpToLivePill({ show, newCount, onClick }: { show: boolean; newCount: number; onClick: () => void }) {
  if (!show) return null;
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-[10px] z-10 flex justify-center">
      <button
        onClick={onClick}
        aria-live="polite"
        className="pointer-events-auto flex items-center gap-[7px] rounded-full border border-edge bg-rail/95 px-[13px] py-[6px] text-[11.5px] font-medium text-ink-2 shadow-tab backdrop-blur transition-colors hover:text-ink"
      >
        <span aria-hidden className="text-[12px] leading-none">
          ↓
        </span>
        <span>jump to live</span>
        {newCount > 0 && (
          <span className="rounded-full bg-chip px-[6px] py-[1px] font-mono text-[10px] tabular-nums text-ink-2">
            {newCount} new
          </span>
        )}
      </button>
    </div>
  );
}
