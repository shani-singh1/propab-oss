import { useEffect, useState } from "react";

// A coarse clock that re-renders on an interval so live elapsed timers tick.
// Pauses (no interval) when `active` is false — completed campaigns don't churn.
export function useNow(active: boolean, everyMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const t = window.setInterval(() => setNow(Date.now()), everyMs);
    return () => window.clearInterval(t);
  }, [active, everyMs]);
  return now;
}
