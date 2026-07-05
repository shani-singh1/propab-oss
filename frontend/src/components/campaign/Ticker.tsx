import { useEffect, useMemo, useState } from "react";
import type { PropabEvent } from "../../types";
import { eventLabel, PHASE_LABEL, phaseOf } from "../../lib/events";

// Rotates through the most recent event lines, like a live console footer.
export default function Ticker({ events }: { events: PropabEvent[] }) {
  const lines = useMemo(() => {
    const recent = events.slice(-9);
    return recent.map((e) => {
      const src = e.source || PHASE_LABEL[phaseOf(e.event_type)].toLowerCase();
      return `${src} · ${eventLabel(e)}`;
    });
  }, [events]);

  const [i, setI] = useState(0);
  useEffect(() => {
    if (lines.length <= 1) return;
    const t = setInterval(() => setI((p) => (p + 1) % lines.length), 2200);
    return () => clearInterval(t);
  }, [lines.length]);

  const line = lines.length ? lines[i % lines.length] : "waiting for activity…";

  return (
    <div className="flex h-10 shrink-0 items-center gap-[9px] border-t border-line bg-rail px-4">
      <span className="h-[6px] w-[6px] shrink-0 animate-ppulse rounded-full bg-pos" />
      <span
        key={i}
        className="animate-ptick truncate font-mono text-[11px] leading-none text-ink-2"
      >
        {line}
      </span>
    </div>
  );
}
