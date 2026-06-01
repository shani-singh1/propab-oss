import { useEffect, useRef } from "react";
import { api, normalizeEvent } from "../api";
import { useLiveStore } from "../store";

const TERMINAL = new Set(["completed", "budget_exhausted", "failed", "breakthrough"]);

// Connects a campaign to live data: initial event backfill + state snapshot, an SSE
// subscription for new events, and a periodic snapshot poll for the overview/tree.
export function useCampaignLive(id: string | undefined) {
  const init = useLiveStore((s) => s.init);
  const setCampaign = useLiveStore((s) => s.setCampaign);
  const setEvents = useLiveStore((s) => s.setEvents);
  const addEvent = useLiveStore((s) => s.addEvent);
  const setConnected = useLiveStore((s) => s.setConnected);
  const setError = useLiveStore((s) => s.setError);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!id) return;
    init(id);
    let cancelled = false;
    let poll: number | undefined;

    const loadSnapshot = async () => {
      try {
        const c = await api.getCampaign(id);
        if (!cancelled) setCampaign(c);
        return c;
      } catch (e: any) {
        if (!cancelled) setError(e?.message ?? String(e));
        return null;
      }
    };

    (async () => {
      try {
        const events = await api.getEvents(id, 1500);
        if (!cancelled) setEvents(events);
      } catch {
        /* events backfill is best-effort */
      }
      const c = await loadSnapshot();

      // SSE for live events.
      const es = new EventSource(api.streamUrl(id));
      esRef.current = es;
      es.onopen = () => !cancelled && setConnected(true);
      es.onerror = () => !cancelled && setConnected(false);
      es.onmessage = (msg) => {
        if (cancelled) return;
        try {
          addEvent(normalizeEvent(JSON.parse(msg.data)));
        } catch {
          /* ignore malformed frame */
        }
      };

      // Snapshot poll (tree/metrics) — slower when terminal.
      const status = c?.summary?.status ?? "active";
      const interval = TERMINAL.has(status) ? 0 : 6000;
      if (interval > 0) {
        poll = window.setInterval(loadSnapshot, interval);
      }
    })();

    return () => {
      cancelled = true;
      esRef.current?.close();
      esRef.current = null;
      if (poll) window.clearInterval(poll);
      setConnected(false);
    };
  }, [id]);
}
