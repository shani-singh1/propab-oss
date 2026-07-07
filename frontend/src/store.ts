import { create } from "zustand";
import type { CampaignState, PropabEvent } from "./types";

const MAX_EVENTS = 4000;

interface LiveStore {
  campaignId: string | null;
  campaign: CampaignState | null;
  events: PropabEvent[];
  /** event_ids currently held — for O(1) de-dupe across backfill/SSE overlap. */
  seen: Set<string>;
  connected: boolean;
  error: string | null;

  init: (id: string) => void;
  setCampaign: (c: CampaignState) => void;
  setEvents: (e: PropabEvent[]) => void;
  addEvent: (e: PropabEvent) => void;
  mergeEvents: (e: PropabEvent[]) => void;
  setConnected: (v: boolean) => void;
  setError: (e: string | null) => void;
  reset: () => void;
}

function trim(events: PropabEvent[]): { events: PropabEvent[]; seen: Set<string> } {
  const next = events.length > MAX_EVENTS ? events.slice(-MAX_EVENTS) : events;
  return { events: next, seen: new Set(next.map((e) => e.event_id)) };
}

export const useLiveStore = create<LiveStore>((set, get) => ({
  campaignId: null,
  campaign: null,
  events: [],
  seen: new Set(),
  connected: false,
  error: null,

  init: (id) => {
    if (get().campaignId !== id) {
      set({ campaignId: id, campaign: null, events: [], seen: new Set(), connected: false, error: null });
    }
  },
  setCampaign: (c) => set({ campaign: c }),
  setEvents: (e) => set(trim(e)),
  addEvent: (e) =>
    set((s) => {
      if (s.seen.has(e.event_id)) return s;
      const { events, seen } = trim([...s.events, e]);
      return { events, seen };
    }),
  // Merge a backfilled batch (e.g. after a reconnect) keeping chronological order
  // and dropping anything already held.
  mergeEvents: (batch) =>
    set((s) => {
      const fresh = batch.filter((e) => !s.seen.has(e.event_id));
      if (!fresh.length) return s;
      const merged = [...s.events, ...fresh].sort((a, b) =>
        a.timestamp === b.timestamp ? 0 : a.timestamp < b.timestamp ? -1 : 1,
      );
      return trim(merged);
    }),
  setConnected: (v) => set({ connected: v }),
  setError: (e) => set({ error: e }),
  reset: () =>
    set({ campaignId: null, campaign: null, events: [], seen: new Set(), connected: false, error: null }),
}));
