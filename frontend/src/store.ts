import { create } from "zustand";
import type { CampaignState, PropabEvent } from "./types";

const MAX_EVENTS = 4000;

interface LiveStore {
  campaignId: string | null;
  campaign: CampaignState | null;
  events: PropabEvent[];
  connected: boolean;
  error: string | null;

  init: (id: string) => void;
  setCampaign: (c: CampaignState) => void;
  setEvents: (e: PropabEvent[]) => void;
  addEvent: (e: PropabEvent) => void;
  setConnected: (v: boolean) => void;
  setError: (e: string | null) => void;
  reset: () => void;
}

export const useLiveStore = create<LiveStore>((set, get) => ({
  campaignId: null,
  campaign: null,
  events: [],
  connected: false,
  error: null,

  init: (id) => {
    if (get().campaignId !== id) {
      set({ campaignId: id, campaign: null, events: [], connected: false, error: null });
    }
  },
  setCampaign: (c) => set({ campaign: c }),
  setEvents: (e) => set({ events: e.slice(-MAX_EVENTS) }),
  addEvent: (e) =>
    set((s) => {
      // de-dupe by event_id (SSE may overlap the initial backfill)
      if (s.events.length && s.events[s.events.length - 1]?.event_id === e.event_id) return s;
      const next = [...s.events, e];
      return { events: next.length > MAX_EVENTS ? next.slice(-MAX_EVENTS) : next };
    }),
  setConnected: (v) => set({ connected: v }),
  setError: (e) => set({ error: e }),
  reset: () => set({ campaignId: null, campaign: null, events: [], connected: false, error: null }),
}));
