import type {
  CampaignListItem,
  CampaignState,
  PaperPayload,
  PropabEvent,
} from "./types";

// API base: VITE_API_BASE wins; otherwise use the dev proxy (/api) so SSE is same-origin.
export const API_BASE: string =
  (import.meta as any).env?.VITE_API_BASE?.replace(/\/$/, "") || "/api";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 300)}`);
  }
  return (await res.json()) as T;
}

export interface CreateCampaignBody {
  question: string;
  compute_budget_hours: number;
  breakthrough_criteria: {
    metric_name: string;
    improvement_threshold: number;
    direction: string;
    min_confidence: number;
    min_replications: number;
  };
}

export const api = {
  listCampaigns: () =>
    getJSON<{ campaigns: CampaignListItem[] }>("/campaigns").then((d) => d.campaigns),

  getCampaign: (id: string) => getJSON<CampaignState>(`/campaigns/${id}`),

  getEvents: (id: string, limit?: number) =>
    getJSON<{ events: any[] }>(
      `/sessions/${id}/events${limit ? `?limit=${limit}` : ""}`,
    ).then((d) => d.events.map(normalizeEvent)),

  getPaper: (id: string) =>
    getJSON<{ paper: PaperPayload }>(`/sessions/${id}/paper`).then((d) => d.paper),

  createCampaign: async (body: CreateCampaignBody): Promise<{ campaign_id: string }> => {
    const res = await fetch(`${API_BASE}/campaigns`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`${res.status}: ${text.slice(0, 300)}`);
    }
    return res.json();
  },

  streamUrl: (id: string) => `${API_BASE}/stream/${id}`,
};

// Persisted events use payload_json/created_at/id; SSE uses payload/timestamp/event_id.
export function normalizeEvent(raw: any): PropabEvent {
  return {
    event_id: raw.event_id ?? raw.id ?? crypto.randomUUID(),
    session_id: raw.session_id ?? "",
    timestamp: raw.timestamp ?? raw.created_at ?? new Date().toISOString(),
    source: raw.source ?? "",
    event_type: raw.event_type ?? "",
    step: raw.step ?? "",
    payload:
      typeof raw.payload === "object" && raw.payload !== null
        ? raw.payload
        : typeof raw.payload_json === "object" && raw.payload_json !== null
          ? raw.payload_json
          : safeParse(raw.payload_json) ?? {},
    parent_event_id: raw.parent_event_id ?? null,
    hypothesis_id: raw.hypothesis_id ?? null,
  };
}

function safeParse(v: unknown): Record<string, any> | null {
  if (typeof v !== "string") return null;
  try {
    return JSON.parse(v);
  } catch {
    return null;
  }
}
