import { useEffect, useState } from "react";
import { api } from "../api";
import type { CampaignListItem } from "../types";

// Polls the campaign list. Used by the left navigator and the home grid.
export function useCampaigns(pollMs = 8000) {
  const [campaigns, setCampaigns] = useState<CampaignListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const load = () =>
      api
        .listCampaigns()
        .then((c) => {
          if (!active) return;
          setCampaigns(c);
          setError(null);
        })
        .catch((e) => active && setError(e?.message ?? String(e)))
        .finally(() => active && setLoading(false));
    load();
    const t = window.setInterval(load, pollMs);
    return () => {
      active = false;
      window.clearInterval(t);
    };
  }, [pollMs]);

  return { campaigns, loading, error };
}
