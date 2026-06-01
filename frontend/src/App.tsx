import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { api } from "./api";
import type { CampaignListItem } from "./types";
import { StatusDot } from "./components/ui";
import { truncate } from "./lib/format";

function Sidebar() {
  const [campaigns, setCampaigns] = useState<CampaignListItem[]>([]);
  const loc = useLocation();

  useEffect(() => {
    let active = true;
    const load = () =>
      api
        .listCampaigns()
        .then((c) => active && setCampaigns(c))
        .catch(() => {});
    load();
    const t = window.setInterval(load, 8000);
    return () => {
      active = false;
      window.clearInterval(t);
    };
  }, [loc.pathname]);

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-border bg-surface">
      <Link to="/" className="flex items-center gap-2 px-4 py-4">
        <span className="grid h-7 w-7 place-items-center rounded-lg bg-brand/20 text-brand">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path
              d="M12 2l2.5 6.5L21 11l-6.5 2.5L12 20l-2.5-6.5L3 11l6.5-2.5L12 2z"
              fill="currentColor"
            />
          </svg>
        </span>
        <span className="text-lg font-semibold tracking-tight">Propab</span>
      </Link>

      <Link
        to="/new"
        className="mx-3 mb-3 flex items-center justify-center gap-2 rounded-lg border border-brand/40 bg-brand/10 py-2 text-sm font-medium text-brand transition hover:bg-brand/20"
      >
        + New campaign
      </Link>

      <div className="px-3 pb-2 text-[11px] uppercase tracking-wide text-text-muted">
        Campaigns
      </div>
      <nav className="flex-1 space-y-0.5 overflow-y-auto scrollbar-thin px-2 pb-4">
        {campaigns.length === 0 && (
          <div className="px-2 py-3 text-xs text-text-muted">No campaigns yet.</div>
        )}
        {campaigns.map((c) => (
          <NavLink
            key={c.id}
            to={`/campaign/${c.id}`}
            className={({ isActive }) =>
              `block rounded-lg px-2.5 py-2 text-sm transition ${
                isActive ? "bg-raised text-text-primary" : "text-text-secondary hover:bg-raised/60"
              }`
            }
          >
            <div className="flex items-center gap-2">
              <StatusDot status={c.status} />
              <span className="truncate">{truncate(c.question, 34)}</span>
            </div>
            <div className="mt-0.5 pl-4 text-[11px] text-text-muted">
              {c.total_confirmed}/{c.total_hypotheses} confirmed · {c.status}
            </div>
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
