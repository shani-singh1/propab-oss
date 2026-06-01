import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import type { CampaignListItem } from "../types";
import { Badge, Card, ProgressBar, Spinner, StatusDot } from "../components/ui";
import { budgetPct, fmtDuration, fmtPct, fmtRelative, truncate } from "../lib/format";

export default function Dashboard() {
  const [campaigns, setCampaigns] = useState<CampaignListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const nav = useNavigate();

  useEffect(() => {
    let active = true;
    const load = () =>
      api
        .listCampaigns()
        .then((c) => active && (setCampaigns(c), setError(null)))
        .catch((e) => active && setError(e.message));
    load();
    const t = window.setInterval(load, 8000);
    return () => {
      active = false;
      window.clearInterval(t);
    };
  }, []);

  const active = (campaigns ?? []).filter((c) =>
    ["active", "running"].includes(c.status.toLowerCase()),
  );
  const done = (campaigns ?? []).filter(
    (c) => !["active", "running"].includes(c.status.toLowerCase()),
  );

  return (
    <div className="h-full overflow-y-auto scrollbar-thin">
      <div className="mx-auto max-w-5xl px-8 py-10">
        <div className="mb-8 flex items-end justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Campaigns</h1>
            <p className="mt-1 text-sm text-text-secondary">
              Autonomous research runs — live progress, hypotheses, and papers.
            </p>
          </div>
          <Link
            to="/new"
            className="rounded-lg bg-brand px-4 py-2 text-sm font-medium text-white transition hover:bg-brand/90"
          >
            + New campaign
          </Link>
        </div>

        {error && (
          <Card className="mb-6 border-refuted/40 p-4 text-sm text-refuted">
            Couldn’t reach the API: {error}
          </Card>
        )}
        {campaigns === null && !error && <Spinner />}

        {active.length > 0 && (
          <Section title="Active">
            {active.map((c) => (
              <CampaignCard key={c.id} c={c} onOpen={() => nav(`/campaign/${c.id}`)} />
            ))}
          </Section>
        )}

        {done.length > 0 && (
          <Section title="Completed">
            {done.map((c) => (
              <CampaignCard key={c.id} c={c} onOpen={() => nav(`/campaign/${c.id}`)} />
            ))}
          </Section>
        )}

        {campaigns !== null && campaigns.length === 0 && (
          <Card className="p-10 text-center">
            <p className="text-text-secondary">No campaigns yet.</p>
            <Link to="/new" className="mt-3 inline-block text-brand hover:underline">
              Start your first campaign →
            </Link>
          </Card>
        )}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-8">
      <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-text-muted">
        {title}
      </h2>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function CampaignCard({ c, onOpen }: { c: CampaignListItem; onOpen: () => void }) {
  const pct = budgetPct(c.compute_seconds_used, c.compute_budget_seconds);
  const isActive = ["active", "running"].includes(c.status.toLowerCase());
  return (
    <Card
      className="cursor-pointer p-4 transition hover:border-brand/40 hover:bg-raised/40"
      // @ts-expect-error onClick on div
      onClick={onOpen}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <StatusDot status={c.status} />
            <h3 className="truncate font-medium text-text-primary">{truncate(c.question, 90)}</h3>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-text-secondary">
            <span>
              <span className="font-semibold text-text-primary">{c.total_hypotheses}</span> tested
            </span>
            <span>
              <span className="font-semibold text-confirmed">{c.total_confirmed}</span> confirmed
            </span>
            {c.best_metric > 0 && (
              <span>
                best <span className="font-semibold text-text-primary">{c.best_metric}</span>
              </span>
            )}
            {c.improvement_pct != null && c.improvement_pct !== 0 && (
              <Badge tone="confirmed">{fmtPct(c.improvement_pct)} vs baseline</Badge>
            )}
            <span className="text-text-muted">{fmtRelative(c.started_at)}</span>
          </div>
        </div>
        <div className="flex flex-col items-end gap-2">
          <Badge tone={isActive ? "running" : c.status === "failed" ? "refuted" : "neutral"}>
            {c.status}
          </Badge>
        </div>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <ProgressBar pct={pct} tone={pct > 90 ? "warning" : "brand"} />
        <span className="shrink-0 text-[11px] text-text-muted">
          {pct}% · {fmtDuration(c.compute_seconds_used)} / {fmtDuration(c.compute_budget_seconds)}
        </span>
      </div>
    </Card>
  );
}
