import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useCampaigns } from "../hooks/useCampaigns";
import { useTheme } from "../theme";
import { campaignProgress, statusView, toneColor } from "../lib/status";
import { truncate } from "../lib/format";
import { Bar, Dot } from "./primitives";
import type { CampaignListItem } from "../types";

function CampaignRow({ c, selected }: { c: CampaignListItem; selected: boolean }) {
  const nav = useNavigate();
  const sv = statusView(c.status);
  const pct = campaignProgress(c);
  return (
    <button
      onClick={() => nav(`/campaign/${c.id}`)}
      className="pp-row flex w-full flex-col gap-[7px] rounded-[7px] px-[11px] py-[9px] text-left"
      style={
        selected
          ? { background: "var(--rowSel)", boxShadow: "inset 2px 0 0 var(--text)" }
          : { background: "transparent" }
      }
    >
      <div className="flex w-full items-center gap-2">
        <Dot color={toneColor(sv.tone)} pulse={sv.active} size={7} />
        <span className="truncate font-mono text-[12.5px] font-semibold leading-[1.1] text-ink">
          {truncate(c.question, 22)}
        </span>
        <span className="ml-auto whitespace-nowrap text-[10.5px] font-medium text-ink-3">
          {sv.label}
        </span>
      </div>
      <div className="flex w-full items-center gap-2 pl-[15px]">
        <Bar pct={pct} height={3} className="flex-1" />
        <span className="whitespace-nowrap font-mono text-[10px] font-medium text-ink-3">
          {c.total_confirmed}/{c.total_hypotheses}
        </span>
      </div>
    </button>
  );
}

function Section({
  label,
  items,
  selectedId,
  defaultOpen,
}: {
  label: string;
  items: CampaignListItem[];
  selectedId?: string;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mb-[10px]">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-[7px] px-[6px] py-[6px] text-left"
      >
        <span className="w-2 text-[8px] text-ink-4">{open ? "▾" : "▸"}</span>
        <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-3">
          {label}
        </span>
        <span className="ml-auto font-mono text-[10px] font-medium text-ink-4">{items.length}</span>
      </button>
      {open && (
        <div className="mt-0.5 flex flex-col gap-0.5">
          {items.length === 0 && (
            <div className="px-[11px] py-2 text-[11px] text-ink-4">None</div>
          )}
          {items.map((c) => (
            <CampaignRow key={c.id} c={c} selected={c.id === selectedId} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function Navigator() {
  const { campaigns, error } = useCampaigns();
  const { theme, toggle } = useTheme();
  const { id } = useParams();
  const nav = useNavigate();

  const { active, concluded } = useMemo(() => {
    const a: CampaignListItem[] = [];
    const z: CampaignListItem[] = [];
    for (const c of campaigns) {
      (statusView(c.status).active || c.status === "queued" ? a : z).push(c);
    }
    return { active: a, concluded: z };
  }, [campaigns]);

  return (
    <aside className="flex w-[258px] shrink-0 flex-col border-r border-line bg-rail">
      {/* brand */}
      <div className="flex h-[52px] shrink-0 items-center gap-[9px] border-b border-line px-4">
        <Link to="/" className="flex items-center gap-[9px]">
          <span className="flex h-[18px] w-[18px] items-center justify-center rounded-full border-2 border-ink">
            <span className="h-[6px] w-[6px] rounded-full bg-ink" />
          </span>
          <span className="text-[15px] font-semibold leading-none tracking-[-0.01em] text-ink">
            Propab
          </span>
        </Link>
        <button
          onClick={toggle}
          className="ml-auto rounded-[20px] border border-edge px-[9px] py-[5px] text-[11px] font-medium leading-none text-ink-3 hover:text-ink"
        >
          {theme === "dark" ? "Light" : "Dark"}
        </button>
      </div>

      {/* new campaign */}
      <div className="px-3 pb-2 pt-[10px]">
        <button
          onClick={() => nav("/new")}
          className="flex w-full items-center justify-center gap-[7px] rounded-lg border border-edge px-[10px] py-[9px] text-[12.5px] font-semibold leading-none text-ink hover:bg-rowhover"
        >
          <span className="text-[14px] font-normal">+</span> New campaign
        </button>
      </div>

      {/* sections */}
      <div className="pp-scroll flex-1 overflow-y-auto px-3 pb-[10px] pt-0.5">
        {error && (
          <div className="px-2 py-3 text-[11px] leading-relaxed text-ink-3">
            Can’t reach the API. Is the backend running?
          </div>
        )}
        <Section label="Active" items={active} selectedId={id} defaultOpen />
        <Section label="Concluded" items={concluded} selectedId={id} defaultOpen={false} />
      </div>

      {/* footer */}
      <div className="flex shrink-0 items-center gap-[14px] border-t border-line px-[14px] py-[10px] text-[11px] font-medium leading-none text-ink-3">
        <span className="cursor-default">Settings</span>
        <a
          href="https://github.com/shani-singh1/propab-oss"
          target="_blank"
          rel="noreferrer"
          className="hover:text-ink"
        >
          Docs
        </a>
        <span className="ml-auto font-mono text-ink-4">v0.4</span>
      </div>
    </aside>
  );
}
