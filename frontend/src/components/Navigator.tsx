import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useCampaigns } from "../hooks/useCampaigns";
import { useTheme } from "../theme";
import { useUIStore } from "../uiStore";
import { campaignProgress, statusView, toneColor } from "../lib/status";
import { fmtPct, fmtRelative, truncate } from "../lib/format";
import { Bar, Dot } from "./primitives";
import type { CampaignListItem } from "../types";

type SortKey = "recent" | "best" | "confirmed";
type FilterKey = "all" | "active" | "done";

function improvement(c: CampaignListItem): number | null {
  if (c.improvement_pct != null) return c.improvement_pct;
  if (c.baseline_metric && c.best_metric && c.baseline_metric !== 0) {
    return ((c.best_metric - c.baseline_metric) / Math.abs(c.baseline_metric)) * 100;
  }
  return null;
}

function CampaignRow({
  c,
  selected,
  dense,
}: {
  c: CampaignListItem;
  selected: boolean;
  dense: boolean;
}) {
  const nav = useNavigate();
  const sv = statusView(c.status);
  const pct = campaignProgress(c);
  const imp = improvement(c);
  const impColor = imp == null ? "var(--text4)" : imp > 0 ? "var(--green)" : imp < 0 ? "var(--red)" : "var(--text3)";
  return (
    <button
      onClick={() => nav(`/campaign/${c.id}`)}
      aria-current={selected ? "page" : undefined}
      className={`pp-row pp-enter flex w-full flex-col rounded-[7px] text-left ${
        dense ? "gap-[5px] px-[10px] py-[7px]" : "gap-[7px] px-[11px] py-[9px]"
      }`}
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
      <div className="flex w-full items-center gap-[10px] pl-[15px] font-mono text-[9.5px] font-medium leading-none text-ink-4">
        <span style={{ color: impColor }}>{imp == null ? "— vs base" : `${fmtPct(imp)} vs base`}</span>
        <span>{c.total_confirmed} confirmed</span>
        {sv.active && <span className="ml-auto text-ink-3">running</span>}
      </div>
    </button>
  );
}

function Section({
  label,
  items,
  selectedId,
  defaultOpen,
  dense,
}: {
  label: string;
  items: CampaignListItem[];
  selectedId?: string;
  defaultOpen: boolean;
  dense: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  if (items.length === 0) return null;
  return (
    <div className="mb-[10px]">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-[7px] px-[6px] py-[6px] text-left"
        aria-expanded={open}
      >
        <span className="w-2 text-[8px] text-ink-4">{open ? "▾" : "▸"}</span>
        <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-3">
          {label}
        </span>
        <span className="ml-auto font-mono text-[10px] font-medium text-ink-4">{items.length}</span>
      </button>
      {open && (
        <div className="mt-0.5 flex flex-col gap-0.5">
          {items.map((c) => (
            <CampaignRow key={c.id} c={c} selected={c.id === selectedId} dense={dense} />
          ))}
        </div>
      )}
    </div>
  );
}

const SORTS: { key: SortKey; label: string }[] = [
  { key: "recent", label: "Recent" },
  { key: "best", label: "Best" },
  { key: "confirmed", label: "Confirmed" },
];

const FILTERS: { key: FilterKey; label: string }[] = [
  { key: "all", label: "All" },
  { key: "active", label: "Active" },
  { key: "done", label: "Done" },
];

export default function Navigator() {
  const { campaigns, loading, error } = useCampaigns();
  const { theme, toggle } = useTheme();
  const density = useUIStore((s) => s.density);
  const toggleDensity = useUIStore((s) => s.toggleDensity);
  const openPalette = useUIStore((s) => s.openPalette);
  const { id } = useParams();
  const nav = useNavigate();

  const [q, setQ] = useState("");
  const [sort, setSort] = useState<SortKey>("recent");
  const [filter, setFilter] = useState<FilterKey>("all");
  const dense = density === "compact";

  const runningCount = useMemo(
    () => campaigns.filter((c) => statusView(c.status).active).length,
    [campaigns],
  );

  const { active, concluded, total } = useMemo(() => {
    const query = q.trim().toLowerCase();
    let list = campaigns;
    if (query) list = list.filter((c) => c.question.toLowerCase().includes(query));

    const isActive = (c: CampaignListItem) =>
      statusView(c.status).active || c.status === "queued";
    if (filter === "active") list = list.filter(isActive);
    else if (filter === "done") list = list.filter((c) => !isActive(c));

    const ts = (c: CampaignListItem) =>
      new Date(c.completed_at ?? c.started_at ?? 0).getTime() || 0;
    const cmp: Record<SortKey, (a: CampaignListItem, b: CampaignListItem) => number> = {
      recent: (a, b) => ts(b) - ts(a),
      best: (a, b) => (improvement(b) ?? -Infinity) - (improvement(a) ?? -Infinity),
      confirmed: (a, b) => b.total_confirmed - a.total_confirmed,
    };
    const sorted = [...list].sort(cmp[sort]);

    const a: CampaignListItem[] = [];
    const z: CampaignListItem[] = [];
    for (const c of sorted) (isActive(c) ? a : z).push(c);
    return { active: a, concluded: z, total: list.length };
  }, [campaigns, q, filter, sort]);

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
        <div className="ml-auto flex items-center gap-[6px]">
          <button
            onClick={toggleDensity}
            title={`Density: ${density} — click for ${dense ? "comfortable" : "compact"}`}
            aria-label="Toggle density"
            className="rounded-[6px] border border-edge px-[7px] py-[5px] text-[11px] font-medium leading-none text-ink-3 hover:text-ink"
          >
            {dense ? "▪" : "▤"}
          </button>
          <button
            onClick={toggle}
            className="rounded-[20px] border border-edge px-[9px] py-[5px] text-[11px] font-medium leading-none text-ink-3 hover:text-ink"
          >
            {theme === "dark" ? "Light" : "Dark"}
          </button>
        </div>
      </div>

      {/* new campaign — prominent, filled CTA */}
      <div className="px-3 pb-[6px] pt-[10px]">
        <button
          onClick={() => nav("/new")}
          className="pp-row flex w-full items-center justify-center gap-[7px] rounded-lg bg-ink px-[10px] py-[9px] text-[12.5px] font-semibold leading-none shadow-tab"
          style={{ color: "var(--centerBg)" }}
        >
          <span className="text-[14px] font-normal">+</span> New campaign
        </button>
      </div>

      {/* search + palette shortcut */}
      <div className="px-3 pb-[6px] pt-1">
        <div className="flex items-center gap-2 rounded-[8px] border border-edge px-[9px] py-[6px]">
          <span className="text-[11px] text-ink-4">⌕</span>
          <input
            id="pp-nav-search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search campaigns"
            spellCheck={false}
            autoComplete="off"
            className="min-w-0 flex-1 bg-transparent text-[12px] font-medium text-ink outline-none"
          />
          {q ? (
            <button
              onClick={() => setQ("")}
              aria-label="Clear search"
              className="text-[12px] leading-none text-ink-4 hover:text-ink"
            >
              ×
            </button>
          ) : (
            <span className="pp-kbd">/</span>
          )}
        </div>
      </div>

      {/* filter + sort */}
      <div className="flex items-center gap-1 px-3 pb-2 pt-0.5">
        <div className="flex items-center gap-0.5">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className="rounded-[6px] px-[7px] py-[4px] text-[10.5px] font-semibold leading-none"
              style={{
                color: filter === f.key ? "var(--text)" : "var(--text3)",
                background: filter === f.key ? "var(--chip)" : "transparent",
              }}
            >
              {f.label}
            </button>
          ))}
        </div>
        <div className="relative ml-auto">
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as SortKey)}
            aria-label="Sort campaigns"
            className="cursor-pointer appearance-none rounded-[6px] border border-edge bg-transparent py-[4px] pl-[8px] pr-[18px] text-[10.5px] font-semibold leading-none text-ink-3 outline-none hover:text-ink"
          >
            {SORTS.map((s) => (
              <option key={s.key} value={s.key} style={{ background: "var(--railBg)", color: "var(--text)" }}>
                {s.label}
              </option>
            ))}
          </select>
          <span className="pointer-events-none absolute right-[6px] top-1/2 -translate-y-1/2 text-[8px] text-ink-4">
            ▾
          </span>
        </div>
      </div>

      {/* sections */}
      <div className="pp-scroll flex-1 overflow-y-auto px-3 pb-[10px] pt-0.5">
        {error && (
          <div className="rounded-[8px] border border-edge px-3 py-3 text-[11px] leading-relaxed text-ink-3">
            Can’t reach the API. Is the backend running?
          </div>
        )}

        {!error && loading && campaigns.length === 0 && (
          <div className="flex flex-col gap-2 px-1 pt-2">
            {[0, 1, 2].map((i) => (
              <div key={i} className="flex flex-col gap-[7px] rounded-[7px] px-1 py-2">
                <div className="pp-skeleton h-[13px] w-[80%]" />
                <div className="pp-skeleton h-[3px] w-full" />
              </div>
            ))}
          </div>
        )}

        {!error && !loading && total === 0 && campaigns.length > 0 && (
          <div className="px-2 py-6 text-center">
            <div className="text-[12px] font-semibold text-ink">No matches</div>
            <div className="mt-1 text-[11px] leading-relaxed text-ink-3">
              Nothing here fits “{q || filter}”. Try a different search or filter.
            </div>
          </div>
        )}

        {!error && !loading && campaigns.length === 0 && (
          <div className="px-2 py-6 text-center">
            <div className="text-[12.5px] font-semibold text-ink">No campaigns yet</div>
            <div className="mt-1 text-[11px] leading-relaxed text-ink-3">
              Pose a research question and Propab goes to work — reading, hypothesizing, testing.
            </div>
            <button
              onClick={() => nav("/new")}
              className="mt-3 rounded-[7px] border border-edge px-3 py-[7px] text-[11.5px] font-semibold text-ink hover:bg-rowhover"
            >
              + Start one
            </button>
          </div>
        )}

        {filter !== "done" && (
          <Section label="Active" items={active} selectedId={id} defaultOpen dense={dense} />
        )}
        {filter !== "active" && (
          <Section label="Concluded" items={concluded} selectedId={id} defaultOpen={filter === "done"} dense={dense} />
        )}
      </div>

      {/* footer — status + shortcuts hint */}
      <div className="flex shrink-0 flex-col gap-[6px] border-t border-line px-[14px] py-[9px]">
        <div className="flex items-center gap-[10px] text-[10.5px] font-medium leading-none text-ink-3">
          {runningCount > 0 ? (
            <span className="flex items-center gap-[5px]">
              <Dot color="var(--text)" pulse size={6} />
              {runningCount} running
            </span>
          ) : (
            <span className="text-ink-4">{campaigns.length} total</span>
          )}
          <button
            onClick={openPalette}
            className="ml-auto flex items-center gap-[5px] text-ink-3 hover:text-ink"
            title="Command palette"
          >
            <span className="pp-kbd">⌘K</span>
          </button>
        </div>
        <div className="flex items-center gap-[14px] text-[11px] font-medium leading-none text-ink-3">
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
      </div>
    </aside>
  );
}
