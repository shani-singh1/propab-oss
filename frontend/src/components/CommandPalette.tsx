import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useUIStore } from "../uiStore";
import { useCampaigns } from "../hooks/useCampaigns";
import { useLiveStore } from "../store";
import { buildCampaignModel } from "../lib/model";
import { appNavigate } from "../lib/appEvents";
import { statusView, toneColor } from "../lib/status";
import { fmtPct, truncate } from "../lib/format";
import { Dot } from "./primitives";

// ⌘K command palette. Mounted at the app shell; self-gates on `paletteOpen` so
// the data hooks and model build only run while it is open. It reaches the
// campaign panels through the `pp:navigate` event bus (see lib/appEvents) rather
// than importing their internals.

interface Command {
  id: string;
  title: string;
  subtitle?: string;
  group: string;
  keywords?: string;
  accent?: string;
  glyph?: React.ReactNode;
  run: () => void;
}

export default function CommandPalette() {
  const open = useUIStore((s) => s.paletteOpen);
  // Gate the heavy body entirely so nothing polls or derives while closed.
  return open ? <PaletteBody /> : null;
}

function PaletteBody() {
  const nav = useNavigate();
  const close = useUIStore((s) => s.closePalette);
  const toggleTheme = useUIStore((s) => s.toggleTheme);
  const toggleDensity = useUIStore((s) => s.toggleDensity);
  const setRightOpen = useUIStore((s) => s.setRightOpen);
  const setRightTab = useUIStore((s) => s.setRightTab);
  const theme = useUIStore((s) => s.theme);
  const density = useUIStore((s) => s.density);

  const { campaigns } = useCampaigns();
  const campaignId = useLiveStore((s) => s.campaignId);
  const events = useLiveStore((s) => s.events);
  const model = useMemo(() => (events.length ? buildCampaignModel(events) : null), [events]);
  const hasPaper = useMemo(() => events.some((e) => e.event_type === "paper.ready"), [events]);

  const [query, setQuery] = useState("");
  const [sel, setSel] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const goTab = (tab: "workers" | "tasks" | "tree" | "beliefs") => {
    setRightOpen(true);
    setRightTab(tab);
    appNavigate({ tab, openRight: true });
    if (campaignId && campaignId !== "demo") nav(`/campaign/${campaignId}`);
  };

  const commands: Command[] = useMemo(() => {
    const cmds: Command[] = [];

    // ── View ────────────────────────────────────────────────────────────────
    cmds.push({
      id: "theme",
      group: "View",
      title: `Switch to ${theme === "dark" ? "light" : "dark"} theme`,
      subtitle: `Currently ${theme}`,
      keywords: "theme dark light appearance toggle",
      glyph: theme === "dark" ? "◐" : "◑",
      run: () => toggleTheme(),
    });
    cmds.push({
      id: "density",
      group: "View",
      title: `Use ${density === "comfortable" ? "compact" : "comfortable"} density`,
      subtitle: `Currently ${density}`,
      keywords: "density compact comfortable spacing",
      glyph: "⇕",
      run: () => toggleDensity(),
    });

    // ── Go to ───────────────────────────────────────────────────────────────
    cmds.push({
      id: "nav-home",
      group: "Go to",
      title: "All campaigns",
      keywords: "home campaigns list overview",
      glyph: "⌂",
      run: () => nav("/"),
    });
    cmds.push({
      id: "nav-new",
      group: "Go to",
      title: "New campaign",
      subtitle: "Start a research program",
      keywords: "new create campaign start",
      glyph: "+",
      run: () => nav("/new"),
    });

    // ── This campaign (tabs / paper / rounds / workers) ─────────────────────
    if (campaignId) {
      (["workers", "tasks", "tree", "beliefs"] as const).forEach((tab) =>
        cmds.push({
          id: `tab-${tab}`,
          group: "This campaign",
          title: `Open ${tab[0].toUpperCase() + tab.slice(1)} tab`,
          keywords: `tab panel ${tab}`,
          glyph: "▤",
          run: () => goTab(tab),
        }),
      );
      if (hasPaper) {
        cmds.push({
          id: "paper",
          group: "This campaign",
          title: "Open paper",
          subtitle: "Compiled write-up",
          keywords: "paper pdf write-up results",
          accent: "var(--green)",
          glyph: "❧",
          run: () => nav(`/campaign/${campaignId}/paper`),
        });
      }
      for (const r of model?.rounds ?? []) {
        if (r.isSetup) continue;
        cmds.push({
          id: `round-${r.number}`,
          group: "Jump to round",
          title: `Round ${r.number}`,
          subtitle: `${r.confirmed} confirmed · ${r.refuted} refuted · ${r.hypothesesGenerated} hyp`,
          keywords: `round ${r.number}`,
          glyph: String(r.number),
          run: () => {
            appNavigate({ round: r.number });
            if (campaignId !== "demo") nav(`/campaign/${campaignId}`);
          },
        });
      }
      for (const w of (model?.workers ?? []).slice(0, 40)) {
        const sv = w.status;
        cmds.push({
          id: `worker-${w.hypothesisId}`,
          group: "Workers",
          title: truncate(w.text || `Worker ${w.shortId}`, 56),
          subtitle: `${sv}${w.confidence != null ? ` · ${Math.round(w.confidence * 100)}%` : ""}`,
          keywords: `worker hypothesis ${w.shortId} ${w.text}`,
          accent:
            sv === "confirmed" ? "var(--green)" : sv === "refuted" ? "var(--red)" : undefined,
          glyph: "◉",
          run: () => {
            appNavigate({ worker: w.hypothesisId, tab: "workers", openRight: true });
            setRightOpen(true);
            setRightTab("workers");
            if (campaignId !== "demo") nav(`/campaign/${campaignId}`);
          },
        });
      }
    }

    // ── Other campaigns ─────────────────────────────────────────────────────
    for (const c of campaigns) {
      if (c.id === campaignId) continue;
      const sv = statusView(c.status);
      cmds.push({
        id: `campaign-${c.id}`,
        group: "Campaigns",
        title: truncate(c.question, 52),
        subtitle: sv.label,
        keywords: `campaign ${c.question} ${sv.label}`,
        accent: toneColor(sv.tone),
        glyph: "●",
        run: () => nav(`/campaign/${c.id}`),
      });
    }

    return cmds;
  }, [
    theme,
    density,
    campaignId,
    campaigns,
    model,
    hasPaper,
    nav,
    toggleTheme,
    toggleDensity,
    setRightOpen,
    setRightTab,
  ]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    const terms = q.split(/\s+/);
    const scored: { cmd: Command; score: number }[] = [];
    for (const cmd of commands) {
      const hay = `${cmd.title} ${cmd.subtitle ?? ""} ${cmd.keywords ?? ""} ${cmd.group}`.toLowerCase();
      let ok = true;
      let score = 0;
      for (const t of terms) {
        const i = hay.indexOf(t);
        if (i < 0) {
          ok = false;
          break;
        }
        score += i === 0 || hay[i - 1] === " " ? 2 : 1;
        if (cmd.title.toLowerCase().includes(t)) score += 3;
      }
      if (ok) scored.push({ cmd, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.map((s) => s.cmd);
  }, [commands, query]);

  useEffect(() => {
    setSel(0);
  }, [query]);

  // Keep the selected row in view.
  useEffect(() => {
    const el = listRef.current?.querySelector(`[data-idx="${sel}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  const runAt = (i: number) => {
    const cmd = filtered[i];
    if (!cmd) return;
    close();
    // Defer so the overlay unmount doesn't race the navigation/toggle.
    requestAnimationFrame(() => cmd.run());
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSel((s) => Math.min(s + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSel((s) => Math.max(s - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      runAt(sel);
    } else if (e.key === "Home") {
      setSel(0);
    } else if (e.key === "End") {
      setSel(filtered.length - 1);
    }
  };

  // Render grouped, preserving the filtered order per group.
  const groups: { name: string; items: { cmd: Command; idx: number }[] }[] = [];
  filtered.forEach((cmd, idx) => {
    let g = groups.find((x) => x.name === cmd.group);
    if (!g) {
      g = { name: cmd.group, items: [] };
      groups.push(g);
    }
    g.items.push({ cmd, idx });
  });

  return (
    <div
      className="pp-overlay fixed inset-0 z-50 flex items-start justify-center px-4 pt-[12vh]"
      onMouseDown={close}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      <div
        className="pp-modal flex max-h-[68vh] w-full max-w-[560px] flex-col overflow-hidden rounded-[13px] border border-edge bg-rail shadow-win"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-[10px] border-b border-line px-[15px] py-[12px]">
          <span className="text-[13px] text-ink-3">⌘</span>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Search commands, campaigns, rounds, workers…"
            className="min-w-0 flex-1 bg-transparent text-[14px] font-medium text-ink outline-none"
            spellCheck={false}
            autoComplete="off"
          />
          <span className="pp-kbd shrink-0">esc</span>
        </div>

        <div ref={listRef} className="pp-scroll min-h-0 flex-1 overflow-y-auto py-2">
          {filtered.length === 0 ? (
            <div className="px-5 py-10 text-center">
              <div className="text-[13px] font-semibold text-ink">Nothing matches “{query}”</div>
              <div className="mt-1 text-[11.5px] text-ink-3">
                Try a campaign name, a round number, or “theme”.
              </div>
            </div>
          ) : (
            groups.map((g) => (
              <div key={g.name} className="mb-1">
                <div className="px-[15px] pb-1 pt-2 font-mono text-[9.5px] font-semibold uppercase tracking-[0.14em] text-ink-4">
                  {g.name}
                </div>
                {g.items.map(({ cmd, idx }) => (
                  <button
                    key={cmd.id}
                    data-idx={idx}
                    onMouseMove={() => setSel(idx)}
                    onClick={() => runAt(idx)}
                    className="flex w-full items-center gap-[11px] px-[15px] py-[9px] text-left"
                    style={{ background: idx === sel ? "var(--rowSel)" : "transparent" }}
                  >
                    <span
                      className="flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-[6px] bg-chip font-mono text-[11px] leading-none"
                      style={{ color: cmd.accent ?? "var(--text3)" }}
                    >
                      {cmd.accent && cmd.glyph === "●" ? (
                        <Dot color={cmd.accent} size={7} />
                      ) : (
                        cmd.glyph
                      )}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[13px] font-medium leading-tight text-ink">
                        {cmd.title}
                      </span>
                      {cmd.subtitle && (
                        <span className="block truncate text-[11px] leading-tight text-ink-3">
                          {cmd.subtitle}
                        </span>
                      )}
                    </span>
                    {idx === sel && <span className="pp-kbd shrink-0">↵</span>}
                  </button>
                ))}
              </div>
            ))
          )}
        </div>

        <div className="flex items-center gap-[14px] border-t border-line px-[15px] py-[8px] text-[10.5px] text-ink-4">
          <span className="flex items-center gap-1">
            <span className="pp-kbd">↑</span>
            <span className="pp-kbd">↓</span>
            navigate
          </span>
          <span className="flex items-center gap-1">
            <span className="pp-kbd">↵</span>
            open
          </span>
          <span className="ml-auto flex items-center gap-1">
            <span className="pp-kbd">?</span>
            shortcuts
          </span>
        </div>
      </div>
    </div>
  );
}
