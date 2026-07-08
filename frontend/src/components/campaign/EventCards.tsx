import { useState } from "react";
import { Link } from "react-router-dom";
import type { PropabEvent } from "../../types";
import type { DiscoverySummary } from "../../lib/model";
import { eventLabel } from "../../lib/events";
import { fmtMetric, fmtOffset, fmtPct } from "../../lib/format";
import { StatusDot } from "./bits";

// ── Special event cards + registry (design.md §B) ────────────────────────────
// Weighty moments (a candidate record, a notable confirmed finding, a finished
// paper, the baseline) deserve a card with color/motion, not a gray row. The
// NarrativeStream dispatches every standalone milestone event through
// `EventCard`, which picks a card by `event_type` and falls back to a plain
// milestone row for everything else. A refuted result is never surfaced here as
// an alarm — refutations read as "information gained" in the round cards.
//
// TODO(design.md §3, item 6): the breakthrough card's witness + certification
// are derived from `campaign.best_finding` via `discoverySummary()`. Once the
// backend emits a first-class candidate-record / certification event carrying
// the witness + certification booleans + metric-vs-best-known, render from that
// event payload directly instead of the derived `discovery` prop.

// ── Shared: collapsible witness JSON ("verify independently") ────────────────
export function WitnessBlock({ witness }: { witness: unknown }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const json = safeJson(witness);
  if (json == null) return null;

  const copy = () => {
    navigator.clipboard?.writeText(json).then(
      () => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      },
      () => undefined,
    );
  };

  return (
    <div className="rounded-[8px] border border-line">
      <div className="flex items-center gap-[10px] px-[11px] py-[7px]">
        <button
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="font-mono text-[10.5px] font-medium text-ink-3 hover:text-ink"
        >
          {open ? "▾ hide" : "▸ show"} witness
        </button>
        <span className="font-mono text-[9.5px] text-ink-4">verify independently</span>
        <button
          onClick={copy}
          className="ml-auto font-mono text-[9.5px] text-ink-3 hover:text-ink"
          title="Copy the witness JSON"
        >
          {copied ? "copied ✓" : "copy"}
        </button>
      </div>
      {open && (
        <pre className="pp-scroll max-h-[220px] overflow-auto border-t border-line px-[11px] py-[9px] font-mono text-[10.5px] leading-[1.5] text-ink-2">
          {json}
        </pre>
      )}
    </div>
  );
}

// ── Shared: certification check booleans ─────────────────────────────────────
export function CertChecks({
  checks,
  certified,
}: {
  checks: Record<string, boolean> | null;
  certified: boolean | null;
}) {
  if (!checks && certified == null) return null;
  return (
    <div className="flex flex-col gap-[5px]">
      {certified != null && (
        <div className="flex items-center gap-[7px]">
          <span
            className="font-mono text-[11px] font-semibold"
            style={{ color: certified ? "var(--green)" : "var(--text2)" }}
          >
            {certified ? "✓ Certified" : "Not yet certified"}
          </span>
          <span className="font-mono text-[9.5px] text-ink-4">
            {certified ? "witness passes every check" : "see checks below"}
          </span>
        </div>
      )}
      {checks && (
        <div className="flex flex-wrap gap-x-[14px] gap-y-[4px]">
          {Object.entries(checks).map(([k, ok]) => (
            <span key={k} className="flex items-center gap-[5px] font-mono text-[10.5px]">
              <span style={{ color: ok ? "var(--green)" : "var(--text3)" }}>{ok ? "✓" : "✗"}</span>
              <span className="text-ink-3">{k.replace(/_/g, " ")}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ── The money shot: full-width celebratory breakthrough card ─────────────────
function BreakthroughCard({
  e,
  start,
  discovery,
  campaignId,
}: {
  e: PropabEvent;
  start: string | null;
  discovery: DiscoverySummary | null;
  campaignId: string | undefined;
}) {
  const p = e.payload || {};
  const metricName = discovery?.metricName ?? (typeof p.metric_name === "string" ? p.metric_name : null);
  const baseline = num(p.baseline_metric) ?? discovery?.meter.baseline ?? null;
  const best = num(p.best_metric) ?? discovery?.best ?? null;
  // event payload carries a *fractional* improvement (e.g. 0.113); the derived
  // meter carries percentage points (e.g. 11.3). Prefer the meter for display.
  const improvementPct =
    discovery?.meter.improvementPct ?? (num(p.improvement_pct) != null ? num(p.improvement_pct)! * 100 : null);
  const bestKnown = discovery?.bestKnown ?? null;
  const witness = discovery?.witness ?? null;
  const checks = discovery?.checks ?? null;
  const certified = discovery?.certified ?? null;

  return (
    <div
      className="mb-[16px] overflow-hidden rounded-[12px] border animate-ptick motion-reduce:animate-none"
      style={{ borderColor: "var(--green)", background: "var(--greenDim)" }}
    >
      <div className="flex items-center gap-[10px] px-[16px] pb-[10px] pt-[13px]">
        <StatusDot color="var(--green)" live size={10} />
        <span className="text-[14px] font-semibold leading-none text-ink">
          Breakthrough — candidate record
        </span>
        <span className="ml-auto font-mono text-[10px] text-ink-4">{fmtOffset(start, e.timestamp)}</span>
      </div>

      {/* the metric it beat */}
      <div className="flex flex-wrap items-baseline gap-x-[20px] gap-y-[5px] px-[16px]">
        {best != null && (
          <span className="flex items-baseline gap-[6px]">
            <span className="font-mono text-[20px] font-semibold leading-none tabular-nums text-ink">
              {fmtMetric(best)}
            </span>
            <span className="font-mono text-[9.5px] uppercase tracking-[0.06em] text-ink-3">
              best{metricName ? ` ${metricName}` : ""}
            </span>
          </span>
        )}
        {baseline != null && (
          <span className="font-mono text-[11px] text-ink-3">
            from baseline <span className="text-ink-2">{fmtMetric(baseline)}</span>
          </span>
        )}
        {bestKnown != null && (
          <span className="font-mono text-[11px] text-ink-3">
            vs best-known <span className="text-ink-2">{fmtMetric(bestKnown)}</span>
          </span>
        )}
        {improvementPct != null && (
          <span className="font-mono text-[12px] font-semibold text-pos">{fmtPct(improvementPct)}</span>
        )}
      </div>

      {discovery?.statement && (
        <div className="mt-[10px] px-[16px] text-[12.5px] leading-[1.5] text-ink-2">
          {discovery.statement}
        </div>
      )}

      {(checks || certified != null || witness != null) && (
        <div className="mt-[12px] flex flex-col gap-[9px] px-[16px]">
          {(checks || certified != null) && <CertChecks checks={checks} certified={certified} />}
          {witness != null && <WitnessBlock witness={witness} />}
        </div>
      )}

      <div className="mt-[12px] flex items-center gap-[12px] border-t border-line px-[16px] py-[9px]">
        <span className="font-mono text-[9.5px] text-ink-4">
          A lower-bound improvement — certifies a record, not optimality.
        </span>
        {campaignId && (
          <Link
            to={`/campaign/${campaignId}/paper`}
            className="ml-auto font-mono text-[10px] text-ink-3 underline decoration-ink-4 underline-offset-2 hover:text-ink"
          >
            open paper
          </Link>
        )}
      </div>
    </div>
  );
}

// ── A notable confirmed finding surfaced by the synthesizer ──────────────────
function FindingCard({ e, start }: { e: PropabEvent; start: string | null }) {
  const p = e.payload || {};
  const finding = typeof p.finding === "string" ? p.finding : typeof p.key_finding === "string" ? p.key_finding : null;
  const round = num(p.round);
  return (
    <div
      className="mb-[14px] overflow-hidden rounded-[10px] border animate-ptick motion-reduce:animate-none"
      style={{ borderColor: "var(--green)", background: "var(--greenDim)" }}
    >
      <div className="flex items-start gap-[10px] px-[15px] py-[12px]">
        <span className="mt-[2px]">
          <StatusDot color="var(--green)" size={9} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-[9px]">
            <span className="text-[12.5px] font-semibold leading-none text-ink">
              Confirmed finding
            </span>
            {round != null && (
              <span className="font-mono text-[10px] text-ink-3">round {round}</span>
            )}
          </div>
          {finding && (
            <div className="mt-[6px] text-[12.5px] leading-[1.5] text-ink-2">{finding}</div>
          )}
        </div>
        <span className="shrink-0 font-mono text-[10px] text-ink-4">{fmtOffset(start, e.timestamp)}</span>
      </div>
    </div>
  );
}

// ── Paper ready ──────────────────────────────────────────────────────────────
function PaperCard({
  e,
  start,
  campaignId,
}: {
  e: PropabEvent;
  start: string | null;
  campaignId: string | undefined;
}) {
  const p = e.payload || {};
  const abstract = typeof p.abstract_latex === "string" ? stripTex(p.abstract_latex) : null;
  const chars = num(p.full_tex_chars);
  const figures = num(p.figures_embedded);
  const pages = chars != null ? Math.max(1, Math.round(chars / 2800)) : null; // ~chars/page estimate
  const pdf = typeof p.pdf_url === "string" ? p.pdf_url : null;

  return (
    <div
      className="mb-[14px] animate-ptick overflow-hidden rounded-[10px] border border-edge motion-reduce:animate-none"
      style={{ background: "var(--railBg)" }}
    >
      <div className="flex items-center gap-[10px] px-[15px] pb-[8px] pt-[12px]">
        <StatusDot color="var(--green)" size={9} />
        <span className="text-[12.5px] font-semibold leading-none text-ink">Paper ready</span>
        <span className="ml-auto font-mono text-[10px] text-ink-4">{fmtOffset(start, e.timestamp)}</span>
      </div>
      {abstract && (
        <div className="line-clamp-3 px-[15px] text-[12px] leading-[1.55] text-ink-2">{abstract}</div>
      )}
      <div className="mt-[10px] flex flex-wrap items-center gap-x-[12px] gap-y-[5px] border-t border-line px-[15px] py-[9px] font-mono text-[10px] text-ink-3">
        {pages != null && (
          <span>
            <span className="font-semibold text-ink-2">~{pages}</span> pages
          </span>
        )}
        {figures != null && (
          <span>
            <span className="font-semibold text-ink-2">{figures}</span> figures
          </span>
        )}
        <span className="ml-auto flex items-center gap-[12px]">
          {campaignId && (
            <Link
              to={`/campaign/${campaignId}/paper`}
              className="underline decoration-ink-4 underline-offset-2 hover:text-ink"
            >
              read paper
            </Link>
          )}
          {pdf && (
            <a href={pdf} target="_blank" rel="noreferrer" className="underline decoration-ink-4 underline-offset-2 hover:text-ink">
              download PDF
            </a>
          )}
        </span>
      </div>
    </div>
  );
}

// ── Baseline measured (a small framed card) ──────────────────────────────────
function BaselineCard({ e, start }: { e: PropabEvent; start: string | null }) {
  const p = e.payload || {};
  const skipped = !!p.note;
  const metric = num(p.baseline_metric);
  const metricName = typeof p.metric_name === "string" ? p.metric_name : null;
  return (
    <div className="mb-[14px] flex animate-ptick items-center gap-[11px] rounded-[10px] border border-edge px-[15px] py-[11px] motion-reduce:animate-none">
      <StatusDot color="var(--text3)" size={8} />
      <div className="min-w-0 flex-1">
        <span className="text-[12px] font-semibold text-ink">Baseline measured</span>
        <span className="ml-[9px] font-mono text-[11px] text-ink-2">
          {skipped
            ? "skipped — verification campaign (target-count objective)"
            : `${metricName ? metricName + " " : ""}${metric != null ? fmtMetric(metric) : "—"}`}
        </span>
      </div>
      <span className="shrink-0 font-mono text-[10px] text-ink-4">{fmtOffset(start, e.timestamp)}</span>
    </div>
  );
}

// ── Fallback: plain lifecycle milestone row (unchanged look) ──────────────────
function milestoneMeta(e: PropabEvent): { color: string; emphatic: boolean } {
  const t = e.event_type;
  if (t === "campaign.completed" && !e.payload?.stop_reason) return { color: "var(--green)", emphatic: true };
  if (t === "campaign.budget_exhausted" || t === "session.failed")
    return { color: "var(--red)", emphatic: true };
  return { color: "var(--text)", emphatic: false };
}

function FallbackMilestone({ e, start }: { e: PropabEvent; start: string | null }) {
  const m = milestoneMeta(e);
  return (
    <div
      className="mb-[14px] flex animate-ptick items-center gap-[11px] rounded-[10px] px-[15px] py-[12px] motion-reduce:animate-none"
      style={{
        border: `1px solid ${m.emphatic ? m.color : "var(--border)"}`,
        background: m.emphatic ? "var(--chip)" : "transparent",
      }}
    >
      <StatusDot color={m.color} size={9} />
      <span className="flex-1 text-[13px] font-semibold leading-none text-ink">{eventLabel(e)}</span>
      <span className="font-mono text-[10.5px] text-ink-4">{fmtOffset(start, e.timestamp)}</span>
    </div>
  );
}

// ── Registry dispatcher ──────────────────────────────────────────────────────
export default function EventCard({
  e,
  start,
  discovery,
  campaignId,
}: {
  e: PropabEvent;
  start: string | null;
  discovery: DiscoverySummary | null;
  campaignId: string | undefined;
}) {
  switch (e.event_type) {
    case "campaign.breakthrough":
      return <BreakthroughCard e={e} start={start} discovery={discovery} campaignId={campaignId} />;
    case "synthesis.breakthrough":
      return <FindingCard e={e} start={start} />;
    case "paper.ready":
      return <PaperCard e={e} start={start} campaignId={campaignId} />;
    case "campaign.baseline_measured":
      return <BaselineCard e={e} start={start} />;
    default:
      return <FallbackMilestone e={e} start={start} />;
  }
}

// ── local utilities ──────────────────────────────────────────────────────────
function num(v: unknown): number | null {
  if (typeof v === "number") return isFinite(v) ? v : null;
  if (typeof v === "string" && v.trim() !== "" && isFinite(Number(v))) return Number(v);
  return null;
}

function safeJson(v: unknown): string | null {
  if (v == null) return null;
  try {
    return typeof v === "string" ? v : JSON.stringify(v, null, 2);
  } catch {
    return null;
  }
}

// Cheap LaTeX de-noise for the abstract preview (strip the commonest commands).
function stripTex(s: string): string {
  return s
    .replace(/\\(?:textbf|textit|emph|section\*?|subsection\*?)\{([^}]*)\}/g, "$1")
    .replace(/\\[a-zA-Z]+\*?/g, "")
    .replace(/[{}]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}
