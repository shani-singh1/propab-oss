import type { DiscoverySummary } from "../../lib/model";
import { fmtMetric, fmtPct } from "../../lib/format";
import { CertChecks, WitnessBlock } from "./EventCards";

// ── Discovery Hero (design.md §B) ────────────────────────────────────────────
// Pinned atop the center column. Summarizes the discovery state honestly from
// `campaign.best_finding` + summary metrics: best-so-far vs best-known, what it
// beats / still needs, and — for a certified record — the witness + certification.
//
// TODO(design.md §3, item 6): once the backend emits a first-class
// `finding.best_updated` / candidate-record / certification event, prefer that
// event's payload over the inference in `discoverySummary()` so the witness and
// certification booleans render from real data rather than `best_finding`.

// A short state pill describing where the campaign stands on the record.
function statePill(d: DiscoverySummary): { label: string; color: string; bg: string } {
  if (d.certified) return { label: "Certified record", color: "var(--green)", bg: "var(--greenDim)" };
  if (d.beatsBestKnown) return { label: "Record candidate", color: "var(--green)", bg: "var(--greenDim)" };
  if (d.meter.crossed) return { label: "Breakthrough", color: "var(--green)", bg: "var(--greenDim)" };
  if (d.hasFinding || d.meter.hasMetric) return { label: "Chasing record", color: "var(--text2)", bg: "var(--chip)" };
  return { label: "No result yet", color: "var(--text3)", bg: "var(--chip)" };
}

// The best-so-far · best-known · need triplet, rendered honestly. Missing pieces
// are simply omitted rather than faked.
function ScoreLine({ d }: { d: DiscoverySummary }) {
  const cells: { k: string; v: string; color?: string; strong?: boolean }[] = [];
  if (d.best != null) cells.push({ k: "best found", v: fmtMetric(d.best), strong: true });
  if (d.bestKnown != null)
    cells.push({ k: "best-known", v: fmtMetric(d.bestKnown) });
  if (d.need != null) cells.push({ k: "need", v: fmtMetric(d.need), color: "var(--text)" });
  else if (d.beatsBestKnown) cells.push({ k: "beats", v: "published best", color: "var(--green)" });

  if (!cells.length) return null;
  return (
    <div className="flex flex-wrap items-baseline gap-x-[18px] gap-y-[4px]">
      {cells.map((c) => (
        <span key={c.k} className="flex items-baseline gap-[6px]">
          <span
            className="font-mono text-[17px] font-semibold leading-none tabular-nums"
            style={{ color: c.color ?? (c.strong ? "var(--text)" : "var(--text2)") }}
          >
            {c.v}
          </span>
          <span className="font-mono text-[9.5px] uppercase tracking-[0.06em] text-ink-3">{c.k}</span>
        </span>
      ))}
    </div>
  );
}

export default function DiscoveryHero({ discovery }: { discovery: DiscoverySummary }) {
  const d = discovery;
  const pill = statePill(d);
  const record = d.certified || d.beatsBestKnown;
  const m = d.meter;

  // Nothing to say yet at all — keep it honest and calm, not empty.
  const barren = !d.hasFinding && !m.hasMetric && d.best == null;

  return (
    <section
      className="mb-[16px] overflow-hidden rounded-[12px] border"
      style={{
        borderColor: record ? "var(--green)" : "var(--border)",
        background: record ? "var(--greenDim)" : "var(--railBg)",
      }}
      aria-label="Discovery status"
    >
      <div className="px-[16px] py-[13px]">
        <div className="mb-[9px] flex items-center gap-[9px]">
          <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-3">
            Discovery
          </span>
          <span
            className="rounded-full px-[8px] py-[2px] font-mono text-[9.5px] font-semibold uppercase tracking-[0.06em]"
            style={{ color: pill.color, background: pill.bg }}
          >
            {pill.label}
          </span>
          {d.metricName && (
            <span className="ml-auto font-mono text-[10px] text-ink-4">{d.metricName}</span>
          )}
        </div>

        {barren ? (
          <div className="text-[12.5px] leading-relaxed text-ink-2">
            No result yet — the campaign is building its prior, measuring the baseline, and
            dispatching its first hypotheses. The best-so-far will appear here the moment a
            sub-agent reports a measured result.
          </div>
        ) : (
          <>
            <ScoreLine d={d} />

            {d.statement && (
              <div className="mt-[9px] text-[12.5px] leading-[1.5] text-ink-2">{d.statement}</div>
            )}

            {/* one-line distance-to-breakthrough read, complementing the HUD meter */}
            {m.hasMetric && m.thresholdPct > 0 && (
              <div className="mt-[8px] font-mono text-[10.5px] text-ink-3">
                {m.crossed ? (
                  <span className="text-pos">
                    Crossed the +{m.thresholdPct}% breakthrough threshold
                    {m.improvementPct != null ? ` (${fmtPct(m.improvementPct)})` : ""}.
                  </span>
                ) : (
                  <span>
                    {Math.round(m.progress * 100)}% of the way to a +{m.thresholdPct}% breakthrough
                    {m.improvementPct != null ? ` · currently ${fmtPct(m.improvementPct)}` : ""}
                  </span>
                )}
              </div>
            )}

            {/* certified record → surface the certification + witness inline */}
            {(d.checks || d.certified != null) && (
              <div className="mt-[11px]">
                <CertChecks checks={d.checks} certified={d.certified} />
              </div>
            )}
            {d.witness != null && (
              <div className="mt-[9px]">
                <WitnessBlock witness={d.witness} />
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
