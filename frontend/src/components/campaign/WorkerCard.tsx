import { workerResult, type Worker } from "../../lib/model";
import { fmtElapsed, fmtMetric } from "../../lib/format";
import { ActivityStrip, StatusDot, TimerPill, workerAccent } from "./bits";
import WorkerDetail from "./WorkerDetail";

// The star of the right panel: every sub-agent gets its own self-contained card,
// not a dense row. A status accent (left edge + faint tint; running breathes),
// the hypothesis text (2-line clamp → full on expand), a live timer, a compact
// activity strip, the current step while running, and — on completion — the
// verdict, confidence, and (for math/coding domains) a metric-vs-best-known line
// plus witness size. Click to expand into the full trace + latest code.

// The result summary line, only meaningful once the worker has finished.
function ResultLine({ w }: { w: Worker }) {
  const res = workerResult(w);
  if (!res.hasMetric && res.sigGatePassed == null) return null;
  const dirSym = res.direction === "higher_is_better" ? "↑" : "↓";
  return (
    <div className="mt-[9px] flex flex-wrap items-center gap-x-[10px] gap-y-[4px] rounded-[7px] bg-chip px-[9px] py-[7px] font-mono text-[10.5px] leading-none">
      {res.metricValue != null && (
        <span className="flex items-center gap-[5px]">
          <span className="text-ink-3">{res.metricName || "metric"}</span>
          <span className="font-semibold text-ink">{fmtMetric(res.metricValue)}</span>
          {res.bestKnown != null && (
            <span
              className="text-ink-3"
              style={res.beatsBestKnown ? { color: "var(--green)" } : undefined}
            >
              {dirSym} best-known {fmtMetric(res.bestKnown)}
              {res.beatsBestKnown ? " · beats" : ""}
            </span>
          )}
        </span>
      )}
      {res.witnessSize != null && (
        <span className="flex items-center gap-[5px]">
          <span className="text-ink-3">witness</span>
          <span className="font-semibold text-ink">{res.witnessSize}</span>
        </span>
      )}
      {res.sigGatePassed != null && (
        <span
          className="ml-auto flex items-center gap-[4px]"
          style={{ color: res.sigGatePassed ? "var(--green)" : "var(--text3)" }}
        >
          {res.sigGatePassed ? "✓ verified" : "gate not met"}
        </span>
      )}
    </div>
  );
}

export default function WorkerCard({
  w,
  now,
  open,
  onToggle,
}: {
  w: Worker;
  now: number;
  open: boolean;
  onToggle: () => void;
}) {
  const a = workerAccent(w.status);
  const running = w.status === "running";
  const elapsed = fmtElapsed(w.startedAt, running ? now : w.endedAt);
  const lastStep = w.steps[w.steps.length - 1];

  return (
    <div className="px-[12px] py-[6px]">
      <div
        className="overflow-hidden rounded-[11px] border border-edge"
        style={{ background: a.tint === "transparent" ? "var(--rightBg)" : a.tint }}
      >
        <div className="flex">
          {/* status accent edge */}
          <span
            aria-hidden
            className={`w-[3px] shrink-0 ${a.breathe ? "animate-ppulse motion-reduce:animate-none" : ""}`}
            style={{ background: a.edge }}
          />
          <div className="min-w-0 flex-1">
            <button
              onClick={onToggle}
              aria-expanded={open}
              className="pp-row block w-full px-[13px] py-[11px] text-left"
            >
              {/* header: status + timer */}
              <div className="mb-[7px] flex items-center gap-[8px]">
                <StatusDot color={a.edge} live={a.breathe} size={7} />
                <span
                  className="font-mono text-[10px] font-semibold uppercase tracking-[0.08em]"
                  style={{ color: a.labelColor }}
                >
                  {a.label}
                </span>
                {w.confidence != null && !running && (
                  <span className="font-mono text-[10px] text-ink-3">
                    conf {w.confidence.toFixed(2)}
                  </span>
                )}
                {w.round != null && (
                  <span className="font-mono text-[10px] text-ink-4">R{w.round}</span>
                )}
                <span className="ml-auto flex items-center gap-[7px]">
                  <TimerPill text={elapsed} live={running} />
                  <span className="text-[11px] text-ink-4">{open ? "▾" : "▸"}</span>
                </span>
              </div>

              {/* hypothesis text: 2-line clamp collapsed, full when open */}
              <div
                className={`text-[12.5px] font-medium leading-[1.4] text-ink ${open ? "" : "line-clamp-2"}`}
              >
                {w.text || `Hypothesis ${w.shortId}`}
              </div>

              {/* current step while running */}
              {running && lastStep && (
                <div className="mt-[7px] flex items-center gap-[6px] font-mono text-[10.5px] leading-none text-ink-3">
                  <span className="inline-flex gap-[2px]" aria-hidden>
                    <span className="h-[3px] w-[3px] animate-pdots rounded-full bg-ink-3 [animation-delay:0ms]" />
                    <span className="h-[3px] w-[3px] animate-pdots rounded-full bg-ink-3 [animation-delay:200ms]" />
                    <span className="h-[3px] w-[3px] animate-pdots rounded-full bg-ink-3 [animation-delay:400ms]" />
                  </span>
                  <span className="truncate">{lastStep.label}</span>
                </div>
              )}

              {/* activity strip */}
              <div className="mt-[9px]">
                <ActivityStrip w={w} />
              </div>

              {/* result summary on completion */}
              {!running && <ResultLine w={w} />}
            </button>

            {open && <WorkerDetail w={w} now={now} />}
          </div>
        </div>
      </div>
    </div>
  );
}
