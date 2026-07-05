import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";

const label = "mb-2 text-[12px] font-semibold leading-none text-ink-2";
const fieldBox =
  "w-full rounded-[9px] border border-edge bg-rail px-[14px] py-3 text-[14px] leading-none text-ink outline-none focus:border-ink-3";

export default function NewCampaign() {
  const nav = useNavigate();
  const [question, setQuestion] = useState("");
  const [budget, setBudget] = useState(4);
  const [metric, setMetric] = useState("val_accuracy");
  const [threshold, setThreshold] = useState(5); // percent
  const [direction, setDirection] = useState("higher_is_better");
  const [minConf, setMinConf] = useState(0.85);
  const [minReps, setMinReps] = useState(3);
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const canLaunch = question.trim().length >= 8 && !submitting;

  const launch = async () => {
    if (!canLaunch) return;
    setSubmitting(true);
    setErr(null);
    try {
      const { campaign_id } = await api.createCampaign({
        question: question.trim(),
        compute_budget_hours: budget,
        breakthrough_criteria: {
          metric_name: metric.trim() || "val_accuracy",
          improvement_threshold: Math.max(0.001, threshold / 100),
          direction,
          min_confidence: minConf,
          min_replications: minReps,
        },
      });
      nav(`/campaign/${campaign_id}`);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
      setSubmitting(false);
    }
  };

  return (
    <main className="flex min-w-0 flex-1 flex-col bg-center">
      <div className="flex shrink-0 items-center gap-[10px] border-b border-line px-[26px] py-4">
        <span className="text-[16px] font-semibold leading-none text-ink">New campaign</span>
        <span className="text-[12px] font-medium leading-none text-ink-3">
          Define the question — Propab runs the science
        </span>
      </div>

      <div className="pp-scroll min-h-0 flex-1 overflow-y-auto p-[26px]">
        <div className="flex max-w-[560px] flex-col gap-6">
          {/* question */}
          <div>
            <div className={label}>Research question</div>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What molecular determinants govern pathological tau oligomerization?"
              className="min-h-[72px] w-full resize-none rounded-[9px] border border-edge bg-rail px-[14px] py-3 text-[14.5px] leading-[1.5] text-ink outline-none focus:border-ink-3"
            />
          </div>

          {/* metric objective */}
          <div>
            <div className={label}>Breakthrough metric</div>
            <div className="flex flex-wrap items-center gap-2">
              <input
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
                className="flex-1 rounded-[9px] border border-edge bg-rail px-[14px] py-3 font-mono text-[13px] leading-none text-ink outline-none focus:border-ink-3"
              />
              {(["higher_is_better", "lower_is_better"] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => setDirection(d)}
                  className="rounded-[20px] border px-[13px] py-[9px] text-[12px] font-medium leading-none"
                  style={
                    direction === d
                      ? { borderColor: "var(--text)", color: "var(--text)" }
                      : { borderColor: "var(--border)", color: "var(--text3)" }
                  }
                >
                  {d === "higher_is_better" ? "Higher ↑" : "Lower ↓"}
                </button>
              ))}
            </div>
          </div>

          {/* budget + threshold grid */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className={label}>Compute budget (hours)</div>
              <input
                type="number"
                min={0.1}
                max={168}
                step={0.5}
                value={budget}
                onChange={(e) => setBudget(Number(e.target.value))}
                className={fieldBox}
              />
            </div>
            <div>
              <div className={label}>Improvement threshold (%)</div>
              <input
                type="number"
                min={0.1}
                max={100}
                step={0.5}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className={fieldBox}
              />
            </div>
            <div>
              <div className={label}>Min confidence</div>
              <input
                type="number"
                min={0.5}
                max={1}
                step={0.05}
                value={minConf}
                onChange={(e) => setMinConf(Number(e.target.value))}
                className={fieldBox}
              />
            </div>
            <div>
              <div className={label}>Min replications</div>
              <input
                type="number"
                min={1}
                max={20}
                step={1}
                value={minReps}
                onChange={(e) => setMinReps(Number(e.target.value))}
                className={fieldBox}
              />
            </div>
          </div>

          {err && (
            <div
              className="rounded-[9px] border px-[14px] py-3 text-[12.5px] leading-relaxed"
              style={{ borderColor: "var(--red)", color: "var(--red)", background: "var(--redDim)" }}
            >
              {err}
            </div>
          )}

          <div className="flex gap-[10px] pt-0.5">
            <button
              onClick={launch}
              disabled={!canLaunch}
              className="rounded-lg px-5 py-[11px] text-[13px] font-semibold leading-none disabled:opacity-40"
              style={{ background: "var(--text)", color: "var(--centerBg)" }}
            >
              {submitting ? "Launching…" : "Launch campaign"}
            </button>
            <button
              onClick={() => nav("/")}
              className="rounded-lg border border-edge px-5 py-[11px] text-[13px] font-medium leading-none text-ink-2"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
