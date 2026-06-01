import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { Card } from "../components/ui";

type Kind = "empirical" | "verification";

export default function NewCampaign() {
  const nav = useNavigate();
  const [question, setQuestion] = useState("");
  const [kind, setKind] = useState<Kind>("empirical");
  const [hours, setHours] = useState(4);
  const [metric, setMetric] = useState("val_accuracy");
  const [threshold, setThreshold] = useState(5);
  const [minConfidence, setMinConfidence] = useState(0.85);
  const [minReplications, setMinReplications] = useState(2);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onKind = (k: Kind) => {
    setKind(k);
    // Verification campaigns have no numeric training baseline; a non-ML metric name
    // tells the backend to skip baseline measurement and judge by deterministic checks.
    setMetric(k === "verification" ? "verified_instances" : "val_accuracy");
    setMinReplications(k === "verification" ? 2 : 3);
  };

  const submit = async () => {
    if (question.trim().length < 8) {
      setError("Please enter a research question (at least 8 characters).");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const { campaign_id } = await api.createCampaign({
        question: question.trim(),
        compute_budget_hours: hours,
        breakthrough_criteria: {
          metric_name: metric.trim() || "val_accuracy",
          improvement_threshold: Math.max(0.001, threshold / 100),
          direction: "higher_is_better",
          min_confidence: minConfidence,
          min_replications: minReplications,
        },
      });
      nav(`/campaign/${campaign_id}`);
    } catch (e: any) {
      setError(e?.message ?? String(e));
      setSubmitting(false);
    }
  };

  return (
    <div className="h-full overflow-y-auto scrollbar-thin">
      <div className="mx-auto max-w-2xl px-8 py-10">
        <h1 className="text-2xl font-semibold tracking-tight">New campaign</h1>
        <p className="mt-1 text-sm text-text-secondary">
          Pose a question Propab can attack computationally and verify.
        </p>

        <Card className="mt-6 space-y-6 p-6">
          <Field label="Research question">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={4}
              placeholder="e.g. Investigate the Erdős–Straus conjecture: for every integer n ≥ 2, can 4/n be written as 1/x + 1/y + 1/z?"
              className="w-full resize-none rounded-lg border border-border bg-bg px-3 py-2.5 text-sm text-text-primary outline-none transition focus:border-brand"
            />
          </Field>

          <Field label="Campaign type">
            <div className="grid grid-cols-2 gap-3">
              <KindCard
                active={kind === "empirical"}
                onClick={() => onKind("empirical")}
                title="Empirical"
                desc="Measure a metric and beat a baseline (ML, simulation, benchmarks)."
              />
              <KindCard
                active={kind === "verification"}
                onClick={() => onKind("verification")}
                title="Verification"
                desc="Exactly-checkable claims (number theory, combinatorics, constructions)."
              />
            </div>
          </Field>

          <div className="grid grid-cols-2 gap-5">
            <Field label={`Time budget — ${hours}h`}>
              <input
                type="range"
                min={0.25}
                max={12}
                step={0.25}
                value={hours}
                onChange={(e) => setHours(parseFloat(e.target.value))}
                className="w-full accent-brand"
              />
            </Field>
            <Field label="Metric name">
              <input
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
                className="w-full rounded-lg border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-brand"
              />
            </Field>
          </div>

          <div className="grid grid-cols-3 gap-5">
            <Field label={`Improvement target — ${threshold}%`}>
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={threshold}
                onChange={(e) => setThreshold(parseInt(e.target.value))}
                className="w-full accent-brand"
              />
            </Field>
            <Field label={`Min confidence — ${minConfidence.toFixed(2)}`}>
              <input
                type="range"
                min={0.5}
                max={0.99}
                step={0.01}
                value={minConfidence}
                onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                className="w-full accent-brand"
              />
            </Field>
            <Field label={`Min replications — ${minReplications}`}>
              <input
                type="range"
                min={1}
                max={6}
                step={1}
                value={minReplications}
                onChange={(e) => setMinReplications(parseInt(e.target.value))}
                className="w-full accent-brand"
              />
            </Field>
          </div>

          {error && <div className="text-sm text-refuted">{error}</div>}

          <div className="flex justify-end gap-3 pt-2">
            <button
              onClick={() => nav("/")}
              className="rounded-lg border border-border px-4 py-2 text-sm text-text-secondary transition hover:bg-raised"
            >
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={submitting}
              className="rounded-lg bg-brand px-5 py-2 text-sm font-medium text-white transition hover:bg-brand/90 disabled:opacity-50"
            >
              {submitting ? "Starting…" : "Start campaign →"}
            </button>
          </div>
        </Card>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-text-muted">
        {label}
      </span>
      {children}
    </label>
  );
}

function KindCard({
  active,
  onClick,
  title,
  desc,
}: {
  active: boolean;
  onClick: () => void;
  title: string;
  desc: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg border p-3 text-left transition ${
        active ? "border-brand bg-brand/10" : "border-border bg-bg hover:bg-raised"
      }`}
    >
      <div className="text-sm font-semibold text-text-primary">{title}</div>
      <div className="mt-1 text-xs text-text-secondary">{desc}</div>
    </button>
  );
}
