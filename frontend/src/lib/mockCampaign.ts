// Offline fixtures for `/campaign/demo` — a synthetic but realistic multi-round
// campaign event log + snapshot. Used to develop and sanity-check the live UI
// (grouped narrative, workers, background tasks) without a running backend. Only
// reached when the route id is the literal "demo"; real campaigns are untouched.

import type { CampaignState, PropabEvent } from "../types";

let seq = 0;
const T0 = Date.now() - 1000 * 60 * 22; // started 22 min ago
function ev(
  offsetSec: number,
  source: string,
  event_type: string,
  step: string,
  payload: Record<string, any>,
  hypothesis_id: string | null = null,
): PropabEvent {
  return {
    event_id: `demo-${seq++}`,
    session_id: "demo",
    timestamp: new Date(T0 + offsetSec * 1000).toISOString(),
    source,
    event_type,
    step,
    payload,
    parent_event_id: null,
    hypothesis_id,
  };
}

const HYPS: Record<string, { text: string; verdict: string; conf: number }> = {
  h1a: { text: "Feature scaling on income improves LOFO R² by >5%", verdict: "confirmed", conf: 0.91 },
  h1b: { text: "Dropping collinear age/tenure pair reduces variance", verdict: "refuted", conf: 0.88 },
  h1c: { text: "Target encoding on region beats one-hot", verdict: "inconclusive", conf: 0.41 },
  h2a: { text: "Interaction term income×region captures nonlinearity", verdict: "confirmed", conf: 0.86 },
  h2b: { text: "Log-transform of balance stabilizes residuals", verdict: "inconclusive", conf: 0.5 },
  h3a: { text: "Gradient-boosted depth-4 dominates linear baseline", verdict: "running", conf: 0 },
  h3b: { text: "Monotonic constraint on tenure aids generalization", verdict: "running", conf: 0 },
};

function workerEvents(id: string, startSec: number, done: boolean, round: number): PropabEvent[] {
  const h = HYPS[id];
  const out: PropabEvent[] = [];
  // agent.* events now carry an authoritative `round`; llm.prompt/response pair
  // by a shared `call_id`; llm.response carries duration_ms + tokens_in/out.
  const cid1 = `demo-call-${id}-1`;
  out.push(ev(startSec, "worker", "agent.started", `experiment.${id}`, { hypothesis_id: id, text: h.text, round }, id));
  out.push(ev(startSec + 3, "worker", "agent.step_started", `experiment.${id}.step_0`, { tool: "load_dataset", source: "heuristic_seed" }, id));
  out.push(ev(startSec + 4, "worker", "tool.called", `experiment.${id}.step_0`, { tool: "load_dataset" }, id));
  out.push(ev(startSec + 6, "worker", "tool.result", `experiment.${id}.step_0`, { tool: "load_dataset" }, id));
  out.push(ev(startSec + 8, "worker", "llm.prompt", `llm.decide_next`, { purpose: "decide_next", model: "claude-opus", call_id: cid1 }, id));
  out.push(ev(startSec + 11, "worker", "llm.response", `llm.decide_next`, { purpose: "decide_next", model: "claude-opus", call_id: cid1, duration_ms: 2600 + Math.round(h.conf * 900), tokens_in: 1800, tokens_out: 420 }, id));
  out.push(ev(startSec + 12, "worker", "agent.step_started", `experiment.${id}.step_1`, { action: "code", reasoning: "Fit model and measure LOFO R² against the baseline split." }, id));
  out.push(
    ev(startSec + 13, "worker", "code.generated", `experiment.${id}.step_1`, {
      code: `import pandas as pd\nfrom sklearn.linear_model import Ridge\n\ndf = load()\nX, y = features(df), target(df)\nmodel = Ridge(alpha=1.0).fit(X_train, y_train)\nr2 = lofo_r2(model, X, y)\nprint({"sandbox": "ok", "lofo_r2": r2})`,
      rewrite_after_timeout: false,
    }, id),
  );
  if (done) {
    out.push(ev(startSec + 16, "worker", "code.result", `experiment.${id}.step_1`, { stdout_json: { sandbox: "ok", lofo_r2: 0.7 + h.conf * 0.1 }, attempt: 1 }, id));
    out.push(ev(startSec + 18, "worker", "agent.step_started", `experiment.${id}.step_2`, { action: "stop", reasoning: "Significance gate satisfied." }, id));
    out.push(
      ev(startSec + 20, "worker", "agent.completed", `experiment.${id}.complete`, {
        verdict: h.verdict,
        confidence: h.conf,
        sig_gate_passed: h.verdict !== "inconclusive",
        mean_r2: 0.7 + h.conf * 0.1,
        round,
      }, id),
    );
  } else {
    // leave it mid-flight: code submitted but no result yet (an in-flight task)
    out.push(ev(startSec + 14, "worker", "code.submitted", `experiment.${id}.step_1`, { timeout_sec: 20, execution: "inline_stub", attempt: 1 }, id));
    // a heartbeat + an unpaired llm.prompt (in-flight LLM task, exact call_id)
    out.push(ev(startSec + 15, "worker", "agent.progress", `experiment.${id}.progress`, { hypothesis_id: id, round, alive: true, heartbeat_seq: 1 }, id));
    out.push(ev(startSec + 16, "worker", "llm.prompt", `llm.decide_next`, { purpose: "decide_next", model: "claude-opus", call_id: `demo-call-${id}-2` }, id));
  }
  return out;
}

export function buildDemoEvents(): PropabEvent[] {
  seq = 0;
  const e: PropabEvent[] = [];
  e.push(ev(0, "orchestrator", "campaign.started", "campaign.start", { question: DEMO_QUESTION }));
  e.push(ev(2, "orchestrator", "campaign.progress", "campaign.phase", { phase: "prior_build", detail: "Building literature prior (LLM + retrieval) over 41 papers." }));
  e.push(ev(4, "orchestrator", "llm.prompt", "llm.prior", { purpose: "prior_synthesis", model: "claude-opus", call_id: "demo-prior" }));
  e.push(ev(30, "orchestrator", "llm.response", "llm.prior", { purpose: "prior_synthesis", model: "claude-opus", call_id: "demo-prior", duration_ms: 26000, tokens_in: 14200, tokens_out: 3100 }));
  // Orchestrator narrates its opening moves (redesign §5 orchestrator.* events).
  e.push(ev(34, "orchestrator", "orchestrator.literature", "orchestrator.literature", { decision: "reviewed literature", detail: "reviewed the literature — 5 established fact(s), 3 open gap(s), 1 contested claim(s)", established_facts: 5, open_gaps: 3, contested_claims: 1, key_papers: 41, evidence_status: "mixed" }));
  e.push(ev(60, "orchestrator", "campaign.baseline_measured", "campaign.baseline_measured", { baseline_metric: 0.612 }));
  e.push(ev(62, "orchestrator", "orchestrator.reasoning", "orchestrator.reasoning", { decision: "baseline measured", detail: "baseline LOFO R² = 0.612 on the held-out split — the bar every hypothesis must clear." }));

  // Round 1
  e.push(ev(70, "orchestrator", "round.started", "round.1.start", { round: 1, round_id: "r1" }));
  e.push(ev(72, "orchestrator", "hypothesis.generated", "hypothesis.generate", { hypotheses: [{}, {}, {}] }));
  e.push(ev(73, "orchestrator", "orchestrator.reasoning", "orchestrator.reasoning", { decision: "seed hypotheses", detail: "drafted 3 falsifiable seed hypotheses spanning feature scaling, collinearity, and categorical encoding.", generation: 1, count: 3 }));
  for (const [i, id] of ["h1a", "h1b", "h1c"].entries())
    e.push(ev(74 + i, "orchestrator", "orchestrator.hypothesis_written", "orchestrator.hypothesis_written", { node_id: id, parent_id: null, text: HYPS[id].text, kind: "seed", generation: 1 }));
  for (const [i, id] of ["h1a", "h1b", "h1c"].entries()) e.push(...workerEvents(id, 80 + i * 40, true, 1));
  // Orchestrator judges each returned result centrally (redesign §3.5).
  e.push(ev(224, "orchestrator", "orchestrator.decision", "orchestrator.decision", { node_id: "h1a", hypothesis_text: HYPS.h1a.text, verdict: "confirmed", effective_verdict: "confirmed", worker_verdict: "confirmed", downgraded: false, action: "deepen", why: "LOFO R² rose to 0.68 with a permutation null p = 0.004 — a real, significant lift.", metric_name: "LOFO R²", metric_value: 0.681, null_p: 0.004, inconclusive_reason: null }));
  e.push(ev(226, "orchestrator", "orchestrator.decision", "orchestrator.decision", { node_id: "h1b", hypothesis_text: HYPS.h1b.text, verdict: "refuted", effective_verdict: "refuted", worker_verdict: "refuted", downgraded: false, action: "drop", why: "Dropping the collinear pair left R² unchanged (null p = 0.61) — no support.", metric_name: "LOFO R²", metric_value: 0.61, null_p: 0.61, inconclusive_reason: null }));
  e.push(ev(228, "orchestrator", "orchestrator.decision", "orchestrator.decision", { node_id: "h1c", hypothesis_text: HYPS.h1c.text, verdict: "inconclusive", effective_verdict: "inconclusive", worker_verdict: "confirmed", downgraded: true, action: "retune", why: "Worker read it as a win, but the gain sits inside the noise band (null p = 0.14) — not yet decisive.", metric_name: "LOFO R²", metric_value: 0.63, null_p: 0.14, inconclusive_reason: "effect within noise band" }));
  e.push(ev(230, "orchestrator", "synthesis.ledger_updated", "synthesis.ledger", { round: 1 }));
  e.push(ev(232, "orchestrator", "synthesis.breakthrough", "synthesis.breakthrough", { round: 1, finding: "Income scaling lifts R² to 0.68" }, "h1a"));
  // First-class discovery event: a new best-so-far (feeds the Discovery Hero).
  e.push(ev(233, "orchestrator", "finding.best_updated", "campaign.best_updated", { metric_name: "LOFO R²", best_metric: 0.681, previous_best: 0.612, baseline_metric: 0.612, direction: "higher_is_better", metric_value: 0.681 }, "h1a"));
  e.push(ev(236, "orchestrator", "round.completed", "round.1.complete", { round: 1, confirmed: 1, refuted: 1, inconclusive: 1, marginal_return: 0.42 }));

  // Round 2
  e.push(ev(240, "orchestrator", "round.started", "round.2.start", { round: 2, round_id: "r2" }));
  e.push(ev(241, "orchestrator", "orchestrator.reasoning", "orchestrator.reasoning", { decision: "synthesize follow-ups", detail: "the scaling result is promising — synthesizing 2 follow-up hypotheses to probe interactions and residual structure.", generation: 2, count: 2, source: "synthesis" }));
  e.push(ev(242, "orchestrator", "hypothesis.generated", "hypothesis.generate", { hypotheses: [{}, {}] }));
  for (const [i, id] of ["h2a", "h2b"].entries())
    e.push(ev(243 + i, "orchestrator", "orchestrator.hypothesis_written", "orchestrator.hypothesis_written", { node_id: id, parent_id: "h1a", text: HYPS[id].text, kind: "child", expansion_type: "deepen", generation: 2 }));
  for (const [i, id] of ["h2a", "h2b"].entries()) e.push(...workerEvents(id, 250 + i * 40, true, 2));
  // Orchestrator judges the round-2 results centrally — a deepen and a retune.
  e.push(ev(356, "orchestrator", "orchestrator.decision", "orchestrator.decision", { node_id: "h2a", hypothesis_text: HYPS.h2a.text, verdict: "confirmed", effective_verdict: "confirmed", worker_verdict: "confirmed", downgraded: false, action: "deepen", why: "The income×region interaction added a clean +0.03 lift with a permutation null p = 0.008 — the nonlinearity is real, so I'll deepen this line.", metric_name: "LOFO R²", metric_value: 0.712, null_p: 0.008, inconclusive_reason: null }));
  e.push(ev(358, "orchestrator", "orchestrator.decision", "orchestrator.decision", { node_id: "h2b", hypothesis_text: HYPS.h2b.text, verdict: "inconclusive", effective_verdict: "inconclusive", worker_verdict: "inconclusive", downgraded: false, action: "retune", why: "Log-transforming balance moved R² by less than the noise band (null p = 0.22) — worth a retuned second pass before I rule it out.", metric_name: "LOFO R²", metric_value: 0.688, null_p: 0.22, inconclusive_reason: "effect within noise band" }));
  e.push(ev(360, "orchestrator", "synthesis.ledger_updated", "synthesis.ledger", { round: 2 }));
  e.push(ev(362, "orchestrator", "round.completed", "round.2.complete", { round: 2, confirmed: 1, refuted: 0, inconclusive: 1, marginal_return: 0.19 }));

  // Round 3 — in flight
  e.push(ev(370, "orchestrator", "round.started", "round.3.start", { round: 3, round_id: "r3" }));
  e.push(ev(372, "orchestrator", "campaign.progress", "campaign.phase", { phase: "pipelined_sub_agents", detail: "Dispatching 2 sub-agents for round 3 verification." }));
  e.push(ev(374, "orchestrator", "hypothesis.generated", "hypothesis.generate", { hypotheses: [{}, {}, {}] }));
  // Orchestrator narrates its round-3 plan: chase the residual anomaly, deepen the
  // winning line, and place one lateral bet.
  e.push(ev(375, "orchestrator", "orchestrator.reasoning", "orchestrator.reasoning", { decision: "chase anomaly", detail: "Round 2 left an unexplained residual cluster in the high-balance tail — I'm chasing it with a monotonic-constraint probe before it biases the next round.", generation: 3 }));
  e.push(ev(376, "orchestrator", "orchestrator.hypothesis_written", "orchestrator.hypothesis_written", { node_id: "h3a", parent_id: "h2a", text: HYPS.h3a.text, kind: "child", expansion_type: "deepen", generation: 3 }));
  e.push(ev(377, "orchestrator", "orchestrator.hypothesis_written", "orchestrator.hypothesis_written", { node_id: "h3b", parent_id: "h2a", text: HYPS.h3b.text, kind: "child", expansion_type: "deepen", generation: 3 }));
  e.push(ev(378, "orchestrator", "orchestrator.hypothesis_written", "orchestrator.hypothesis_written", { node_id: "h3c", parent_id: null, text: "Lateral bet: quantile-binning balance may capture the tail nonlinearity a monotonic constraint misses.", kind: "lateral", generation: 3 }));
  e.push(...workerEvents("h3a", 380, false, 3));
  e.push(...workerEvents("h3b", 395, false, 3));
  // an orchestrator LLM call in flight (unpaired prompt → in-flight LLM task)
  e.push(ev(410, "orchestrator", "llm.prompt", "llm.rank", { purpose: "rank_hypotheses", model: "claude-opus", call_id: "demo-rank" }));

  return e.sort((a, b) => (a.timestamp < b.timestamp ? -1 : a.timestamp > b.timestamp ? 1 : 0));
}

const DEMO_QUESTION =
  "Which feature-engineering strategies most improve leave-one-feature-out R² on the churn dataset?";

export function buildDemoSnapshot(): CampaignState {
  const nodes: CampaignState["campaign"]["hypothesis_tree"]["nodes"] = {};
  Object.entries(HYPS).forEach(([id, h], i) => {
    nodes[id] = {
      id,
      text: h.text,
      parent_id: null,
      depth: 1,
      generation: i < 3 ? 1 : i < 5 ? 2 : 3,
      verdict: (h.verdict === "running" ? "pending" : h.verdict) as any,
      confidence: h.conf,
      children: [],
    };
  });
  return {
    campaign_id: "demo",
    campaign: {
      id: "demo",
      question: DEMO_QUESTION,
      status: "active",
      hypothesis_tree: { nodes, frontier: ["h3a", "h3b"], confirmed: ["h1a", "h2a"], exhausted: ["h1b"] },
      baseline_metric: 0.612,
      best_metric: 0.681,
      improvement_pct: 11.3,
      best_finding: { statement: "Income feature scaling lifts LOFO R² from 0.61 to 0.68." },
      breakthrough_criteria: {},
      compute_budget_seconds: 3600 * 3,
      compute_seconds_used: 60 * 22,
      started_at: new Date(T0).toISOString(),
      belief_state: {
        active_beliefs: [
          { statement: "Feature scaling on skewed monetary columns materially improves generalization.", confidence: "strong", supporting_nodes: ["h1a", "h2a"], contradicting_nodes: [], status: "strengthened", exhaustion_rounds: 0 },
          { statement: "Dropping collinear features helps.", confidence: "weak", supporting_nodes: [], contradicting_nodes: ["h1b"], status: "weakened", exhaustion_rounds: 1 },
          { statement: "Target encoding beats one-hot for high-cardinality region.", confidence: "unclear", supporting_nodes: ["h1c"], contradicting_nodes: [], status: "active", exhaustion_rounds: 0 },
        ],
      },
    },
    summary: {
      id: "demo",
      question: DEMO_QUESTION,
      status: "active",
      total_hypotheses: 7,
      total_confirmed: 2,
      baseline_metric: 0.612,
      best_metric: 0.681,
      improvement_pct: 11.3,
      elapsed_sec: 60 * 22,
      remaining_sec: 3600 * 3 - 60 * 22,
      breakthrough_threshold_pct: 15,
      tree: { total_nodes: 7, frontier_size: 2, confirmed_count: 2, exhausted_count: 1, max_depth: 3, verdict_counts: { confirmed: 2, refuted: 1, inconclusive: 2, pending: 2 } },
    },
    research_session: { id: "demo", question: DEMO_QUESTION, status: "running", stage: "experimentation" },
    event_counts_by_type: { "llm.prompt": 38, "tool.called": 22, "code.result": 9 },
  };
}
