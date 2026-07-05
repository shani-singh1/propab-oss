// Shared types mirroring the Propab backend contract (see design.md §14.1).

export type Verdict = "pending" | "confirmed" | "refuted" | "inconclusive";

export interface HypothesisNode {
  id: string;
  text: string;
  parent_id: string | null;
  depth: number;
  generation: number;
  verdict: Verdict;
  confidence: number;
  evidence_summary?: string | null;
  expansion_type?: string | null;
  children?: string[];
}

export interface HypothesisTree {
  nodes: Record<string, HypothesisNode>;
  frontier: string[];
  confirmed: string[];
  exhausted: string[];
}

export interface TreeSummary {
  total_nodes: number;
  frontier_size: number;
  confirmed_count: number;
  exhausted_count: number;
  max_depth: number;
  verdict_counts: Record<string, number>;
}

export interface CampaignSummary {
  id: string;
  question: string;
  status: string;
  total_hypotheses: number;
  total_confirmed: number;
  baseline_metric: number;
  best_metric: number;
  improvement_pct: number | null;
  elapsed_sec: number;
  remaining_sec: number;
  breakthrough_threshold_pct: number;
  tree: TreeSummary;
}

export type ConfidenceLevel = "strong" | "weak" | "unclear";
export type BeliefStatus = "active" | "strengthened" | "weakened" | "abandoned";

export interface BeliefObject {
  statement: string;
  confidence: ConfidenceLevel;
  supporting_nodes: string[];
  contradicting_nodes: string[];
  status: BeliefStatus;
  exhaustion_rounds: number;
}

export interface CampaignBeliefState {
  active_beliefs: BeliefObject[];
  [k: string]: unknown;
}

export interface CampaignState {
  campaign_id: string;
  campaign: {
    id: string;
    question: string;
    status: string;
    hypothesis_tree: HypothesisTree;
    baseline_metric: number;
    best_metric: number;
    improvement_pct: number;
    best_finding: Record<string, unknown> | null;
    breakthrough_criteria: Record<string, unknown>;
    compute_budget_seconds: number;
    compute_seconds_used: number;
    started_at: string;
    stop_reason?: string | null;
    belief_state?: CampaignBeliefState;
  };
  summary: CampaignSummary;
  research_session: {
    id: string;
    question: string;
    status: string;
    stage: string;
    created_at?: string;
    completed_at?: string;
  } | null;
  event_counts_by_type: Record<string, number>;
}

export interface CampaignListItem {
  id: string;
  question: string;
  status: string;
  baseline_metric: number;
  best_metric: number;
  improvement_pct: number | null;
  total_hypotheses: number;
  total_confirmed: number;
  compute_budget_seconds: number;
  compute_seconds_used: number;
  started_at: string | null;
  completed_at: string | null;
}

// Normalized event used across the UI (SSE and persisted log unified).
export interface PropabEvent {
  event_id: string;
  session_id: string;
  timestamp: string;
  source: string;
  event_type: string;
  step: string;
  payload: Record<string, any>;
  parent_event_id: string | null;
  hypothesis_id: string | null;
}

export interface PaperPayload {
  abstract_latex?: string;
  methods_latex?: string;
  results_latex?: string;
  pdf_url?: string;
  tex_url?: string;
  full_tex_chars?: number;
  figures_embedded?: number;
  [k: string]: unknown;
}
