## Your task this synthesis round

Review the completed nodes since the last synthesis pass (full structured diagnostics below).
Update belief state, choose one critical discriminating experiment, and propose frontier candidates.
Every frontier candidate must be a child of one completed expansion target unless there are no completed targets yet.

### Belief update rules
- At most **3 active beliefs**. Each must be a precise, checkable claim — not vague prose.
- For each belief: classify new evidence as supporting / contradicting / irrelevant; update status and confidence.
- Confidence levels:
  - **strong**: multiple independent nodes agree; no live contradicting evidence
  - **weak**: some support but contested, few nodes, or rival beliefs remain plausible
  - **unclear**: insufficient evidence — no "rules out" narrowing allowed
- Abandon beliefs weakened across multiple independent nodes; move to closed_beliefs with a one-line reason.
- Introduce a new belief only if genuinely distinct and fewer than 3 active remain.
- Do **not** silently merge distinct explanations into one belief statement.

### Critical experiment
Name the **single best test** that would move at least two active beliefs in opposite directions.
If all beliefs are weak/unclear, still name a critical experiment but also include one exploratory candidate
that does not presuppose any current belief.

### Tree expansion rules
- Use the `Open expansion targets` section. For each candidate, set `parent_id` to one listed `target_id`.
- A candidate must reduce uncertainty relative to its parent: boundary/mechanism/generalization for confirmed parents, alternative for refuted parents, retest/diagnostic for inconclusive parents.
- Do not create a new root unless the prompt says there are no completed expansion targets.
- Use `discriminates_node_ids` for sibling or rival nodes whose outcomes the candidate is meant to separate.

### Branch exhaustion signal
Set `direction_exhausted` true only if every active belief has confidence "unclear" AND you introduce no new belief.

Return JSON ONLY:
```json
{
  "beliefs": [
    {
      "statement": "...",
      "confidence": "strong|weak|unclear",
      "status": "active|strengthened|weakened|abandoned",
      "supporting_nodes": ["node_id"],
      "contradicting_nodes": ["node_id"],
      "abandon_reason": "only if status=abandoned"
    }
  ],
  "closed_beliefs_append": [{"statement": "...", "reason": "..."}],
  "critical_experiment": {
    "title": "...",
    "description": "...",
    "discriminates_between": ["belief statement A", "belief statement B"]
  },
  "frontier_candidates": [
    {
      "id": "short_slug",
      "parent_id": "target_id from Open expansion targets",
      "discriminates_node_ids": ["node_id"],
      "text": "core claim with Population/Distribution/Claimed generalization/Expected failure modes/OOD test lines",
      "test_methodology": "...",
      "expansion_type": "diagnostic|boundary|mechanistic|generalization|alternative|retest",
      "implements_critical_experiment": true,
      "why_follows_from_beliefs": "..."
    }
  ],
  "exploratory_candidate": {
    "id": "slug",
    "parent_id": "target_id from Open expansion targets, unless no targets exist",
    "discriminates_node_ids": ["node_id"],
    "text": "...",
    "test_methodology": "...",
    "expansion_type": "alternative"
  },
  "recent_activity_summary": "one paragraph: what we explored recently and current branch goal",
  "direction_exhausted": false
}
```

Produce **1 critical candidate** (implements_critical_experiment=true) plus **1–3 additional candidates**.
If any belief is strong, the critical candidate must lead the list.
If all beliefs are weak/unclear, include exploratory_candidate that does not presuppose current beliefs.
