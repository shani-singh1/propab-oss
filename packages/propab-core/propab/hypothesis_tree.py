"""
Hypothesis tree — grows the hypothesis space as evidence accumulates.

Confirmed nodes expand into boundary/mechanistic/generalization children.
Refuted nodes generate alternative hypotheses.
Inconclusive nodes get better-powered retests.

This is the mechanism that allows a campaign to reach 18,000+ hypotheses
without generating them all upfront.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


# ── Expansion type labels ───────────────────────────────────────────────────

EXPANSION_CONFIRMED = "confirmed"
EXPANSION_REFUTED = "refuted"
EXPANSION_INCONCLUSIVE = "inconclusive"


# ── LLM expansion prompts ───────────────────────────────────────────────────

_EXPANSION_INSTRUCTIONS: dict[str, str] = {
    EXPANSION_CONFIRMED: """\
1. Probe BOUNDARY CONDITIONS: where does this effect break down?
2. Ask WHY: what mechanism drives this effect mechanistically?
3. Ask HOW FAR: does this generalize to other architectures/datasets?
4. Test whether this finding enables a downstream improvement.""",
    EXPANSION_REFUTED: """\
1. Generate ALTERNATIVE approaches: what else could achieve the goal?
2. Ask WHAT IF: change the variable that caused refutation.
3. Reframe the question so it is still testable.""",
    EXPANSION_INCONCLUSIVE: """\
1. Design a BETTER-POWERED version of the same experiment.
2. Add more replications, tighter controls, or a direct baseline comparison.
3. Reduce confounds by isolating the variable of interest.""",
}

_EXPAND_PROMPT_TEMPLATE = """\
This hypothesis was {verdict} with confidence {confidence:.2f}.

Hypothesis: {parent_text}

Finding: {evidence_summary}
{mechanism_block}
Depth in tree: {depth} (root = 0). Hypotheses at depth >= 8 will not be expanded further.

Generate 3-5 child hypotheses that:
{expansion_instructions}

Rules:
- Do not repeat the parent hypothesis.
- Each child must be MORE specific than the parent, not broader.
- Each child must include a one-sentence test_methodology naming specific tools.
- Do not generate hypotheses that contradict confirmed findings from other branches.

Return a JSON array only (no prose):
[{{
  "id": "<short unique slug>",
  "text": "...",
  "test_methodology": "...",
  "expansion_type": "boundary|mechanistic|generalization|alternative|retest"
}}]
"""


# ── Core data structures ─────────────────────────────────────────────────────

@dataclass
class HypothesisNode:
    """A single node in the hypothesis tree."""

    id: str
    text: str
    parent_id: str | None           # None for seed hypotheses
    depth: int                      # distance from seed (root depth = 0)
    verdict: str = "pending"        # pending | confirmed | refuted | inconclusive
    confidence: float = 0.0
    children: list[str] = field(default_factory=list)
    generation: int = 0             # which campaign round spawned this
    evidence_summary: str | None = None
    expansion_type: str | None = None  # boundary | mechanistic | generalization | alternative | retest
    # fixes.md P0.1 / P4.1 diagnostics
    claim_type: str | None = None
    verification_method: str | None = None
    theme_id: str | None = None
    question_relevance_score: float | None = None
    frontier_score: float | None = None
    expansion_reason: str | None = None
    mechanism: str | None = None
    finding: dict[str, Any] | None = None
    inconclusive_expansions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "children": self.children,
            "generation": self.generation,
            "evidence_summary": self.evidence_summary,
            "expansion_type": self.expansion_type,
            "claim_type": self.claim_type,
            "verification_method": self.verification_method,
            "theme_id": self.theme_id,
            "question_relevance_score": self.question_relevance_score,
            "frontier_score": self.frontier_score,
            "expansion_reason": self.expansion_reason,
            "mechanism": self.mechanism,
            "finding": self.finding,
            "inconclusive_expansions": self.inconclusive_expansions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypothesisNode":
        return cls(
            id=data["id"],
            text=data["text"],
            parent_id=data.get("parent_id"),
            depth=data.get("depth", 0),
            verdict=data.get("verdict", "pending"),
            confidence=data.get("confidence", 0.0),
            children=data.get("children", []),
            generation=data.get("generation", 0),
            evidence_summary=data.get("evidence_summary"),
            expansion_type=data.get("expansion_type"),
            claim_type=data.get("claim_type"),
            verification_method=data.get("verification_method"),
            theme_id=data.get("theme_id"),
            question_relevance_score=data.get("question_relevance_score"),
            frontier_score=data.get("frontier_score"),
            expansion_reason=data.get("expansion_reason"),
            mechanism=data.get("mechanism"),
            finding=data.get("finding"),
            inconclusive_expansions=int(data.get("inconclusive_expansions") or 0),
        )


@dataclass
class FrontierScoringContext:
    """Campaign context for information-gain frontier scoring (fixes.md P1.1)."""

    question: str = ""
    prior_snippets: list[str] = field(default_factory=list)
    theme_saturation_penalty: float = 0.15


@dataclass
class HypothesisTree:
    """
    Directed tree of hypotheses grown from seed hypotheses.

    The frontier holds IDs of nodes that are ready to be dispatched for
    experimentation.  After a result comes back, update_node() records the
    verdict; expand() generates children for confirmed nodes and adds them to
    the frontier; prune() marks dead-end branches exhausted.
    """

    nodes: dict[str, HypothesisNode] = field(default_factory=dict)
    frontier: list[str] = field(default_factory=list)     # ready to test
    confirmed: list[str] = field(default_factory=list)    # confirmed node IDs
    exhausted: list[str] = field(default_factory=list)    # done, no more children
    _generation: int = field(default=0, repr=False)        # current campaign generation
    _scoring_context: FrontierScoringContext = field(default_factory=FrontierScoringContext, repr=False)

    # ── Mutation ─────────────────────────────────────────────────────────────

    def set_scoring_context(
        self,
        question: str,
        prior_snippets: list[str] | None = None,
        *,
        theme_saturation_penalty: float = 0.15,
    ) -> None:
        self._scoring_context = FrontierScoringContext(
            question=question,
            prior_snippets=list(prior_snippets or []),
            theme_saturation_penalty=theme_saturation_penalty,
        )

    def theme_counts(self) -> dict[str, int]:
        """Theme histogram for saturation control (fixes.md P1.4)."""
        counts: dict[str, int] = {}
        for node in self.nodes.values():
            tid = node.theme_id or "general"
            counts[tid] = counts.get(tid, 0) + 1
        return counts

    def add_seeds(self, hypotheses: list[dict[str, Any]], generation: int = 0) -> list[HypothesisNode]:
        """Add seed (root) hypotheses and push them to the frontier.

        Node identity is always tree-owned (a fresh UUID), never the LLM-supplied ``id``.
        Re-seeding when the frontier empties otherwise reuses ids like ``h1``..``h5`` and
        silently overwrites earlier nodes — including confirmed ones, which get reset to
        pending and re-dispatched, corrupting the findings ledger.
        """
        added: list[HypothesisNode] = []
        for h in hypotheses:
            node = HypothesisNode(
                id=str(uuid4()),
                text=h["text"],
                parent_id=None,
                depth=0,
                generation=generation,
                expansion_type=None,
                theme_id=h.get("theme_id"),
                question_relevance_score=h.get("question_relevance_score"),
            )
            self.nodes[node.id] = node
            if node.id not in self.frontier:
                self.frontier.append(node.id)
            added.append(node)
        return added

    def add_to_frontier(self, nodes: list[HypothesisNode]) -> None:
        """Push generated child nodes into the tree and frontier."""
        for node in nodes:
            self.nodes[node.id] = node
            if node.id not in self.frontier and node.id not in self.exhausted:
                self.frontier.append(node.id)
            # Register as child of parent
            if node.parent_id and node.parent_id in self.nodes:
                parent = self.nodes[node.parent_id]
                if node.id not in parent.children:
                    parent.children.append(node.id)

    def update_node(
        self,
        node_id: str,
        verdict: str,
        confidence: float,
        evidence_summary: str | None = None,
    ) -> bool:
        """Record experiment result on a node and remove it from the frontier.

        Returns False if ``node_id`` is not in the tree (caller should not count
        this as a completed campaign hypothesis).
        """
        node = self.nodes.get(node_id)
        if node is None:
            return False
        node.verdict = verdict
        node.confidence = confidence
        if evidence_summary:
            node.evidence_summary = evidence_summary

        if node_id in self.frontier:
            self.frontier.remove(node_id)

        if verdict == "confirmed":
            if node_id not in self.confirmed:
                self.confirmed.append(node_id)
        else:
            # A node re-evaluated to a non-confirmed verdict must leave the confirmed
            # list, or ``confirmed_count`` (len of list) diverges from the verdict scan.
            if node_id in self.confirmed:
                self.confirmed.remove(node_id)
            self._maybe_exhaust(node_id)
        return True

    def _maybe_exhaust(self, node_id: str) -> None:
        """Mark a node exhausted if pruning rules apply."""
        node = self.nodes.get(node_id)
        if node is None:
            return
        if node.verdict in ("refuted", "inconclusive"):
            # Check if already retested once (inconclusive twice = exhausted)
            if node.expansion_type == "retest":
                if node_id not in self.exhausted:
                    self.exhausted.append(node_id)
                return
        if node.depth >= 8:
            if node_id not in self.exhausted:
                self.exhausted.append(node_id)

    # ── Expansion ────────────────────────────────────────────────────────────

    def build_expand_prompt(self, node_id: str) -> str | None:
        """
        Build the LLM prompt for expanding a node.  Returns None if the node
        should not be expanded (exhausted, too deep, already has children, etc.).
        """
        node = self.nodes.get(node_id)
        if node is None or node.verdict == "pending":
            return None
        if node.id in self.exhausted:
            return None
        if node.depth >= 8:
            return None
        if len(node.children) >= 5:
            return None

        verdict_key = node.verdict if node.verdict in _EXPANSION_INSTRUCTIONS else EXPANSION_INCONCLUSIVE
        mechanism_block = ""
        if node.mechanism:
            mechanism_block = f"\nExtracted mechanism: {node.mechanism}\n"
        return _EXPAND_PROMPT_TEMPLATE.format(
            verdict=node.verdict,
            confidence=node.confidence,
            parent_text=node.text,
            evidence_summary=(node.evidence_summary or "No evidence summary available."),
            mechanism_block=mechanism_block,
            depth=node.depth,
            expansion_instructions=_EXPANSION_INSTRUCTIONS[verdict_key],
        )

    def parse_expanded_nodes(
        self,
        node_id: str,
        llm_response: str,
        generation: int,
    ) -> list[HypothesisNode]:
        """
        Parse raw LLM JSON response into HypothesisNode objects.
        Returns empty list on any parse failure (caller may log and continue).
        """
        parent = self.nodes.get(node_id)
        if parent is None:
            return []
        try:
            raw = llm_response.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            items: list[dict] = json.loads(raw)
        except (json.JSONDecodeError, IndexError, ValueError):
            return []

        children: list[HypothesisNode] = []
        for item in items:
            if not isinstance(item, dict) or not item.get("text"):
                continue
            child = HypothesisNode(
                id=str(uuid4()),
                text=item["text"],
                parent_id=parent.id,
                depth=parent.depth + 1,
                verdict="pending",
                generation=generation,
                expansion_type=item.get("expansion_type"),
                theme_id=item.get("theme_id"),
                question_relevance_score=item.get("question_relevance_score"),
            )
            children.append(child)
        return children

    # ── Frontier selection ───────────────────────────────────────────────────

    def next_batch(self, size: int, strategy: str = "highest_expected_value") -> list[HypothesisNode]:
        """
        Return up to `size` nodes from the frontier.
        strategy='highest_expected_value' scores by lineage + depth + specificity.
        strategy='fifo' returns in insertion order.
        """
        candidates = [
            self.nodes[nid]
            for nid in self.frontier
            if nid in self.nodes and self.nodes[nid].verdict == "pending"
        ]
        if not candidates:
            return []

        if strategy == "highest_expected_value":
            candidates.sort(key=self._information_gain_score, reverse=True)
        return candidates[:size]

    def next_dispatch_candidate(
        self,
        exclude_ids: set[str] | frozenset[str] | None = None,
        *,
        strategy: str = "highest_expected_value",
    ) -> HypothesisNode | None:
        """
        Best next frontier node to dispatch, excluding ids already in-flight to Celery.

        Used for pipelined campaign dispatch so we never double-dispatch the same node
        while it remains on the frontier until ``update_node`` removes it.
        """
        ex: frozenset[str] = frozenset(exclude_ids or ())
        candidates = [
            self.nodes[nid]
            for nid in self.frontier
            if nid in self.nodes
            and self.nodes[nid].verdict == "pending"
            and nid not in ex
        ]
        if not candidates:
            return None
        if strategy == "highest_expected_value":
            candidates.sort(key=self._information_gain_score, reverse=True)
        return candidates[0]

    def _information_gain_score(self, node: HypothesisNode) -> float:
        """
        Expected information gain for frontier ranking (fixes.md P1.1).

        Components: question relevance, novelty (inverse theme saturation),
        evidence uncertainty from parent verdict, theme coverage bonus.
        """
        counts = self.theme_counts()
        theme_id = node.theme_id or "general"
        theme_count = counts.get(theme_id, 1)
        max_count = max(counts.values()) if counts else 1
        saturation = theme_count / max(max_count, 1)
        penalty = self._scoring_context.theme_saturation_penalty

        relevance = float(node.question_relevance_score if node.question_relevance_score is not None else 0.5)
        novelty = max(0.05, 1.0 - saturation * penalty * 3.0)

        parent = self.nodes.get(node.parent_id) if node.parent_id else None
        if parent is None:
            uncertainty = 0.55
        elif parent.verdict == "inconclusive":
            uncertainty = 0.85
        elif parent.verdict == "refuted":
            uncertainty = 0.70
        elif parent.verdict == "confirmed":
            uncertainty = 0.45
        else:
            uncertainty = 0.60

        coverage_bonus = max(0.0, 1.0 - min(1.0, theme_count / 12.0))
        depth_bonus = 1.0 / (1.0 + node.depth * 0.25)

        score = (
            0.30 * relevance
            + 0.25 * novelty
            + 0.25 * uncertainty
            + 0.10 * coverage_bonus
            + 0.10 * depth_bonus
        )
        node.frontier_score = round(score, 4)
        return score

    def _expected_value_score(self, node: HypothesisNode) -> float:
        """Legacy alias — delegates to information-gain scoring."""
        return self._information_gain_score(node)

    def expansion_passes_merit_gate(
        self,
        node_id: str,
        *,
        novelty_min: float = 0.25,
        info_gain_min: float = 0.30,
    ) -> tuple[bool, str]:
        """Expand only when novelty or information gain exceeds threshold (fixes.md P1.3)."""
        node = self.nodes.get(node_id)
        if node is None:
            return False, "missing_node"
        relevance = float(node.question_relevance_score if node.question_relevance_score is not None else 0.0)
        info_gain = self._information_gain_score(node)
        if relevance >= novelty_min or info_gain >= info_gain_min:
            return True, f"relevance={relevance:.3f},info_gain={info_gain:.3f}"
        return False, f"merit_gate_failed:relevance={relevance:.3f},info_gain={info_gain:.3f}"

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "frontier": self.frontier,
            "confirmed": self.confirmed,
            "exhausted": self.exhausted,
            "generation": self._generation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypothesisTree":
        tree = cls()
        tree.nodes = {k: HypothesisNode.from_dict(v) for k, v in data.get("nodes", {}).items()}
        tree.frontier = data.get("frontier", [])
        tree.confirmed = data.get("confirmed", [])
        tree.exhausted = data.get("exhausted", [])
        tree._generation = data.get("generation", 0)
        return tree

    # ── Summary helpers ──────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        verdicts: dict[str, int] = {}
        for node in self.nodes.values():
            verdicts[node.verdict] = verdicts.get(node.verdict, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "frontier_size": len(self.frontier),
            # Count by verdict (authoritative) rather than len(self.confirmed); the list is a
            # convenience index for lineage scoring and must never define the reported count.
            "confirmed_count": verdicts.get("confirmed", 0),
            "exhausted_count": len(self.exhausted),
            "max_depth": max((n.depth for n in self.nodes.values()), default=0),
            "verdict_counts": verdicts,
        }

    def confirmed_findings_text(self, max_n: int = 10) -> str:
        """Compact text of top confirmed findings for hypothesis generation prompts."""
        lines: list[str] = []
        for nid in self.confirmed[:max_n]:
            node = self.nodes.get(nid)
            if node:
                ev = (node.evidence_summary or "")[:200]
                lines.append(f"- [confirmed, depth={node.depth}] {node.text}. {ev}")
        return "\n".join(lines) if lines else "No confirmed findings yet."
