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
        )


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

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_seeds(self, hypotheses: list[dict[str, Any]], generation: int = 0) -> list[HypothesisNode]:
        """Add seed (root) hypotheses and push them to the frontier."""
        added: list[HypothesisNode] = []
        for h in hypotheses:
            raw_id = h.get("id")
            node = HypothesisNode(
                id=str(raw_id) if raw_id is not None else str(uuid4()),
                text=h["text"],
                parent_id=None,
                depth=0,
                generation=generation,
                expansion_type=None,
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
        return _EXPAND_PROMPT_TEMPLATE.format(
            verdict=node.verdict,
            confidence=node.confidence,
            parent_text=node.text,
            evidence_summary=(node.evidence_summary or "No evidence summary available."),
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
            candidates.sort(key=self._expected_value_score, reverse=True)
        return candidates[:size]

    def _expected_value_score(self, node: HypothesisNode) -> float:
        depth_score = 1.0 / (1.0 + node.depth * 0.3)
        lineage_score = 1.5 if (node.parent_id and node.parent_id in self.confirmed) else 1.0
        specificity_score = min(1.0, len(node.text) / 200.0)
        return depth_score * lineage_score * 0.5 + specificity_score * 0.5

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
            "confirmed_count": len(self.confirmed),
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
