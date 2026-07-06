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
import re
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

# KEPT: not used in the live campaign loop (Tier-2 synthesis replaces per-node
# expansion). Retained because build_expand_prompt/parse_expanded_nodes still
# encode the scope-gate-on-expansion behavior covered by
# tests/test_scoped_tree_expansion.py and tests/test_research_quality.py.
_EXPAND_PROMPT_TEMPLATE = """\
This hypothesis was {verdict} with confidence {confidence:.2f}.

Hypothesis: {parent_text}

## Structured failure diagnostics
verdict_reason: {verdict_reason}
inconclusive_reason: {inconclusive_reason}
failure_signature: {failure_signature}
artifact_gate: {artifact_gate_summary}

Finding: {evidence_summary}
{mechanism_block}
Depth in tree: {depth} (root = 0). Hypotheses at depth >= 8 will not be expanded further.

ParentScope (inherit unless you intentionally narrow or shift):
Population: {parent_population}
Distribution: {parent_distribution}
Claimed generalization: {parent_claimed_generalization}
Expected failure modes: {parent_expected_failure_modes}
OOD test: {parent_ood_test}

Generate 3-5 child hypotheses that:
{expansion_instructions}

Rules:
- Do not repeat the parent hypothesis.
- Each child must be MORE specific than the parent, not broader.
- Inherit the parent's scope unless you intentionally change it; if you change scope, state the new values explicitly.
- Every child MUST include ALL scope fields (same schema as seed generation):
  population, distribution, claimed_generalization, expected_failure_modes, ood_test
- Children missing any scope field are invalid and will be rejected.
- Each child must include a one-sentence test_methodology naming specific tools.
- Do not generate hypotheses that contradict confirmed findings from other branches.

Return a JSON array only (no prose):
[{{
  "id": "<short unique slug>",
  "text": "<core claim without scope labels>",
  "population": "...",
  "distribution": "...",
  "claimed_generalization": "...",
  "expected_failure_modes": "...",
  "ood_test": "...",
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
    node_role: str = "DISCOVERY"
    primary_theme: str | None = None
    secondary_themes: list[str] = field(default_factory=list)
    evidence_hash: str | None = None
    verification_hash: str | None = None
    replication_level: str | None = None
    inconclusive_reason: str | None = None
    failure_signature: str | None = None
    theme_confidence: float | None = None
    lineage_length: int | None = None
    # fixes.md P2 — preserve seed experiment metadata
    test_methodology: str | None = None
    feature_subset: list[str] = field(default_factory=list)
    mechanism_id: str | None = None
    claim_scope: dict[str, str] | None = None
    scope_delta: str | None = None

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
            "node_role": self.node_role,
            "primary_theme": self.primary_theme,
            "secondary_themes": self.secondary_themes,
            "evidence_hash": self.evidence_hash,
            "verification_hash": self.verification_hash,
            "replication_level": self.replication_level,
            "inconclusive_reason": self.inconclusive_reason,
            "failure_signature": self.failure_signature,
            "theme_confidence": self.theme_confidence,
            "lineage_length": self.lineage_length,
            "test_methodology": self.test_methodology,
            "feature_subset": self.feature_subset,
            "mechanism_id": self.mechanism_id,
            "claim_scope": self.claim_scope,
            "scope_delta": self.scope_delta,
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
            node_role=data.get("node_role", "DISCOVERY"),
            primary_theme=data.get("primary_theme"),
            secondary_themes=data.get("secondary_themes") or [],
            evidence_hash=data.get("evidence_hash"),
            verification_hash=data.get("verification_hash"),
            replication_level=data.get("replication_level"),
            inconclusive_reason=data.get("inconclusive_reason"),
            failure_signature=data.get("failure_signature"),
            theme_confidence=data.get("theme_confidence"),
            lineage_length=data.get("lineage_length"),
            test_methodology=data.get("test_methodology"),
            feature_subset=data.get("feature_subset") or [],
            mechanism_id=data.get("mechanism_id"),
            claim_scope=data.get("claim_scope"),
            scope_delta=data.get("scope_delta"),
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
    _used_evidence_hashes: set[str] = field(default_factory=set, repr=False)
    _used_confirmed_claim_keys: set[str] = field(default_factory=set, repr=False)
    finding_ledger: list[dict[str, Any]] = field(default_factory=list, repr=False)

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
        """Theme histogram for saturation control (fixes.md P2.2)."""
        counts: dict[str, int] = {}
        for node in self.nodes.values():
            tid = node.primary_theme or node.theme_id or "general"
            counts[tid] = counts.get(tid, 0) + 1
        return counts

    def register_evidence_hash(self, h: str) -> bool:
        """Return True if hash is new; False if reused (P0.2)."""
        if not h:
            return True
        if h in self._used_evidence_hashes:
            return False
        self._used_evidence_hashes.add(h)
        return True

    def register_confirmed_claim(self, claim_key: str) -> bool:
        """Return True if this claim text is new among confirmations; False if duplicate claim."""
        if not claim_key:
            return True
        if claim_key in self._used_confirmed_claim_keys:
            return False
        self._used_confirmed_claim_keys.add(claim_key)
        return True

    def add_seeds(self, hypotheses: list[dict[str, Any]], generation: int = 0) -> list[HypothesisNode]:
        """Add seed (root) hypotheses and push them to the frontier.

        Node identity is always tree-owned (a fresh UUID), never the LLM-supplied ``id``.
        Re-seeding when the frontier empties otherwise reuses ids like ``h1``..``h5`` and
        silently overwrites earlier nodes — including confirmed ones, which get reset to
        pending and re-dispatched, corrupting the findings ledger.
        """
        from propab.research_quality import extract_theme_vector, infer_node_role

        added: list[HypothesisNode] = []
        for h in hypotheses:
            primary, secondary, theme_conf = extract_theme_vector(h["text"])
            node = HypothesisNode(
                id=str(uuid4()),
                text=h["text"],
                parent_id=None,
                depth=0,
                generation=generation,
                expansion_type=None,
                node_role=infer_node_role(h["text"]),
                primary_theme=primary,
                secondary_themes=secondary,
                theme_id=h.get("theme_id") or primary,
                theme_confidence=theme_conf,
                lineage_length=1,
                question_relevance_score=h.get("question_relevance_score"),
                test_methodology=h.get("test_methodology"),
                feature_subset=list(h.get("feature_subset") or []),
                mechanism_id=h.get("mechanism_id"),
                claim_scope=h.get("claim_scope"),
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
        from propab.research_quality import NODE_ROLE_CONTROL, extract_theme_vector, infer_node_role, is_discovery_node

        node = self.nodes.get(node_id)
        if node is None:
            return False
        node.node_role = infer_node_role(node.text)
        if not node.primary_theme:
            primary, secondary, theme_conf = extract_theme_vector(node.text)
            node.primary_theme = primary
            node.secondary_themes = secondary
            node.theme_id = primary
            node.theme_confidence = theme_conf
        if node.lineage_length is None:
            node.lineage_length = self.lineage_length(node_id)
        node.verdict = verdict
        node.confidence = confidence
        if evidence_summary:
            node.evidence_summary = evidence_summary

        if node_id in self.frontier:
            self.frontier.remove(node_id)

        if verdict == "confirmed" and is_discovery_node(node):
            if node_id not in self.confirmed:
                self.confirmed.append(node_id)
        elif verdict == "confirmed" and getattr(node, "node_role", "") == NODE_ROLE_CONTROL:
            node.verdict = "inconclusive"
            node.inconclusive_reason = node.inconclusive_reason or "control_calibration"
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

    @staticmethod
    def _parse_evidence_blob(evidence_summary: str | None) -> dict[str, Any]:
        if not evidence_summary:
            return {}
        m = re.search(r"evidence=(\{.*?\});", evidence_summary)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _artifact_gate_summary(evidence_obj: dict[str, Any]) -> str:
        ag = evidence_obj.get("artifact_gate")
        if not isinstance(ag, dict):
            return "n/a"
        ranked = ag.get("ranked_artifacts") or []
        top = ranked[0] if ranked and isinstance(ranked[0], dict) else {}
        top_id = top.get("artifact_id") or "?"
        return f"{ag.get('verdict', '?')} — {top_id}: {ag.get('verdict_reason', '')}"[:300]

    def _failure_fields(self, node: HypothesisNode) -> dict[str, str]:
        ev = self._parse_evidence_blob(node.evidence_summary)
        return {
            "verdict_reason": str(ev.get("verdict_reason") or node.mechanism or "n/a"),
            "inconclusive_reason": str(node.inconclusive_reason or "n/a"),
            "failure_signature": str(node.failure_signature or "n/a"),
            "artifact_gate_summary": self._artifact_gate_summary(ev),
        }

    def _parent_scope_fields(self, node: HypothesisNode, *, question: str = "") -> dict[str, str]:
        from propab.scoped_claim import infer_domain_scope_template, parse_scope_from_methodology

        parent_scope = parse_scope_from_methodology(node.text, node.test_methodology)
        if parent_scope is None:
            parent_scope = infer_domain_scope_template(question)
        return {
            "parent_population": parent_scope.population,
            "parent_distribution": parent_scope.distribution,
            "parent_claimed_generalization": parent_scope.claimed_generalization,
            "parent_expected_failure_modes": parent_scope.expected_failure_modes,
            "parent_ood_test": parent_scope.ood_test,
        }

    def _expansion_gate(self, node_id: str) -> HypothesisNode | None:
        from propab.research_quality import is_discovery_node

        node = self.nodes.get(node_id)
        if node is None or node.verdict == "pending":
            return None
        if not is_discovery_node(node):
            return None
        if node.id in self.exhausted:
            return None
        if node.depth >= 8:
            return None
        if len(node.children) >= 5:
            return None
        return node

    def build_expand_prompt(
        self,
        node_id: str,
        *,
        question: str = "",
    ) -> str | None:
        """
        Build the LLM prompt for expanding a node.

        KEPT: not called from run_campaign_loop (Tier-2 synthesis replaces per-node
        expansion). Retained for the scope-gate-on-expansion tests.
        """
        node = self._expansion_gate(node_id)
        if node is None:
            return None

        scope = self._parent_scope_fields(node, question=question)
        fields = self._failure_fields(node)
        mechanism_block = ""
        if node.mechanism:
            mechanism_block = f"\nExtracted mechanism: {node.mechanism}\n"

        verdict_key = node.verdict if node.verdict in _EXPANSION_INSTRUCTIONS else EXPANSION_INCONCLUSIVE
        return _EXPAND_PROMPT_TEMPLATE.format(
            verdict=node.verdict,
            confidence=node.confidence,
            parent_text=node.text,
            evidence_summary=(node.evidence_summary or "No evidence summary available."),
            mechanism_block=mechanism_block,
            depth=node.depth,
            expansion_instructions=_EXPANSION_INSTRUCTIONS[verdict_key],
            **fields,
            **scope,
        )

    def parse_expanded_nodes(
        self,
        node_id: str,
        llm_response: str,
        generation: int,
        *,
        question: str = "",
    ) -> tuple[list[HypothesisNode], dict[str, Any]]:
        """
        Parse raw LLM JSON response into HypothesisNode objects.
        P3 — scope gate rejects children missing required fields before tree insertion.
        Returns (accepted_children, gate_metrics).
        """
        parent = self.nodes.get(node_id)
        empty_metrics: dict[str, Any] = {
            "n_children_generated": 0,
            "n_children_rejected": 0,
            "n_children_passed": 0,
            "scope_rejection_rate": 0.0,
            "rejection_reasons": {},
            "rejected": [],
        }
        if parent is None:
            return [], empty_metrics
        try:
            raw = llm_response.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            items: list[dict] = json.loads(raw)
        except (json.JSONDecodeError, IndexError, ValueError):
            return [], empty_metrics

        from collections import Counter

        from propab.research_quality import extract_theme_vector, infer_node_role
        from propab.scoped_claim import parse_scope_from_methodology, validate_expansion_child

        parent_scope = parse_scope_from_methodology(parent.text, parent.test_methodology)
        children: list[HypothesisNode] = []
        rejected: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict) or not item.get("text"):
                rejected.append({"text": str(item)[:120], "reason": "missing_text"})
                continue
            ok, enriched, reason = validate_expansion_child(
                item, parent=parent_scope, question=question,
            )
            if not ok:
                rejected.append({
                    "text": str(item.get("text", ""))[:200],
                    "reason": reason or "missing_scope",
                })
                continue
            primary, secondary, theme_conf = extract_theme_vector(enriched["text"])
            child = HypothesisNode(
                id=str(uuid4()),
                text=enriched["text"],
                parent_id=parent.id,
                depth=parent.depth + 1,
                verdict="pending",
                generation=generation,
                expansion_type=item.get("expansion_type"),
                node_role=infer_node_role(enriched["text"]),
                primary_theme=primary,
                secondary_themes=secondary,
                theme_id=item.get("theme_id") or primary,
                theme_confidence=theme_conf,
                lineage_length=self.lineage_length(parent.id) + 1 if parent.id else 1,
                question_relevance_score=item.get("question_relevance_score"),
                test_methodology=enriched.get("test_methodology"),
                feature_subset=list(item.get("feature_subset") or []),
                mechanism_id=item.get("mechanism_id"),
                claim_scope=enriched.get("claim_scope"),
                scope_delta=enriched.get("scope_delta"),
            )
            children.append(child)

        n_gen = len(items)
        n_rej = len(rejected)
        reasons = dict(Counter(r.get("reason", "?") for r in rejected))
        metrics = {
            "n_children_generated": n_gen,
            "n_children_rejected": n_rej,
            "n_children_passed": len(children),
            "scope_rejection_rate": round(n_rej / max(1, n_gen), 4),
            "rejection_reasons": reasons,
            "rejected": rejected[:12],
            "parent_id": node_id,
        }
        return children, metrics

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

    def lineage_length(self, node_id: str) -> int:
        """P1.1 — ancestry depth by parent chain (population lineage, not tree depth fiction)."""
        length = 0
        cur: str | None = node_id
        seen: set[str] = set()
        while cur and cur not in seen:
            seen.add(cur)
            length += 1
            n = self.nodes.get(cur)
            if n is None:
                break
            cur = n.parent_id
        return max(1, length)

    def generation_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for n in self.nodes.values():
            g = str(n.generation)
            hist[g] = hist.get(g, 0) + 1
        return hist

    def _information_gain_score(self, node: HypothesisNode) -> float:
        """
        Expected information gain × closure probability (fixes.md P1.2).

        Components: relevance, novelty, parent uncertainty, coverage; scaled by
        closure_probability so high-IG exploration balances decisive outcomes.
        """
        from propab.research_quality import estimate_closure_probability

        counts = self.theme_counts()
        theme_id = node.primary_theme or node.theme_id or "general"
        theme_count = counts.get(theme_id, 1)
        max_count = max(counts.values()) if counts else 1
        saturation = theme_count / max(max_count, 1)
        penalty = self._scoring_context.theme_saturation_penalty

        relevance = float(node.question_relevance_score if node.question_relevance_score is not None else 0.5)
        novelty = max(0.05, 1.0 - saturation * penalty * 3.0)

        parent = self.nodes.get(node.parent_id) if node.parent_id else None
        # Does this child NARROW its parent (the convergence move — a boundary,
        # mechanistic, or generalization refinement, or an explicit scope delta)?
        narrows = bool(node.scope_delta) or (node.expansion_type or "").lower() in {
            "boundary", "mechanistic", "generalization", "refinement",
        }
        deepening_confirmed = parent is not None and parent.verdict == "confirmed" and narrows

        if parent is None:
            uncertainty = 0.55
        elif parent.verdict == "inconclusive":
            uncertainty = 0.75
        elif parent.verdict == "refuted":
            uncertainty = 0.60
        elif parent.verdict == "confirmed":
            # A confirmed result with an OPEN boundary is the most valuable thing
            # to narrow next (convergence); a lateral re-test of an already-
            # confirmed result is not. Previously this was a flat 0.45, which
            # de-ranked deepening below inconclusive breadth and structurally
            # prevented confirmed lineages from growing (CHANGELOG / investigation
            # report §3.3 — "generation increases, depth does not").
            uncertainty = 0.80 if narrows else 0.45
        else:
            uncertainty = 0.55

        coverage_bonus = max(0.0, 1.0 - min(1.0, theme_count / 12.0))
        lineage = float(node.lineage_length or self.lineage_length(node.id))
        lineage_bonus = min(0.15, lineage * 0.03)

        # Convergence/exploit bonus: reward deepening a confirmed lineage so the
        # search turns a real finding into a narrower one (log-style convergence)
        # instead of spending every dispatch on fresh inconclusive breadth. Scales
        # with how deep the confirmed ancestry already is.
        exploit_bonus = 0.0
        if deepening_confirmed:
            conf_depth = self._confirmed_ancestry_depth(node)
            exploit_bonus = min(0.30, 0.12 + 0.06 * conf_depth)

        info_gain = (
            0.28 * relevance
            + 0.22 * novelty
            + 0.22 * uncertainty
            + 0.08 * coverage_bonus
            + 0.08 * min(1.0, lineage_bonus * 5)
            + exploit_bonus
        )
        closure = estimate_closure_probability(node, parent=parent)
        score = info_gain * closure
        node.frontier_score = round(score, 4)
        return score

    @staticmethod
    def _field(node: Any, name: str) -> Any:
        """Read a node field whether the node is a HypothesisNode or a raw dict.
        A convergence health metric must never raise into the caller (some paths,
        and tests, carry dict-shaped nodes)."""
        if isinstance(node, dict):
            return node.get(name)
        return getattr(node, name, None)

    def _confirmed_ancestry_depth(self, node: HypothesisNode) -> int:
        """Number of CONFIRMED ancestors up the parent chain — how deep a real
        finding has already been narrowed. Used to reward continued convergence."""
        depth = 0
        parent_id = self._field(node, "parent_id")
        cur = self.nodes.get(parent_id) if parent_id else None
        seen: set[str] = set()
        while cur is not None:
            cid = self._field(cur, "id")
            if cid in seen:
                break
            seen.add(cid)
            if self._field(cur, "verdict") == "confirmed":
                depth += 1
            pid = self._field(cur, "parent_id")
            cur = self.nodes.get(pid) if pid else None
        return depth

    def confirmed_lineage_depth(self) -> float:
        """Convergence health metric: mean confirmed-ancestry depth over confirmed
        nodes. Rises when the search deepens real findings into narrower ones;
        stays ~0 when it only adds shallow roots (the failure this fixes). 0 when
        there are no confirmed nodes yet."""
        confirmed = [n for n in self.nodes.values() if self._field(n, "verdict") == "confirmed"]
        if not confirmed:
            return 0.0
        return sum(self._confirmed_ancestry_depth(n) + 1 for n in confirmed) / len(confirmed)

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
            "finding_ledger": self.finding_ledger,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypothesisTree":
        tree = cls()
        tree.nodes = {k: HypothesisNode.from_dict(v) for k, v in data.get("nodes", {}).items()}
        tree.frontier = data.get("frontier", [])
        tree.confirmed = data.get("confirmed", [])
        tree.exhausted = data.get("exhausted", [])
        tree._generation = data.get("generation", 0)
        tree.finding_ledger = list(data.get("finding_ledger") or [])
        for f in tree.finding_ledger:
            for eh in f.get("evidence_hashes") or []:
                tree._used_evidence_hashes.add(str(eh))
        return tree

    # ── Summary helpers ──────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        verdicts: dict[str, int] = {}
        for node in self.nodes.values():
            verdicts[node.verdict] = verdicts.get(node.verdict, 0) + 1
        lineages = [n.lineage_length or self.lineage_length(n.id) for n in self.nodes.values()]
        tested = sum(v for k, v in verdicts.items() if k != "pending")
        decisive = verdicts.get("confirmed", 0) + verdicts.get("refuted", 0)
        return {
            "total_nodes": len(self.nodes),
            "frontier_size": len(self.frontier),
            "confirmed_count": verdicts.get("confirmed", 0),
            "exhausted_count": len(self.exhausted),
            "max_depth": max((n.depth for n in self.nodes.values()), default=0),
            "max_lineage": max(lineages) if lineages else 0,
            "avg_lineage": round(sum(lineages) / len(lineages), 2) if lineages else 0.0,
            "max_generation": max((n.generation for n in self.nodes.values()), default=0),
            "closure_ratio": round(decisive / tested, 4) if tested else 0.0,
            "verdict_counts": verdicts,
        }

    def confirmed_findings_text(self, max_n: int = 10) -> str:
        """Compact text of top confirmed findings for hypothesis generation prompts."""
        from propab.research_quality import paper_eligible_finding

        lines: list[str] = []
        for entry in self.finding_ledger[-max_n:]:
            if isinstance(entry, dict) and paper_eligible_finding(entry):
                lines.append(
                    f"- [{entry.get('claim_type')}, {entry.get('replication_level')}] "
                    f"{entry.get('claim', '')[:300]}"
                )
        for nid in self.confirmed[:max_n]:
            if len(lines) >= max_n:
                break
            node = self.nodes.get(nid)
            if node:
                ev = (node.evidence_summary or "")[:200]
                lines.append(f"- [confirmed, depth={node.depth}] {node.text}. {ev}")
        return "\n".join(lines) if lines else "No confirmed findings yet."
