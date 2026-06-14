"""Operator DAG — state → operator → state → reward (fixes.md #3)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from propab.operator_credit.difference_rewards import node_outcome_reward
from propab.operator_credit.operator_trace import NodeOperatorTrace, OperatorTraceLedger


@dataclass
class OperatorDAGEdge:
    campaign_id: str
    node_id: str
    step_index: int
    family: str
    operator: str
    state_in: list[float]
    state_out: list[float]
    reward: float
    outcome: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorDAG:
    edges: list[OperatorDAGEdge] = field(default_factory=list)

    def add_edge(self, edge: OperatorDAGEdge) -> None:
        self.edges.append(edge)

    def for_campaign(self, campaign_id: str) -> list[OperatorDAGEdge]:
        return [e for e in self.edges if e.campaign_id == campaign_id]

    def operator_sequence(self, campaign_id: str, node_id: str) -> list[OperatorDAGEdge]:
        return [
            e for e in self.edges
            if e.campaign_id == campaign_id and e.node_id == node_id
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"edges": [e.to_dict() for e in self.edges], "count": len(self.edges)}

    @classmethod
    def from_traces(cls, traces: OperatorTraceLedger) -> OperatorDAG:
        dag = cls()
        for trace in traces.traces:
            state = list(trace.state_vector)
            node_reward = node_outcome_reward(trace.outcome)
            for i, step in enumerate(trace.operators_used):
                state_out = list(state)
                if len(state_out) > 2:
                    state_out[2] = round(state_out[2] + 0.05 * (i + 1), 4)
                dag.add_edge(OperatorDAGEdge(
                    campaign_id=trace.campaign_id,
                    node_id=trace.node_id,
                    step_index=i,
                    family=step.family,
                    operator=step.operator,
                    state_in=state,
                    state_out=state_out,
                    reward=round(node_reward / max(1, len(trace.operators_used)), 4),
                    outcome=trace.outcome,
                ))
                state = state_out
        return dag
