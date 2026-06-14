"""Campaign family DAG — lineage and inherited priors (fixes.md #5)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.operator_credit.operator_priors import OperatorPriors
from propab.policy_store import PolicyStore


def family_dag_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "campaign_family_dag.json"


@dataclass
class CampaignFamilyNode:
    campaign_id: str
    parent_campaign_id: str | None
    policy_id: str | None
    baseline_campaign_id: str | None
    inherited_policy_id: str | None = None
    inherited_operators: dict[str, str] = field(default_factory=dict)
    inherited_priors: dict[str, float] = field(default_factory=dict)
    depth: int = 0
    children: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignFamilyDAG:
    nodes: dict[str, CampaignFamilyNode] = field(default_factory=dict)
    roots: list[str] = field(default_factory=list)

    def add(self, node: CampaignFamilyNode) -> None:
        self.nodes[node.campaign_id] = node
        if node.parent_campaign_id and node.parent_campaign_id in self.nodes:
            parent = self.nodes[node.parent_campaign_id]
            if node.campaign_id not in parent.children:
                parent.children.append(node.campaign_id)

    def lineage(self, campaign_id: str) -> list[str]:
        chain: list[str] = []
        cur: str | None = campaign_id
        seen: set[str] = set()
        while cur and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            node = self.nodes.get(cur)
            cur = node.parent_campaign_id if node else None
        return list(reversed(chain))

    def descendants(self, campaign_id: str) -> list[str]:
        out: list[str] = []
        stack = list(self.nodes.get(campaign_id, CampaignFamilyNode(campaign_id, None, None, None)).children)
        while stack:
            cid = stack.pop()
            if cid in out:
                continue
            out.append(cid)
            node = self.nodes.get(cid)
            if node:
                stack.extend(node.children)
        return out

    def families_by_root(self) -> dict[str, list[str]]:
        families: dict[str, list[str]] = {}
        for root in self.roots:
            families[root] = [root] + self.descendants(root)
        for cid, node in self.nodes.items():
            if not node.parent_campaign_id and cid not in self.roots:
                self.roots.append(cid)
                families[cid] = [cid] + self.descendants(cid)
        return families

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "roots": self.roots,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignFamilyDAG:
        dag = cls(roots=list(data.get("roots") or []))
        for cid, raw in (data.get("nodes") or {}).items():
            dag.nodes[cid] = CampaignFamilyNode(**raw)
        return dag

    def save(self, path: Path | None = None) -> Path:
        p = path or family_dag_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> CampaignFamilyDAG:
        p = path or family_dag_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()


def build_campaign_family_dag(
    *,
    campaign_ids: list[str],
    baseline_by_campaign: dict[str, str | None],
    priors: OperatorPriors | None = None,
) -> CampaignFamilyDAG:
    store = PolicyStore.load()
    dag = CampaignFamilyDAG()

    for cid in campaign_ids:
        binding = store.active_bindings.get(cid)
        baseline = baseline_by_campaign.get(cid) or (binding.baseline_campaign_id if binding else None)
        policy_id = binding.policy_id if binding else None
        parent = baseline if baseline and baseline != cid else None
        depth = 0
        inherited_policy: str | None = None
        inherited_ops: dict[str, str] = {}
        inherited_priors: dict[str, float] = {}

        if parent and parent in dag.nodes:
            depth = dag.nodes[parent].depth + 1
            inherited_policy = dag.nodes[parent].policy_id
            inherited_ops = dict(dag.nodes[parent].inherited_operators)
            inherited_priors = dict(dag.nodes[parent].inherited_priors)

        if priors and parent:
            for family in ("branching", "retrieval", "mutation", "verification"):
                op = priors.recommended_operator(family, [0.1, 0.2, 0.3, 0.4, 0.5])
                if op:
                    inherited_ops[family] = op
                    inherited_priors[f"{family}:{op}"] = priors.operator_probability(
                        family, op, [0.1, 0.2, 0.3, 0.4, 0.5],
                    )

        node = CampaignFamilyNode(
            campaign_id=cid,
            parent_campaign_id=parent,
            policy_id=policy_id,
            baseline_campaign_id=baseline,
            inherited_policy_id=inherited_policy,
            inherited_operators=inherited_ops,
            inherited_priors=inherited_priors,
            depth=depth,
        )
        dag.add(node)
        if not parent and cid not in dag.roots:
            dag.roots.append(cid)

    return dag


def load_baselines_from_trajectory_file(path: Path | str) -> dict[str, str | None]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, str | None] = {}
    global_baseline = data.get("baseline_campaign_id")
    for camp in data.get("campaigns") or []:
        cid = camp.get("campaign_id")
        if cid:
            out[cid] = camp.get("baseline_campaign_id") or global_baseline
    return out
