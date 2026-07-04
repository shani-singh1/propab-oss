"""Policy store — candidate / accepted / rejected; no automatic promotion."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from propab.config import settings
from propab.knowledge_graph import new_id
from propab.policy_buckets import bucket_key
from propab.policy_record import PolicyRecord, PolicyStatus, PredictedEffects
from propab.search_policy import SearchPolicy, policy_store_path as legacy_policy_path


def policy_store_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "policy_store.json"


@dataclass
class CampaignPolicyBinding:
    campaign_id: str
    policy_id: str
    policy_mode: Literal["accepted", "candidate"]
    budget_bucket: str
    domain_bucket: str
    baseline_campaign_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyStore:
    version: int = 1
    policies: dict[str, PolicyRecord] = field(default_factory=dict)
    accepted: dict[str, str] = field(default_factory=dict)  # bucket_key -> policy_id
    rejected_ids: list[str] = field(default_factory=list)
    active_bindings: dict[str, CampaignPolicyBinding] = field(default_factory=dict)

    def get_policy(self, policy_id: str) -> PolicyRecord | None:
        return self.policies.get(policy_id)

    def accepted_policy(
        self,
        *,
        domain_bucket: str,
        budget_bucket: str,
    ) -> PolicyRecord:
        key = bucket_key(domain_bucket, budget_bucket)
        pid = self.accepted.get(key)
        if pid and pid in self.policies:
            rec = self.policies[pid]
            if rec.status == PolicyStatus.ACCEPTED:
                return rec
        rec = PolicyRecord.empty_accepted(
            budget_bucket=budget_bucket,
            domain_bucket=domain_bucket,
        )
        self.policies[rec.id] = rec
        self.accepted[key] = rec.id
        return rec

    def latest_candidate(
        self,
        *,
        domain_bucket: str,
        budget_bucket: str,
    ) -> PolicyRecord | None:
        key = bucket_key(domain_bucket, budget_bucket)
        candidates = [
            p for p in self.policies.values()
            if p.status == PolicyStatus.CANDIDATE
            and p.domain_bucket == domain_bucket
            and p.budget_bucket == budget_bucket
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.generation)

    def bind_campaign(
        self,
        *,
        campaign_id: str,
        policy_mode: Literal["accepted", "candidate"],
        domain_bucket: str,
        budget_bucket: str,
        baseline_campaign_id: str | None = None,
    ) -> PolicyRecord:
        if policy_mode == "candidate":
            rec = self.latest_candidate(
                domain_bucket=domain_bucket,
                budget_bucket=budget_bucket,
            )
            if rec is None:
                rec = self.accepted_policy(
                    domain_bucket=domain_bucket,
                    budget_bucket=budget_bucket,
                )
        else:
            rec = self.accepted_policy(
                domain_bucket=domain_bucket,
                budget_bucket=budget_bucket,
            )
        self.active_bindings[campaign_id] = CampaignPolicyBinding(
            campaign_id=campaign_id,
            policy_id=rec.id,
            policy_mode=policy_mode,
            budget_bucket=budget_bucket,
            domain_bucket=domain_bucket,
            baseline_campaign_id=baseline_campaign_id,
        )
        return rec

    def unbind_campaign(self, campaign_id: str) -> CampaignPolicyBinding | None:
        return self.active_bindings.pop(campaign_id, None)

    def add_candidate(
        self,
        *,
        parent: PolicyRecord,
        params: dict[str, Any],
        rationale: str,
        predicted: PredictedEffects,
        falsification: list[str],
    ) -> PolicyRecord:
        rec = PolicyRecord(
            id=new_id("pol"),
            generation=parent.generation + 1,
            parent_policy_id=parent.id,
            budget_bucket=parent.budget_bucket,
            domain_bucket=parent.domain_bucket,
            boosts=dict(params.get("boosts") or {}),
            penalties=dict(params.get("penalties") or {}),
            blocked_failures=list(params.get("blocked_failures") or []),
            saturated_themes=list(params.get("saturated_themes") or []),
            rationale=rationale,
            predicted_effects=predicted,
            falsification_conditions=falsification,
            status=PolicyStatus.CANDIDATE,
            prefer_replication_t2_plus=bool(params.get("prefer_replication_t2_plus", True)),
            closure_target=parent.closure_target,
        )
        self.policies[rec.id] = rec
        return rec

    def accept_policy(self, policy_id: str) -> PolicyRecord:
        rec = self.policies[policy_id]
        rec.status = PolicyStatus.ACCEPTED
        key = bucket_key(rec.domain_bucket, rec.budget_bucket)
        self.accepted[key] = policy_id
        return rec

    def reject_policy(self, policy_id: str) -> PolicyRecord:
        rec = self.policies[policy_id]
        rec.status = PolicyStatus.REJECTED
        if policy_id not in self.rejected_ids:
            self.rejected_ids.append(policy_id)
        return rec

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "policies": {k: v.to_dict() for k, v in self.policies.items()},
            "accepted": self.accepted,
            "rejected_ids": self.rejected_ids,
            "active_bindings": {k: v.to_dict() for k, v in self.active_bindings.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyStore:
        store = cls(version=int(data.get("version") or 1))
        for k, v in (data.get("policies") or {}).items():
            store.policies[k] = PolicyRecord.from_dict(v)
        store.accepted = dict(data.get("accepted") or {})
        store.rejected_ids = list(data.get("rejected_ids") or [])
        for k, v in (data.get("active_bindings") or {}).items():
            store.active_bindings[k] = CampaignPolicyBinding(**v)
        return store

    def save(self, path: Path | None = None) -> Path:
        from propab.lifetime_postgres import lifetime_postgres_enabled, save_policy_store

        if lifetime_postgres_enabled():
            save_policy_store(self)
            self._sync_legacy_search_policy()
            return policy_store_path()
        p = path or policy_store_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        self._sync_legacy_search_policy()
        return p

    def _sync_legacy_search_policy(self) -> None:
        """Write accepted graphs:3h policy to legacy file if present."""
        key = bucket_key("graphs", "3h")
        pid = self.accepted.get(key)
        if pid and pid in self.policies:
            self.policies[pid].to_search_policy().save(legacy_policy_path())

    @classmethod
    def load(cls, path: Path | None = None) -> PolicyStore:
        from propab.lifetime_postgres import lifetime_postgres_enabled, load_policy_store_cls

        if lifetime_postgres_enabled():
            try:
                return load_policy_store_cls()
            except Exception:
                pass
        p = path or policy_store_path()
        if p.is_file():
            try:
                return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        store = cls()
        store._migrate_legacy_search_policy()
        return store

    def _migrate_legacy_search_policy(self) -> None:
        leg = legacy_policy_path()
        if not leg.is_file():
            return
        try:
            sp = SearchPolicy.load(leg)
        except Exception:
            return
        rec = PolicyRecord(
            id=new_id("pol"),
            generation=sp.generation,
            parent_policy_id=None,
            budget_bucket="3h",
            domain_bucket="graphs",
            boosts=dict(sp.theme_boost),
            penalties=dict(sp.theme_penalty),
            blocked_failures=list(sp.blocked_failure_signatures),
            saturated_themes=list(sp.saturated_themes),
            rationale="Migrated from legacy search_policy.json",
            status=PolicyStatus.ACCEPTED,
            prefer_replication_t2_plus=sp.prefer_replication_t2_plus,
            closure_target=sp.closure_target,
        )
        self.policies[rec.id] = rec
        self.accepted[bucket_key("graphs", "3h")] = rec.id
