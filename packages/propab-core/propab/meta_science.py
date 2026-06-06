"""
Phase G — meta-science: measurable research ability across campaigns.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from propab.config import settings


def meta_store_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "meta_science.json"


@dataclass
class CampaignObservation:
    campaign_id: str
    question: str
    tested: int
    confirmed: int
    refuted: int
    inconclusive: int
    closure_ratio: float
    theme_entropy: float
    general_theme_fraction: float
    compute_seconds: int
    policy_generation: int
    knowledge_claims: int
    knowledge_failures: int
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetaScienceLedger:
    observations: list[CampaignObservation] = field(default_factory=list)

    def record(self, obs: CampaignObservation) -> None:
        self.observations = [o for o in self.observations if o.campaign_id != obs.campaign_id]
        self.observations.append(obs)
        self.observations.sort(key=lambda x: x.recorded_at)

    def learning_curve(self) -> dict[str, list[float]]:
        return {
            "closure_ratio": [o.closure_ratio for o in self.observations],
            "confirmed_rate": [
                o.confirmed / max(1, o.tested) for o in self.observations
            ],
            "theme_entropy": [o.theme_entropy for o in self.observations],
            "general_fraction": [o.general_theme_fraction for o in self.observations],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"observations": [o.to_dict() for o in self.observations]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetaScienceLedger:
        obs = [CampaignObservation(**o) for o in (data.get("observations") or [])]
        return cls(observations=obs)

    def save(self, path: Path | None = None) -> Path:
        p = path or meta_store_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> MetaScienceLedger:
        p = path or meta_store_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return cls()
