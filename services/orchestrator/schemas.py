from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Prior:
    """Structured prior aligned with ARCHITECTURE §5.4."""

    established_facts: list[dict]
    contested_claims: list[dict]
    open_gaps: list[dict]
    dead_ends: list[dict]
    key_papers: list[dict]

    def to_dict(self) -> dict:
        return {
            "established_facts": self.established_facts,
            "contested_claims": self.contested_claims,
            "open_gaps": self.open_gaps,
            "dead_ends": self.dead_ends,
            "key_papers": self.key_papers,
        }
