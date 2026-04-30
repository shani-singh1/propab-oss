from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentBudget:
    max_steps: int
    min_steps: int
    deadline: float  # monotonic timestamp


@dataclass
class RoundBudget:
    max_seconds: float
    max_hypotheses: int
    agent_budget: AgentBudget


@dataclass
class ResearchBudget:
    # Hard limits
    max_rounds: int = 5
    max_hours: float = 1.0
    max_hypotheses_total: int = 50
    target_confirmed: int = 3

    # Soft limits
    min_marginal_return: float = 0.05
    max_stale_rounds: int = 2  # stop after N rounds with no new confirmed

    # Per-agent
    agent_max_steps: int = 15
    agent_min_steps: int = 5
    agent_max_seconds: int = 300

    # Per-round
    max_hypotheses_per_round: int = 5
    max_seconds_per_round: int = 600

    # Runtime state (not config)
    _start_time: float = field(default_factory=time.monotonic, repr=False)
    _deadline: float = field(init=False, repr=False)
    rounds_completed: int = field(default=0, repr=False)
    hypotheses_tested: int = field(default=0, repr=False)
    stale_rounds: int = field(default=0, repr=False)
    confirmed_total: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._deadline = self._start_time + self.max_hours * 3600

    @classmethod
    def from_settings(cls, settings: Any) -> "ResearchBudget":
        return cls(
            max_rounds=int(settings.research_max_rounds),
            max_hours=float(settings.research_max_hours),
            max_hypotheses_total=int(settings.research_max_hypotheses),
            target_confirmed=int(settings.research_target_confirmed),
            min_marginal_return=float(settings.research_min_marginal_return),
            agent_max_steps=int(settings.agent_max_steps),
            agent_min_steps=int(settings.agent_min_steps),
            max_hypotheses_per_round=int(settings.sub_agent_max_rounds) * 2,
        )

    def exhausted(self) -> bool:
        if time.monotonic() >= self._deadline:
            return True
        if self.rounds_completed >= self.max_rounds:
            return True
        if self.hypotheses_tested >= self.max_hypotheses_total:
            return True
        if self.confirmed_total >= self.target_confirmed:
            return True
        if self.stale_rounds >= self.max_stale_rounds:
            return True
        return False

    def stop_reason(self) -> str | None:
        if time.monotonic() >= self._deadline:
            return f"time budget exhausted ({self.max_hours}h)"
        if self.rounds_completed >= self.max_rounds:
            return f"round budget exhausted ({self.max_rounds} rounds)"
        if self.hypotheses_tested >= self.max_hypotheses_total:
            return f"hypothesis budget exhausted ({self.max_hypotheses_total} total)"
        if self.confirmed_total >= self.target_confirmed:
            return f"target confirmed findings reached ({self.target_confirmed})"
        if self.stale_rounds >= self.max_stale_rounds:
            return f"diminishing returns: {self.stale_rounds} consecutive rounds with no new confirmed finding"
        return None

    def round_budget(self, round_number: int) -> RoundBudget:
        remaining_seconds = max(0.0, self._deadline - time.monotonic())
        rounds_remaining = max(1, self.max_rounds - round_number)
        per_round_seconds = min(self.max_seconds_per_round, remaining_seconds / rounds_remaining)
        return RoundBudget(
            max_seconds=per_round_seconds,
            max_hypotheses=self.max_hypotheses_per_round,
            agent_budget=AgentBudget(
                max_steps=self.agent_max_steps,
                min_steps=self.agent_min_steps,
                deadline=time.monotonic() + min(self.agent_max_seconds, per_round_seconds / 2),
            ),
        )

    def record_round(self, confirmed: int, refuted: int, inconclusive: int, n_hypotheses: int) -> None:
        self.rounds_completed += 1
        self.hypotheses_tested += n_hypotheses
        self.confirmed_total += confirmed
        if confirmed == 0:
            self.stale_rounds += 1
        else:
            self.stale_rounds = 0

    def summary(self) -> dict:
        elapsed = time.monotonic() - self._start_time
        remaining = max(0.0, self._deadline - time.monotonic())
        return {
            "rounds_completed": self.rounds_completed,
            "rounds_remaining": max(0, self.max_rounds - self.rounds_completed),
            "hypotheses_tested": self.hypotheses_tested,
            "confirmed_total": self.confirmed_total,
            "elapsed_sec": round(elapsed, 1),
            "remaining_sec": round(remaining, 1),
            "stale_rounds": self.stale_rounds,
            "exhausted": self.exhausted(),
            "stop_reason": self.stop_reason(),
        }

    def to_dict(self) -> dict:
        return {
            "max_rounds": self.max_rounds,
            "max_hours": self.max_hours,
            "max_hypotheses_total": self.max_hypotheses_total,
            "target_confirmed": self.target_confirmed,
            "min_marginal_return": self.min_marginal_return,
            "agent_max_steps": self.agent_max_steps,
            "agent_min_steps": self.agent_min_steps,
            **self.summary(),
        }
