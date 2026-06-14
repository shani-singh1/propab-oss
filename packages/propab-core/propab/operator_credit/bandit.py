"""Bandit allocation for operator selection — UCB and Thompson (P5)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BanditArm:
    family: str
    operator: str
    pulls: int = 0
    total_reward: float = 0.0
    successes: int = 0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(1, self.pulls)

    def record(self, reward: float, *, success: bool = False) -> None:
        self.pulls += 1
        self.total_reward += reward
        if success:
            self.successes += 1


@dataclass
class OperatorBandit:
    arms: dict[str, BanditArm] = field(default_factory=dict)
    total_pulls: int = 0

    def _key(self, family: str, operator: str) -> str:
        return f"{family}|{operator}"

    def ensure_arms(self, family: str, operators: tuple[str, ...]) -> None:
        for op in operators:
            key = self._key(family, op)
            if key not in self.arms:
                self.arms[key] = BanditArm(family=family, operator=op)

    def update(self, family: str, operator: str, reward: float, *, success: bool = False) -> None:
        key = self._key(family, operator)
        if key not in self.arms:
            self.arms[key] = BanditArm(family=family, operator=operator)
        self.arms[key].record(reward, success=success)
        self.total_pulls += 1

    def select_ucb(self, family: str, *, c: float = 1.4) -> str:
        candidates = [a for a in self.arms.values() if a.family == family]
        if not candidates:
            return ""
        for arm in candidates:
            if arm.pulls == 0:
                return arm.operator
        log_n = math.log(max(1, self.total_pulls))
        return max(
            candidates,
            key=lambda a: a.mean_reward + c * math.sqrt(log_n / a.pulls),
        ).operator

    def select_thompson(self, family: str) -> str:
        candidates = [a for a in self.arms.values() if a.family == family]
        if not candidates:
            return ""
        best_op = candidates[0].operator
        best_sample = -1.0
        for arm in candidates:
            alpha = 1 + arm.successes
            beta = 1 + arm.pulls - arm.successes
            sample = random.betavariate(max(0.1, alpha), max(0.1, beta))
            if sample > best_sample:
                best_sample = sample
                best_op = arm.operator
        return best_op

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pulls": self.total_pulls,
            "arms": {
                k: {
                    "family": a.family,
                    "operator": a.operator,
                    "pulls": a.pulls,
                    "total_reward": a.total_reward,
                    "successes": a.successes,
                    "mean_reward": round(a.mean_reward, 4),
                }
                for k, a in self.arms.items()
            },
        }
