from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoundSummary:
    round_number: int
    round_id: str
    confirmed: list[str] = field(default_factory=list)
    refuted: list[str] = field(default_factory=list)
    inconclusive: list[str] = field(default_factory=list)
    tools_used: set[str] = field(default_factory=set)
    mean_confidence: float = 0.0

    def all_inconclusive(self) -> bool:
        return len(self.confirmed) == 0 and len(self.refuted) == 0

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "round_id": self.round_id,
            "confirmed": self.confirmed,
            "refuted": self.refuted,
            "inconclusive": self.inconclusive,
            "tools_used": list(self.tools_used),
            "mean_confidence": self.mean_confidence,
        }


@dataclass
class AccumulatedLedger:
    """
    Accumulates research findings across multiple rounds of the research loop.
    Provides summaries for hypothesis generation, paper writing, and budget tracking.
    """

    confirmed: list[str] = field(default_factory=list)    # hypothesis_ids confirmed
    refuted: list[str] = field(default_factory=list)       # hypothesis_ids refuted
    inconclusive: list[str] = field(default_factory=list)  # hypothesis_ids inconclusive

    # Full result objects keyed by hypothesis_id
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    round_summaries: list[RoundSummary] = field(default_factory=list)
    all_hypothesis_texts: list[str] = field(default_factory=list)

    @property
    def total_confirmed(self) -> int:
        return len(self.confirmed)

    @property
    def total_refuted(self) -> int:
        return len(self.refuted)

    @property
    def total_inconclusive(self) -> int:
        return len(self.inconclusive)

    def add_result(self, result: dict[str, Any]) -> None:
        hid = result.get("hypothesis_id", "")
        verdict = result.get("verdict", "inconclusive")
        self.results[hid] = result
        if verdict == "confirmed":
            if hid not in self.confirmed:
                self.confirmed.append(hid)
        elif verdict == "refuted":
            if hid not in self.refuted:
                self.refuted.append(hid)
        else:
            if hid not in self.inconclusive:
                self.inconclusive.append(hid)

    def merge_round(
        self,
        round_number: int,
        round_id: str,
        round_results: list[dict[str, Any]],
        hypothesis_texts: list[str] | None = None,
    ) -> RoundSummary:
        summary = RoundSummary(round_number=round_number, round_id=round_id)
        confidences: list[float] = []

        for result in round_results:
            self.add_result(result)
            hid = result.get("hypothesis_id", "")
            verdict = result.get("verdict", "inconclusive")
            confidence = float(result.get("confidence") or 0.0)
            confidences.append(confidence)

            if verdict == "confirmed":
                summary.confirmed.append(hid)
            elif verdict == "refuted":
                summary.refuted.append(hid)
            else:
                summary.inconclusive.append(hid)

            # Extract tool names from evidence_summary for coverage tracking
            evidence = str(result.get("evidence_summary") or "")
            if "plan_origin" in evidence:
                for tool_hint in ("statistical_significance", "bootstrap_confidence",
                                  "train_model", "evaluate_model"):
                    if tool_hint in evidence:
                        summary.tools_used.add(tool_hint)

        summary.mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        self.round_summaries.append(summary)

        if hypothesis_texts:
            self.all_hypothesis_texts.extend(hypothesis_texts)

        return summary

    def marginal_return(self, round_number: int) -> float:
        """
        Marginal scientific progress made in the given round (0–1 scale).
        Used for diminishing-returns detection.
        """
        if round_number < 1 or len(self.round_summaries) < 2:
            return 1.0

        prev_idx = round_number - 1
        curr_idx = round_number
        if curr_idx >= len(self.round_summaries):
            return 0.0

        prev = self.round_summaries[prev_idx]
        curr = self.round_summaries[curr_idx]

        total = max(1, len(curr.confirmed) + len(curr.refuted) + len(curr.inconclusive))

        # New confirmed findings (most valuable)
        prev_confirmed_set = set(
            hid for rs in self.round_summaries[:curr_idx] for hid in rs.confirmed
        )
        new_confirmed = len([h for h in curr.confirmed if h not in prev_confirmed_set])
        delta_confirmed = new_confirmed / total

        # Confidence improvement in inconclusive hypotheses
        prev_conf = prev.mean_confidence
        curr_conf = curr.mean_confidence
        delta_confidence = max(0.0, (curr_conf - prev_conf) / max(0.01, prev_conf)) * 0.5

        # New tool coverage
        prev_tools = prev.tools_used
        new_tools = curr.tools_used - prev_tools
        delta_coverage = len(new_tools) / max(1, len(curr.tools_used)) * 0.3

        return float(min(1.0, 0.5 * delta_confirmed + 0.3 * delta_confidence + 0.2 * delta_coverage))

    def summary_for_hypothesis_generator(self) -> str:
        """
        Compact text summary of all prior findings, suitable for injection into
        the hypothesis generation prompt for the next round.
        """
        if not self.round_summaries:
            return "No prior round results."

        lines = [f"Prior round results ({len(self.round_summaries)} rounds completed):"]
        for rs in self.round_summaries:
            lines.append(
                f"  Round {rs.round_number}: "
                f"{len(rs.confirmed)} confirmed, {len(rs.refuted)} refuted, "
                f"{len(rs.inconclusive)} inconclusive. "
                f"Mean confidence: {rs.mean_confidence:.2f}."
            )

        # Summarize confirmed findings
        for hid in self.confirmed[:5]:
            r = self.results.get(hid, {})
            finding = (r.get("key_finding") or "")[:200]
            if finding:
                lines.append(f"  CONFIRMED: {finding}")

        # Summarize dead ends (refuted)
        for hid in self.refuted[:5]:
            r = self.results.get(hid, {})
            ev = (r.get("evidence_summary") or "")[:150]
            lines.append(f"  REFUTED: {ev[:100]}")

        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "confirmed": self.confirmed,
            "refuted": self.refuted,
            "inconclusive": self.inconclusive,
            "total_confirmed": self.total_confirmed,
            "total_refuted": self.total_refuted,
            "total_inconclusive": self.total_inconclusive,
            "rounds_completed": len(self.round_summaries),
        }

    def to_dict(self) -> dict:
        return {
            **self.summary(),
            "results": self.results,
            "round_summaries": [rs.to_dict() for rs in self.round_summaries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AccumulatedLedger":
        ledger = cls()
        ledger.confirmed = data.get("confirmed", [])
        ledger.refuted = data.get("refuted", [])
        ledger.inconclusive = data.get("inconclusive", [])
        ledger.results = data.get("results", {})
        for rs_dict in data.get("round_summaries", []):
            ledger.round_summaries.append(
                RoundSummary(
                    round_number=rs_dict["round_number"],
                    round_id=rs_dict["round_id"],
                    confirmed=rs_dict.get("confirmed", []),
                    refuted=rs_dict.get("refuted", []),
                    inconclusive=rs_dict.get("inconclusive", []),
                    tools_used=set(rs_dict.get("tools_used", [])),
                    mean_confidence=rs_dict.get("mean_confidence", 0.0),
                )
            )
        return ledger
