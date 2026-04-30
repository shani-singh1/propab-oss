from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from propab.llm import LLMClient

from .significance import SignificanceResult, any_significance_tool_ran, check_significance

logger = logging.getLogger(__name__)

# ─── Data structures ──────────────────────────────────────────────────────────


@dataclass
class AgentAction:
    action_type: str  # "tool" | "code" | "stop"
    tool_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    code_description: str | None = None
    reasoning: str = ""
    expected_outcome: str = ""


@dataclass
class AgentContext:
    hypothesis_text: str
    test_methodology: str
    learned_from: str | None
    peer_findings: list[dict]
    results_so_far: list[dict]
    tool_names_run: list[str]
    steps_taken: int
    max_steps: int
    min_steps: int

    def significance_status(self) -> SignificanceResult:
        return check_significance(self.results_so_far)

    def to_results_summary(self, max_chars: int = 3000) -> str:
        """Token-bounded compact summary of results collected so far."""
        if not self.results_so_far:
            return "No results yet."
        parts = []
        for i, r in enumerate(self.results_so_far):
            chunk = json.dumps(r, ensure_ascii=False)
            parts.append(f"Step {i + 1}: {chunk[:600]}")
        full = "\n".join(parts)
        if len(full) <= max_chars:
            return full
        # Always show the most recent results; trim from the front.
        ellipsis = "\n...(earlier results truncated)...\n"
        # Reserve space for ellipsis + at least the last result
        tail_parts = parts[-2:]
        tail = "\n".join(tail_parts)
        # Trim tail to fit within max_chars - ellipsis
        available = max(0, max_chars - len(ellipsis))
        tail = tail[-available:] if len(tail) > available else tail
        # Use remaining budget for head
        head_budget = available - len(tail)
        head = full[:head_budget] if head_budget > 0 else ""
        return (head + ellipsis + tail)[:max_chars + len(ellipsis)]

    def to_peer_summary(self, max_chars: int = 800) -> str:
        if not self.peer_findings:
            return "No peer findings yet."
        parts = []
        for pf in self.peer_findings[-5:]:  # last 5 peers
            verdict = pf.get("verdict", "unknown")
            finding = (pf.get("learned") or pf.get("key_finding") or "")[:200]
            parts.append(f"- Peer verdict={verdict}: {finding}")
        return "\n".join(parts)[:max_chars]


# ─── Think prompt ─────────────────────────────────────────────────────────────

_THINK_PROMPT_TMPL = """You are a research sub-agent testing the following hypothesis.

Hypothesis: {hypothesis_text}
Test methodology: {test_methodology}
{learned_block}
Results collected so far ({steps_taken} of max {max_steps} steps):
{results_summary}

Current significance status:
  p_value: {p_value}
  effect_size: {effect_size}
  confidence_interval: {ci}
  Significance gate passed: {gate_passed}
  Has ANY significance-capable tool run yet: {sig_tool_ran}

Peer agent findings this round:
{peer_summary}

Available tools (name → description):
{tools_summary}

What should you do next? Rules:
1. If the significance gate is NOT passed yet and steps_taken >= {min_steps}, you MUST run one of
   these significance tools before stopping: statistical_significance, bootstrap_confidence,
   literature_baseline_compare. Do not stop without statistical evidence.
2. If you have collected enough data and want to run significance tests, do so now.
3. If you have passing significance AND sufficient corroboration (>= {min_steps} steps), you MAY stop.
4. Do not repeat a tool you already ran unless with meaningfully different parameters.

Choose ONE action:
- tool: call a specific tool
- code: write custom Python code (describe what it computes)
- stop: stop collecting evidence (only allowed after significance gate is passed)

Return JSON only:
{{
  "action_type": "tool" | "code" | "stop",
  "tool_name": "tool_name_here",
  "params": {{"param": "value"}},
  "code_description": "what the code computes (only if action_type=code)",
  "reasoning": "one sentence: why this action next",
  "expected_outcome": "what you expect to learn or confirm from this action"
}}
"""

_CORRECTION_PROMPT_TMPL = """You tried to stop before running any significance test.

You MUST run at least one of these tools before stopping:
  - statistical_significance: compare two result vectors, returns p_value and effect_size
  - bootstrap_confidence: compute confidence interval for a metric, returns ci_lower/ci_upper
  - literature_baseline_compare: compare your results to a baseline, returns p_value

Current results you can use as input:
{results_summary}

Return JSON only with action_type="tool" and one of the above tool names:
{{
  "action_type": "tool",
  "tool_name": "statistical_significance",
  "params": {{"results_a": [...], "results_b": [...]}},
  "reasoning": "significance test required before stopping",
  "expected_outcome": "p_value and effect_size for the observed difference"
}}
"""


# ─── Decision logic ───────────────────────────────────────────────────────────


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _tools_summary(specs: list[dict]) -> str:
    lines = []
    for s in specs:
        sig_flag = " [SIGNIFICANCE]" if s.get("significance_capable") else ""
        lines.append(f"  {s['name']}{sig_flag}: {s.get('description', '')}")
    return "\n".join(lines)


async def decide_next_action(
    context: AgentContext,
    specs: list[dict],
    llm: LLMClient,
    session_id: str,
    hypothesis_id: str,
) -> AgentAction:
    """
    Core think step. LLM observes all context and decides what to do next.
    Enforces the significance gate: if agent tries to stop without sig evidence,
    issues a correction prompt and tries again once.
    """
    sig = context.significance_status()
    learned_block = (
        f"Learned from prior work on this hypothesis: {context.learned_from}\n"
        if context.learned_from
        else ""
    )

    prompt = _THINK_PROMPT_TMPL.format(
        hypothesis_text=context.hypothesis_text,
        test_methodology=context.test_methodology,
        learned_block=learned_block,
        steps_taken=context.steps_taken,
        max_steps=context.max_steps,
        min_steps=context.min_steps,
        results_summary=context.to_results_summary(),
        p_value=sig.p_value if sig.p_value is not None else "not measured yet",
        effect_size=sig.effect_size if sig.effect_size is not None else "not measured yet",
        ci=sig.confidence_interval if sig.confidence_interval is not None else "not measured yet",
        gate_passed=sig.gate_passed,
        sig_tool_ran=any_significance_tool_ran(context.tool_names_run),
        peer_summary=context.to_peer_summary(),
        tools_summary=_tools_summary(specs),
    )

    raw = await llm.call(
        prompt=prompt,
        purpose="agent.decide_next_step",
        session_id=session_id,
        hypothesis_id=hypothesis_id,
    )
    data = _extract_json(raw) or {}
    action = _parse_action(data)

    # Enforce significance gate: if agent wants to stop without any sig tool run,
    # issue a correction prompt once.
    if (
        action.action_type == "stop"
        and not any_significance_tool_ran(context.tool_names_run)
        and context.steps_taken >= context.min_steps
    ):
        logger.info(
            "Agent %s tried to stop without significance tool. Issuing correction prompt.",
            hypothesis_id,
        )
        correction_prompt = _CORRECTION_PROMPT_TMPL.format(
            results_summary=context.to_results_summary(max_chars=1500),
        )
        raw2 = await llm.call(
            prompt=correction_prompt,
            purpose="agent.significance_correction",
            session_id=session_id,
            hypothesis_id=hypothesis_id,
        )
        data2 = _extract_json(raw2) or {}
        action = _parse_action(data2)
        # If still returning stop after correction, fall back to forcing statistical_significance
        if action.action_type == "stop":
            action = _fallback_significance_action(context)

    return action


def _parse_action(data: dict) -> AgentAction:
    action_type = str(data.get("action_type", "stop")).strip().lower()
    if action_type not in ("tool", "code", "stop"):
        action_type = "stop"
    return AgentAction(
        action_type=action_type,
        tool_name=str(data.get("tool_name", "")).strip() or None,
        params=data.get("params") if isinstance(data.get("params"), dict) else {},
        code_description=str(data.get("code_description", "")).strip() or None,
        reasoning=str(data.get("reasoning", "")).strip(),
        expected_outcome=str(data.get("expected_outcome", "")).strip(),
    )


def _fallback_significance_action(context: AgentContext) -> AgentAction:
    """
    Hard fallback: extract numeric values from results and force a bootstrap_confidence call.
    This fires when the LLM ignores the correction prompt and still wants to stop.
    """
    values: list[float] = []
    for r in context.results_so_far:
        if isinstance(r, dict):
            for v in r.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values.append(float(v))
                    if len(values) >= 20:
                        break
        if len(values) >= 20:
            break

    if len(values) >= 2:
        return AgentAction(
            action_type="tool",
            tool_name="bootstrap_confidence",
            params={"values": values[:20]},
            reasoning="Forced significance gate: bootstrap CI from accumulated metric values.",
            expected_outcome="confidence_interval for the observed metric distribution",
        )

    return AgentAction(
        action_type="stop",
        reasoning="Significance gate fallback: no numeric values to test.",
    )


def should_stop(context: AgentContext) -> bool:
    """
    Decide whether the agent should stop collecting evidence.
    Called before each think step — if True, skip the LLM call and finalize.
    """
    # Hard budget cap
    if context.steps_taken >= context.max_steps:
        return True
    return False
