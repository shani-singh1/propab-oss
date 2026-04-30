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
        ellipsis = "\n...(earlier results truncated)...\n"
        tail_parts = parts[-2:]
        tail = "\n".join(tail_parts)
        available = max(0, max_chars - len(ellipsis))
        tail = tail[-available:] if len(tail) > available else tail
        head_budget = available - len(tail)
        head = full[:head_budget] if head_budget > 0 else ""
        return (head + ellipsis + tail)[:max_chars + len(ellipsis)]

    def to_peer_summary(self, max_chars: int = 800) -> str:
        if not self.peer_findings:
            return "No peer findings yet."
        parts = []
        for pf in self.peer_findings[-5:]:
            verdict = pf.get("verdict", "unknown")
            finding = (pf.get("learned") or pf.get("key_finding") or "")[:200]
            parts.append(f"- Peer verdict={verdict}: {finding}")
        return "\n".join(parts)[:max_chars]


# ─── Value extraction for chaining ───────────────────────────────────────────


def _extract_named_values(results_so_far: list[dict], tool_names: list[str]) -> str:
    """
    Extract all numeric values and list/array values from prior tool outputs,
    labeled by step and tool name. These are injected into the think prompt so
    the LLM can use real measurements instead of spec-example placeholders.
    """
    if not results_so_far:
        return "  (none yet)"

    lines: list[str] = []
    for i, result in enumerate(results_so_far):
        if not isinstance(result, dict):
            continue
        tool = tool_names[i] if i < len(tool_names) else f"step{i+1}"
        label = f"Step {i + 1} ({tool})"

        # First-level scalars
        for k, v in result.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                lines.append(f"  {label}: {k} = {round(float(v), 6)}")

        # First-level numeric lists (e.g. loss_curve values, etc.)
        # Skip keys handled in dedicated blocks below to avoid duplication
        _skip_list_keys = {"val_losses", "fixed_seed_results", "random_seed_results", "eval_losses"}
        for k, v in result.items():
            if k in _skip_list_keys:
                continue
            if isinstance(v, list) and 2 <= len(v) <= 50:
                nums = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
                if len(nums) == len(v):
                    rounded = [round(float(x), 6) for x in nums[:20]]
                    lines.append(f"  {label}: {k} = {rounded}")

        # Nested: results[] lists from run_experiment_grid
        results_list = result.get("results")
        if isinstance(results_list, list):
            for j, item in enumerate(results_list[:5]):
                if isinstance(item, dict):
                    cfg = item.get("config", {})
                    ms = item.get("mean_score")
                    vl = item.get("val_losses")
                    if isinstance(ms, (int, float)):
                        lines.append(f"  {label}: results[{j}].mean_score = {round(float(ms), 6)} config={cfg}")
                    if isinstance(vl, list) and vl:
                        nums = [round(float(x), 6) for x in vl[:10] if isinstance(x, (int, float))]
                        if nums:
                            lines.append(f"  {label}: results[{j}].val_losses = {nums}")

        # loss_curve summary
        lc = result.get("loss_curve")
        if isinstance(lc, list) and lc:
            last = lc[-1]
            if isinstance(last, dict):
                for k in ("train_loss", "val_loss"):
                    if k in last:
                        lines.append(f"  {label}: loss_curve[-1].{k} = {round(float(last[k]), 6)}")

        # val_losses list (from improved train_model)
        vl = result.get("val_losses")
        if isinstance(vl, list) and len(vl) >= 2:
            nums = [round(float(x), 6) for x in vl[:20] if isinstance(x, (int, float))]
            if nums:
                lines.append(f"  {label}: val_losses = {nums}  ← USE THIS for significance tests")

        # fixed_seed_results / random_seed_results (reproduce_result)
        for k in ("fixed_seed_results", "random_seed_results"):
            v = result.get(k)
            if isinstance(v, list) and len(v) >= 2:
                nums = [round(float(x), 6) for x in v[:10] if isinstance(x, (int, float))]
                if nums:
                    lines.append(f"  {label}: {k} = {nums}")

    return "\n".join(lines) if lines else "  (no numeric values in results yet)"


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

═══════════════════════════════════════════════════════
EXTRACTED NUMERIC VALUES from your prior results.
These are the ACTUAL measurements you must use in significance tool calls:
{extracted_values}

⚠️  CRITICAL DATA CHAINING RULE ⚠️
When calling statistical_significance, bootstrap_confidence, or literature_baseline_compare:
- results_a and results_b MUST be taken from the val_losses or metric lists above
- values MUST come from the measurements listed above
- NEVER use [0.9, 0.88, 0.91] — that is a tool spec example, NOT real data
- NEVER use [0.82, 0.8, 0.79] — same, spec example only
- NEVER use [0.1, 0.2, 0.15, 0.18] — spec example only
If you have val_losses from two different configs, pass those as results_a and results_b.
If you have a single val_losses list, pass it as values to bootstrap_confidence.
═══════════════════════════════════════════════════════

Peer agent findings this round:
{peer_summary}

Available tools (name → description):
{tools_summary}

What should you do next? Rules:
1. If the significance gate is NOT passed yet and steps_taken >= {min_steps}, you MUST run one of
   these significance tools before stopping: statistical_significance, bootstrap_confidence,
   literature_baseline_compare. Do not stop without statistical evidence.
2. If you have real val_losses or metric lists above, pass them to statistical_significance NOW.
3. If you have passing significance AND sufficient corroboration (>= {min_steps} steps), you MAY stop.
4. Do not repeat a tool you already ran unless with meaningfully different parameters.
5. Good sequence: build_mlp → train_model → (train_model again with different config) →
   statistical_significance(results_a=val_losses_A, results_b=val_losses_B) → stop

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

ACTUAL numeric values you have collected (use these — NOT spec examples):
{extracted_values}

If you see val_losses lists above, use them as results_a / results_b for statistical_significance.
If you see a single list of measurements, use it as values for bootstrap_confidence.

FORBIDDEN: Do NOT use [0.9, 0.88, 0.91], [0.82, 0.8, 0.79], or [0.1, 0.2, 0.15, 0.18].
Those are tool specification examples. Your real data is listed above.

Return JSON only with action_type="tool":
{{
  "action_type": "tool",
  "tool_name": "statistical_significance",
  "params": {{"results_a": [<actual values from above>], "results_b": [<actual values from above>]}},
  "reasoning": "significance test on actual measurements before stopping",
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


def _is_spec_example_params(tool_name: str, params: dict) -> bool:
    """Detect when agent is using hardcoded tool-spec example values."""
    if tool_name == "statistical_significance":
        ra = params.get("results_a")
        rb = params.get("results_b")
        example_a = [0.9, 0.88, 0.91]
        example_b = [0.82, 0.8, 0.79]
        if ra == example_a or rb == example_b:
            return True
    if tool_name == "bootstrap_confidence":
        vals = params.get("values")
        if vals == [0.1, 0.2, 0.15, 0.18]:
            return True
    if tool_name == "literature_baseline_compare":
        our = params.get("our_results")
        if our == [0.42, 0.44, 0.41]:
            return True
    return False


async def decide_next_action(
    context: AgentContext,
    specs: list[dict],
    llm: LLMClient,
    session_id: str,
    hypothesis_id: str,
) -> AgentAction:
    """
    Core think step. LLM observes all context including extracted numeric values
    and decides what to do next. Enforces the significance gate and rejects
    spec-example placeholder params.
    """
    sig = context.significance_status()
    learned_block = (
        f"Learned from prior work on this hypothesis: {context.learned_from}\n"
        if context.learned_from
        else ""
    )
    extracted_values = _extract_named_values(context.results_so_far, context.tool_names_run)

    prompt = _THINK_PROMPT_TMPL.format(
        hypothesis_text=context.hypothesis_text,
        test_methodology=context.test_methodology,
        learned_block=learned_block,
        steps_taken=context.steps_taken,
        max_steps=context.max_steps,
        min_steps=context.min_steps,
        results_summary=context.to_results_summary(),
        extracted_values=extracted_values,
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

    # If agent chose a significance tool with spec-example params, issue a correction
    if (
        action.action_type == "tool"
        and action.tool_name in ("statistical_significance", "bootstrap_confidence", "literature_baseline_compare")
        and _is_spec_example_params(action.tool_name, action.params)
    ):
        logger.info(
            "Agent %s used spec-example params for %s. Issuing extraction correction.",
            hypothesis_id,
            action.tool_name,
        )
        correction = _CORRECTION_PROMPT_TMPL.format(
            extracted_values=extracted_values,
        )
        raw2 = await llm.call(
            prompt=correction,
            purpose="agent.data_chaining_correction",
            session_id=session_id,
            hypothesis_id=hypothesis_id,
        )
        data2 = _extract_json(raw2) or {}
        action2 = _parse_action(data2)
        # Accept corrected action unless it still uses spec examples
        if not _is_spec_example_params(action2.tool_name or "", action2.params):
            action = action2
        else:
            # Last resort: extract actual values and force bootstrap
            action = _fallback_significance_action(context)

    # Enforce significance gate: if agent wants to stop without any sig tool run
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
            extracted_values=extracted_values,
        )
        raw2 = await llm.call(
            prompt=correction_prompt,
            purpose="agent.significance_correction",
            session_id=session_id,
            hypothesis_id=hypothesis_id,
        )
        data2 = _extract_json(raw2) or {}
        action = _parse_action(data2)
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
    Hard fallback: extract numeric values from actual results and force a
    bootstrap_confidence or statistical_significance call with real data.
    """
    # Try to find val_losses lists first (highest quality)
    list_a: list[float] = []
    list_b: list[float] = []
    for r in context.results_so_far:
        if not isinstance(r, dict):
            continue
        vl = r.get("val_losses")
        if isinstance(vl, list) and len(vl) >= 2:
            nums = [float(x) for x in vl if isinstance(x, (int, float)) and not isinstance(x, bool)]
            if not list_a:
                list_a = nums[:20]
            elif not list_b:
                list_b = nums[:20]

    if list_a and list_b:
        return AgentAction(
            action_type="tool",
            tool_name="statistical_significance",
            params={"results_a": list_a, "results_b": list_b},
            reasoning="Forced significance gate: comparing val_losses from two training runs.",
            expected_outcome="p_value and effect_size comparing two training configurations",
        )

    # Fall back to bootstrap on any scalars
    values: list[float] = []
    for r in context.results_so_far:
        if isinstance(r, dict):
            for v in r.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values.append(float(v))
                    if len(values) >= 20:
                        break
            # Also pick up from nested lists
            for k, v in r.items():
                if isinstance(v, list) and v:
                    nums = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
                    values.extend(nums[:10])
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
    if context.steps_taken >= context.max_steps:
        return True
    return False
