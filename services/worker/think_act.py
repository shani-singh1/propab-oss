from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from propab.config import settings
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
    # Complete executable Python source for action_type="code". When present, it is run in the
    # Docker sandbox (real execution); code_description is only a human-readable summary.
    code: str | None = None
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
    # Wall-clock cap (monotonic deadline). None = no deadline beyond max_steps.
    deadline_monotonic: float | None = None
    time_budget_exceeded: bool = False
    # Last validation / tool failures for the think prompt (same agent, no re-routing).
    tool_failures: list[dict[str, Any]] = field(default_factory=list)

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


def _tool_failures_block(ctx: AgentContext, *, max_items: int = 4, max_chars: int = 1200) -> str:
    if not ctx.tool_failures:
        return "  (none)"
    parts: list[str] = []
    for item in ctx.tool_failures[-max_items:]:
        if not isinstance(item, dict):
            continue
        blob = json.dumps(item, ensure_ascii=False)
        parts.append(f"  - {blob[:400]}{'...' if len(blob) > 400 else ''}")
    out = "\n".join(parts) if parts else "  (none)"
    return out[:max_chars] + ("..." if len(out) > max_chars else "")


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

Recent tool failures in THIS run (read before next tool call — fix parameters and retry the same tool; do not restart domain routing):
{tool_failures_block}

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
6. action_type="code" is a first-class instrument: write COMPLETE, self-contained, executable
   Python to compute, search, simulate, or verify whatever the hypothesis requires. This is the
   primary way to attack computational/mathematical problems (combinatorics, number theory,
   search/verification, constructions) where no domain tool applies.
   CODE OUTPUT CONTRACT (mandatory): the program MUST end by printing exactly one JSON line to
   stdout of the form: print(json.dumps({{"sandbox": "ok", ...your result fields...}}))
   Put your real measurements/values (e.g. counts, metrics, found constructions, verification
   booleans) as fields in that dict. Code that does not print this JSON line is treated as failed.
   VERIFICATION HYPOTHESES (number theory, combinatorics, constructions, search/proof — anything
   exactly checkable rather than statistical): include a boolean field "verified" — true only if
   your code rigorously checked the claim holds over the stated domain, false if it found a
   counterexample — plus a "certificate" field holding the witness (e.g. the explicit construction,
   the decomposition, or the counterexample). A reproduced verified=true CONFIRMS the hypothesis
   deterministically (no p-value needed); a verified=false (or a non-empty "counterexample") REFUTES
   it. Only set verified=true when the check is exact and exhaustive over the claimed range.
7. Maximum code steps allowed for this hypothesis: {max_code_steps}.
   Code steps already used: {used_code_steps}.
8. Prefer an exact-fit tool over code when one exists. If you are training/evaluating/comparing
   ML models, use tools (train_model, run_experiment_grid, compare_optimizers); reserve code for
   computations the tools do not cover.

SANDBOX ENVIRONMENT (read before writing any action_type="code"):
• Available: Python 3.11 with numpy, scipy, torch, matplotlib. CPU only (no GPU).
• NO network access. You CANNOT download datasets, fetch URLs, or pip install. Code that calls
  torchvision.datasets.*(download=True), urllib/requests, or any network I/O WILL FAIL.
• For training/evaluating on real datasets (e.g. MNIST), DO NOT write download code — call the
  tools (train_model, run_experiment_grid, evaluate_model, compare_optimizers); they handle real
  data. Reserve action_type="code" for math/combinatorics/search/verification/simulation/analysis
  on in-memory or synthetically-generated (e.g. numpy/torch.randn) data.

SANDBOX AND AGENT WALL CLOCK (critical for train_model / run_experiment_grid / any code step):
• Each sandbox run is capped at sandbox_timeout_sec={sandbox_timeout_sec} seconds (Docker, no GPU).
• Sandbox code receives injected helpers: SANDBOX_WALL_SEC (seconds budget) and SANDBOX_REMAINING_SEC().
  If generating custom loops, exit early when SANDBOX_REMAINING_SEC() drops below ~30 seconds.
• This entire hypothesis agent may run at most ~agent_wall_budget_sec={agent_wall_budget_sec} seconds wall-clock total.
• If you already hit one sandbox timeout on **code**, your next code attempt must **halve** n_steps/epochs (or shrink the grid) — identical resubmits are forbidden and will only time out again.

• Train/grid hard step ceiling for this profile: {agent_tool_n_steps_cap} (0 means the worker does not clamp n_steps).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADAPTIVE n_steps GUIDANCE — read before calling train_model or run_experiment_grid:
• If agent_tool_n_steps_cap > 0, treat it as a HARD ceiling (overrides the bullets below when they disagree).
• Subtle effects (activation functions, normalization placement, dropout): prefer n_steps near the cap when the cap is tight.
• Structural effects (depth vs width, layer size, skip connections): smaller n_steps often suffice.
• Convergence speed / optimizer comparison: needs longer runs — if the cap is low, say so in reasoning and use the longest allowed n_steps.
• Default (if unsure): n_steps = {n_steps_default}
• Avoid requesting n_steps far above the cap — the worker will clamp and you lose the intended comparison.
For classification tasks: use dataset='{classification_default_dataset}' unless a strong reason exists not to.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Choose ONE action:
- tool: call a specific tool
- code: write complete executable Python (full source in the "code" field; must print the JSON contract line)
- stop: stop collecting evidence (only allowed after significance gate is passed)

Return JSON only:
{{
  "action_type": "tool" | "code" | "stop",
  "tool_name": "tool_name_here",
  "params": {{"param": "value"}},
  "code_description": "one-line summary of what the code computes (only if action_type=code)",
  "code": "FULL Python source ending in print(json.dumps({{\\"sandbox\\":\\"ok\\", ...}})) (only if action_type=code)",
  "reasoning": "one sentence: why this action next",
  "expected_outcome": "what you expect to learn or confirm from this action"
}}
"""


def _count_code_steps(tool_names_run: list[str]) -> int:
    return sum(1 for n in tool_names_run if str(n).strip().lower() == "__code__")

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


# Legacy hardcoded significance-tool spec examples. Kept ONLY as a belt-and-braces
# fallback for the (rare) case where the caller cannot supply the live tool specs
# to _is_spec_example_params — e.g. a correction re-check where specs are not in
# scope. The generalized, spec-driven check below is the primary guard.
_LEGACY_SPEC_EXAMPLES: dict[str, dict[str, list[float]]] = {
    "statistical_significance": {"results_a": [0.9, 0.88, 0.91], "results_b": [0.82, 0.8, 0.79]},
    "bootstrap_confidence": {"values": [0.1, 0.2, 0.15, 0.18]},
    "literature_baseline_compare": {"our_results": [0.42, 0.44, 0.41]},
}


def _spec_by_name(specs: list[dict] | None, tool_name: str) -> dict | None:
    if not specs:
        return None
    for s in specs:
        if isinstance(s, dict) and s.get("name") == tool_name:
            return s
    return None


def _numeric_list(v: Any) -> list[float] | None:
    """Return a list of floats iff v is a homogeneous numeric list, else None."""
    if not isinstance(v, list) or not v:
        return None
    out: list[float] = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        out.append(float(x))
    return out


def _lists_trivially_equal(a: list[float], b: list[float]) -> bool:
    """
    Equal, or one is the other with a constant scale/offset (a trivial derivation:
    copying the example and multiplying by 10, adding a constant, or reordering it).
    We flag these because they carry no independent measurement — they are just the
    spec example dressed up.
    """
    if a == b:
        return True
    if len(a) != len(b) or len(a) < 2:
        return False
    # Reordering: same multiset of values.
    if sorted(a) == sorted(b):
        return True
    # Constant offset: a[i] - b[i] is (nearly) the same for all i.
    diffs = [a[i] - b[i] for i in range(len(a))]
    if max(diffs) - min(diffs) < 1e-9:
        return True
    # Constant positive scale: a[i] / b[i] is (nearly) the same for all i.
    if all(abs(x) > 1e-12 for x in b):
        ratios = [a[i] / b[i] for i in range(len(a))]
        if max(ratios) - min(ratios) < 1e-9:
            return True
    return False


def _params_match_example(params: dict, example_params: dict) -> bool:
    """
    True when any numeric-list param in ``params`` equals (or is a trivial
    derivation of) the corresponding numeric-list value in the tool's spec
    ``example_params``. This is deliberately general: it matches by VALUE, so
    an agent that copies statistical_significance's example into
    literature_baseline_compare's ``our_results`` is still caught, and any
    future significance tool is covered without a code change.
    """
    if not isinstance(example_params, dict):
        return False
    example_lists: list[list[float]] = []
    for ev in example_params.values():
        el = _numeric_list(ev)
        if el is not None:
            example_lists.append(el)
    if not example_lists:
        return False
    for pv in params.values():
        pl = _numeric_list(pv)
        if pl is None:
            continue
        for el in example_lists:
            if _lists_trivially_equal(pl, el):
                return True
    return False


def _is_spec_example_params(tool_name: str, params: dict, specs: list[dict] | None = None) -> bool:
    """
    Detect when an agent is passing a tool's own spec-example values (or a trivial
    derivation of them) as significance-tool inputs.

    Spec-driven and value-based: reads THIS run's live tool specs (which carry
    ``spec["example"]["params"]``) and flags a match against ANY significance-tool
    example — not a hardcoded 3-array denylist. Because matching is by value, it
    also catches an agent that copies one tool's example into a different tool.

    ``specs`` is optional so legacy call sites (and the correction re-check, where
    specs may not be in scope) still get the old three known examples as a floor.
    """
    if not isinstance(params, dict):
        return False

    matched_examples: list[dict] = []
    # 1. This tool's own live spec example.
    spec = _spec_by_name(specs, tool_name)
    if spec is not None:
        ex = (spec.get("example") or {}).get("params")
        if isinstance(ex, dict):
            matched_examples.append(ex)
    # 2. Every OTHER significance tool's live spec example (cross-tool copy guard).
    if specs:
        for s in specs:
            if not isinstance(s, dict) or not s.get("significance_capable"):
                continue
            ex = (s.get("example") or {}).get("params")
            if isinstance(ex, dict) and ex not in matched_examples:
                matched_examples.append(ex)
    # 3. Legacy hardcoded floor (covers the case where specs are unavailable, and
    #    guarantees we never REGRESS below the previous three-array check).
    for ex in _LEGACY_SPEC_EXAMPLES.values():
        if ex not in matched_examples:
            matched_examples.append(ex)

    return any(_params_match_example(params, ex) for ex in matched_examples)


async def decide_next_action(
    context: AgentContext,
    specs: list[dict],
    llm: LLMClient,
    session_id: str,
    hypothesis_id: str,
    *,
    sandbox_timeout_sec: int = 120,
    agent_wall_budget_sec: int = 600,
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
    steps_cap = int(getattr(settings, "agent_tool_n_steps_cap", 0) or 0)

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
        tool_failures_block=_tool_failures_block(context),
        tools_summary=_tools_summary(specs),
        max_code_steps=int(getattr(settings, "max_code_steps_per_hypothesis", 1)),
        used_code_steps=_count_code_steps(context.tool_names_run),
        n_steps_default=int(getattr(settings, "n_steps_default", 150)),
        classification_default_dataset=str(getattr(settings, "classification_default_dataset", "mnist")),
        sandbox_timeout_sec=int(sandbox_timeout_sec),
        agent_wall_budget_sec=int(agent_wall_budget_sec),
        agent_tool_n_steps_cap=steps_cap,
    )

    raw = await llm.call(
        prompt=prompt,
        purpose="agent.decide_next_step",
        session_id=session_id,
        hypothesis_id=hypothesis_id,
    )
    data = _extract_json(raw) or {}
    action = _parse_action(data)

    # Hard cap: max code steps per hypothesis
    used_code_steps = _count_code_steps(context.tool_names_run)
    max_code_steps = int(getattr(settings, "max_code_steps_per_hypothesis", 1))
    if action.action_type == "code" and used_code_steps >= max_code_steps:
        logger.info(
            "Code-step cap hit for %s (%s/%s). Forcing tool fallback.",
            hypothesis_id,
            used_code_steps,
            max_code_steps,
        )
        action = _fallback_significance_action(context)

    # If agent chose a significance tool with spec-example params, issue a correction
    if (
        action.action_type == "tool"
        and action.tool_name in ("statistical_significance", "bootstrap_confidence", "literature_baseline_compare")
        and _is_spec_example_params(action.tool_name, action.params, specs)
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
        if not _is_spec_example_params(action2.tool_name or "", action2.params, specs):
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
        code=(str(data.get("code")).strip() or None) if data.get("code") is not None else None,
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
    dl = context.deadline_monotonic
    if dl is not None and time.monotonic() >= dl:
        context.time_budget_exceeded = True
        return True
    if context.steps_taken >= context.max_steps:
        return True
    return False
