from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, TypedDict
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.claim_types import classify_claim_type
from propab.config import settings
from propab.tools.types import ToolError, ToolResult
from propab.paper_gate import SUBSTANTIVE_TOOL_NAMES
from propab.sandbox_profiles import effective_sandbox_timeout_sec
from propab.db import create_engine, create_redis, create_session_factory
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.sub_agent_plan import build_tool_plan_via_llm
from propab.tool_chain import refine_next_tool_step
from propab.tool_selection import select_tool_steps
from propab.tools.registry import ToolRegistry
from propab.types import EventType
from services.worker.domain_router import coerce_routed_domain, route_domain
from services.worker.failure_classify import classify_exception
from services.worker.peer_findings import poll_peer_findings
from services.worker.sandbox import run_sandboxed_python
from services.worker.sandbox_code_rewrite import (
    looks_like_heavy_training_code,
    rewrite_sandbox_code_after_timeout,
)
from services.worker.permutation_null import (
    compute_label_permutation_null,
    extract_two_group_arrays,
)
from services.worker.significance import (
    any_significance_tool_ran,
    check_significance,
    scan_verification,
)
from services.worker.think_act import (
    AgentContext,
    decide_next_action,
    should_stop,
)

logger = logging.getLogger(__name__)


def _sandbox_diag_tails(sandbox_out: dict[str, Any], *, maxlen: int = 1500) -> dict[str, Any]:
    """Short tails for code.timeout / code.error payloads (container stderr is the main signal)."""
    if not isinstance(sandbox_out, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("stderr", "stdout", "message"):
        raw = sandbox_out.get(key)
        if raw is None:
            continue
        s = raw if isinstance(raw, str) else str(raw)
        if not s.strip():
            continue
        out[f"{key}_tail"] = s[-maxlen:] if len(s) > maxlen else s
    et = sandbox_out.get("error_type")
    if et:
        out["sandbox_error_type"] = et
    return out


def _think_act_stub_code(code_desc: str) -> str:
    """
    Deterministic stub: must never embed raw ``code_description`` in a ``#`` line.

    A newline inside the description would end the comment and turn the rest into
    executable Python (LLM pasting a full training script → Docker wall timeouts).
    """
    desc = str(code_desc or "custom computation")
    return (
        "import json, sys\n"
        "result = {"
        f"'computation': {json.dumps(desc)}, "
        "'status': 'executed', 'sandbox': 'ok'"
        "}\n"
        "print(json.dumps(result))\n"
    )


def _is_sandbox_wall_timeout(sandbox_out: dict[str, Any]) -> bool:
    """
    True when Docker or the sandbox wrapper hit a wall-clock limit.

    ``run_sandboxed_python`` historically put everything in ``message``; some
    Docker SDK paths use ``Read timed out`` (no substring ``timeout``), which
    must still count as a wall timeout so we never treat it as a generic error
    and spin identical retries.
    """
    if not isinstance(sandbox_out, dict):
        return False
    et = str(sandbox_out.get("error_type", "") or "").lower()
    if "timeout" in et or et in {"docker_read_timeout", "docker_timeout"}:
        return True
    msg = str(sandbox_out.get("message", "") or "").lower()
    stderr = str(sandbox_out.get("stderr", "") or "").lower()
    blob = f"{msg} {stderr}"
    if "unexpected keyword argument 'timeout'" in blob:
        return False
    if "timeout" in blob or "timed out" in blob or "deadline exceeded" in blob:
        return True
    return False


def _is_trusted_inline_sandbox_code(code: str) -> bool:
    """
    Detect agent/heuristic stub snippets that are intentionally tiny JSON printers.
    These must never spend 480s in Docker — they were the dominant source of
    code.generated / code.timeout 1:1 ratios when the worker image/network stalled.

    Think–act stubs that embed ``json.dumps(code_description)`` may contain
    substrings like ``open(`` inside string literals; those are still safe but
    fail this heuristic — callers that built the stub programmatically should pass
    ``force_inline_trusted=True`` to ``run_code_step``.
    """
    s = (code or "").strip()
    if len(s) > 6000:
        return False
    blocked = ("subprocess", "socket", "urllib", "requests", "open(", "Path(", "__import__", "eval(", "exec(")
    if any(b in s for b in blocked):
        return False
    # Think–act stub: import json, sys + result dict + print(json.dumps(result))
    if (
        s.startswith("import json, sys")
        and "result = {" in s
        and "print(json.dumps(result))" in s
        and "'sandbox': 'ok'" in s
    ):
        return True
    # Heuristic tail: one-line print(json.dumps({...sandbox...}))
    if (
        "import json" in s
        and "print(json.dumps(" in s
        and "sandbox" in s
        and s.count("\n") <= 4
    ):
        return True
    return False


def _run_inline_trusted_sandbox_code(code: str) -> dict[str, Any]:
    """Execute trusted stub in-process (no Docker). Returns same shape as run_sandboxed_python."""
    import io
    import contextlib

    buf = io.StringIO()
    g: dict[str, Any] = {"json": json}
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<inline_stub>", "exec"), g, g)
    except Exception as exc:
        return {
            "ok": False,
            "error_type": "inline_stub_error",
            "message": str(exc),
            "stdout": buf.getvalue(),
            "stderr": "",
        }
    out = buf.getvalue()
    parsed = None
    for ln in reversed([x.strip() for x in out.splitlines() if x.strip()]):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                parsed = json.loads(ln)
                break
            except json.JSONDecodeError:
                continue
    if parsed is None and isinstance(g.get("result"), dict):
        parsed = g["result"]
    ok = isinstance(parsed, dict) and parsed.get("sandbox") == "ok"
    return {"ok": ok, "stdout": out, "stderr": "", "parsed": parsed}


_UTILITY_TOOL_NAMES = frozenset({"json_extract", "text_stats", "format_convert"})

# Tools that produce p_value / effect_size / confidence_interval
_SIGNIFICANCE_TOOL_NAMES = frozenset({
    "statistical_significance",
    "bootstrap_confidence",
    "literature_baseline_compare",
})

_MANDRAKE_TOOL_NAMES = frozenset({"mandrake_verification"})
_MATERIALS_TOOL_NAMES = frozenset({"materials_verification"})

_GENERIC_STATS_FOR_MANDRAKE = _SIGNIFICANCE_TOOL_NAMES


class HypothesisEvidence(TypedDict):
    metric_value: float | None
    baseline_value: float | None
    delta: float | None
    delta_pct: float | None
    p_value: float | None
    effect_size: float | None
    confidence_interval: list[float] | None
    n_tool_steps: int
    n_metric_steps: int
    relevance_score: float
    verdict_reason: str
    verified_true_steps: int
    verified_false_steps: int


def _heuristic_tool_plan_merged(
    specs: list[dict],
    *,
    hypothesis_text: str,
    hypothesis: dict,
) -> list[tuple[str, dict]]:
    rounds = max(1, int(settings.sub_agent_max_rounds))
    per = max(1, int(settings.sub_agent_tools_per_round))
    initial_ban = _UTILITY_TOOL_NAMES if len(specs) > 1 else frozenset()
    used_names: set[str] = set()
    merged: list[tuple[str, dict]] = []
    for _ in range(rounds):
        batch = select_tool_steps(
            specs,
            hypothesis_text=hypothesis_text,
            hypothesis=hypothesis,
            max_tools=per,
            exclude_tool_names=frozenset(used_names) | initial_ban,
        )
        before = len(merged)
        for tn, pr in batch:
            if tn in used_names:
                continue
            used_names.add(tn)
            merged.append((tn, pr))
        if len(merged) == before:
            break
    if not merged:
        return select_tool_steps(
            specs, hypothesis_text=hypothesis_text, hypothesis=hypothesis, max_tools=per,
        )
    return merged


def _extend_plan_with_heuristic_rounds(
    base: list[tuple[str, dict]],
    specs: list[dict],
    *,
    hypothesis_text: str,
    hypothesis: dict,
) -> list[tuple[str, dict]]:
    used_names = {t for t, _ in base}
    extra_rounds = max(0, int(settings.sub_agent_max_rounds) - 1)
    per = max(1, int(settings.sub_agent_tools_per_round))
    initial_ban = _UTILITY_TOOL_NAMES if len(specs) > 1 else frozenset()
    out = list(base)
    for _ in range(extra_rounds):
        batch = select_tool_steps(
            specs,
            hypothesis_text=hypothesis_text,
            hypothesis=hypothesis,
            max_tools=per,
            exclude_tool_names=frozenset(used_names) | initial_ban,
        )
        gained = False
        for tn, pr in batch:
            if tn in used_names:
                continue
            used_names.add(tn)
            out.append((tn, pr))
            gained = True
        if not gained:
            break
    return out


def _tokens(text_: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{4,}", text_.lower())}


_ACCURACY_KEY_SUBSTR = (
    "val_accuracy",
    "test_accuracy",
    "validation_accuracy",
    "accuracy",
    "val_acc",
    "test_acc",
)


def _primary_metric_from_tool_output(out: dict[str, Any]) -> float | None:
    """
    Pick a scalar metric from tool JSON without grabbing the first arbitrary number
    (e.g. p_value, learning_rate), which previously drove bogus ~0.01 'accuracies'.
    """
    if not isinstance(out, dict):
        return None
    for k in (
        "val_accuracy",
        "test_accuracy",
        "validation_accuracy",
        "accuracy",
        "metric_value",
        "mean_r2",
        "lofo_r2",
        "best_val_accuracy",
        "final_val_accuracy",
    ):
        v = out.get(k)
        fv: float | None = None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            fv = float(v)
        elif isinstance(v, str):
            vs = v.strip().replace("%", "")
            try:
                fv = float(vs)
            except ValueError:
                fv = None
        if fv is not None:
            fv = fv / 100.0 if fv > 1.001 and fv <= 100.0 else fv
            if 0.0 <= fv <= 1.0:
                return fv
    best_hint: float | None = None
    for path, val in _walk_numeric_values(out).items():
        pl = path.lower()
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        fv = float(val)
        if any(s in pl for s in _ACCURACY_KEY_SUBSTR) and fv > 1.01 and fv <= 100.0:
            fv = fv / 100.0
        if fv < 0.0 or fv > 1.0:
            continue
        if any(s in pl for s in _ACCURACY_KEY_SUBSTR):
            return fv
        if 0.2 <= fv <= 1.0:
            best_hint = fv if best_hint is None else max(best_hint, fv)
    return best_hint


def _walk_numeric_values(payload: Any, *, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_walk_numeric_values(v, prefix=key))
    elif isinstance(payload, list):
        nums = [x for x in payload if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if nums and len(nums) <= 128:
            out[prefix or "list"] = float(nums[-1])
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        out[prefix or "value"] = float(payload)
    return out


def _extract_ci(payload: dict[str, Any]) -> list[float] | None:
    for k in ("confidence_interval", "ci", "interval"):
        v = payload.get(k)
        if isinstance(v, list) and len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2]):
            return [float(v[0]), float(v[1])]
    lo, hi = payload.get("ci_lower"), payload.get("ci_upper")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return [float(lo), float(hi)]
    return None


def _first_key(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = payload.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


# ─── Significance-input provenance ────────────────────────────────────────────
#
# A significance tool (statistical_significance / bootstrap_confidence /
# literature_baseline_compare) computes REAL scipy statistics — but on whatever
# arrays the LLM agent hands it. If those arrays were TYPED by the agent rather
# than lifted from a prior real tool/sandbox output, the resulting p-value is
# real-looking but rests on fabricated data. These helpers record, at call time,
# whether a significance tool's numeric inputs TRACE to a prior computed output
# in the same run. The signal is surfaced on the evidence object so audits can
# see when a "confirmed" rests on agent-typed numbers.

# The numeric-array params each significance tool consumes.
_SIG_INPUT_ARRAY_KEYS: tuple[str, ...] = (
    "results_a",
    "results_b",
    "values",
    "our_results",
    "baseline_results",
)


def _collect_numeric_arrays(payload: Any, *, out: list[list[float]] | None = None) -> list[list[float]]:
    """Recursively collect every homogeneous numeric list (len>=2) inside payload."""
    if out is None:
        out = []
    if isinstance(payload, dict):
        for v in payload.values():
            _collect_numeric_arrays(v, out=out)
    elif isinstance(payload, list):
        nums = [x for x in payload if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if len(nums) == len(payload) and len(nums) >= 2:
            out.append([float(x) for x in nums])
        else:
            for v in payload:
                _collect_numeric_arrays(v, out=out)
    return out


def _array_traces_to_prior(arr: list[float], prior_arrays: list[list[float]]) -> bool:
    """
    True when `arr` matches a prior computed array exactly, or is a contiguous /
    subset slice of one (agents commonly pass val_losses[:20], or a reordering).
    Matching is by value with a small float tolerance.
    """
    if len(arr) < 2:
        return False

    def _val_close(x: float, y: float) -> bool:
        return abs(x - y) <= 1e-9 + 1e-6 * abs(y)

    def _seq_close(a: list[float], b: list[float]) -> bool:
        return len(a) == len(b) and all(_val_close(x, y) for x, y in zip(a, b))

    def _is_multiset_subset(small: list[float], big: list[float]) -> bool:
        """Every value in `small` is matched by a distinct value in `big` (order-free)."""
        remaining = list(big)
        for x in small:
            for i, y in enumerate(remaining):
                if _val_close(x, y):
                    remaining.pop(i)
                    break
            else:
                return False
        return True

    for prior in prior_arrays:
        if len(prior) < len(arr):
            continue
        if _seq_close(arr, prior):
            return True
        # Reordered / sliced subset of a prior computed array (e.g. val_losses[:k]).
        if _is_multiset_subset(arr, prior):
            return True
    return False


def _classify_stat_input_provenance(
    params: dict[str, Any],
    prior_outputs: list[dict[str, Any]],
) -> str:
    """
    Classify a significance tool's numeric-array inputs against arrays produced by
    prior real tool/sandbox outputs in the same run.

    Returns:
      "computed"      — at least one input array traces to a prior real output
                        AND no input array is unaccounted-for (all trace back).
      "agent_literal" — at least one numeric-array input exists but NONE of them
                        trace to any prior real output (agent-typed numbers).
      "unknown"       — no numeric-array inputs present, or no prior outputs to
                        compare against (cannot decide either way).
    """
    input_arrays: list[list[float]] = []
    for k in _SIG_INPUT_ARRAY_KEYS:
        v = params.get(k)
        if isinstance(v, list):
            nums = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
            if len(nums) == len(v) and len(nums) >= 2:
                input_arrays.append([float(x) for x in nums])

    if not input_arrays:
        return "unknown"

    prior_arrays: list[list[float]] = []
    for out in prior_outputs:
        _collect_numeric_arrays(out, out=prior_arrays)
    if not prior_arrays:
        # No real computed arrays exist yet → cannot corroborate. This is itself a
        # red flag, but we do not have positive evidence of fabrication, so report
        # "agent_literal" only if there is genuinely nothing upstream to trace to.
        return "agent_literal"

    traced = [_array_traces_to_prior(a, prior_arrays) for a in input_arrays]
    if all(traced):
        return "computed"
    if any(traced):
        # Mixed: some inputs are real, some are not. Treat as agent_literal so a
        # partially-fabricated comparison is never labeled fully "computed".
        return "agent_literal"
    return "agent_literal"


def _build_evidence(
    *,
    successful_outputs: list[dict[str, Any]],
    relevance_score: float,
    n_tool_steps: int,
    baseline_value: float | None,
) -> HypothesisEvidence:
    metric_value: float | None = None
    p_value: float | None = None
    effect_size: float | None = None
    ci = None
    metric_steps = 0
    for out in successful_outputs:
        cand: float | None = None
        if isinstance(out, dict):
            cand = _primary_metric_from_tool_output(out)
        if cand is not None:
            metric_steps += 1
            metric_value = cand
        if p_value is None:
            p_value = _first_key(out, ("p_value", "p", "pvalue"))
        if effect_size is None:
            effect_size = _first_key(out, ("effect_size", "cohens_d", "d"))
        if ci is None:
            ci = _extract_ci(out)
    delta = None if (metric_value is None or baseline_value is None) else (metric_value - baseline_value)
    delta_pct = None
    if delta is not None and baseline_value not in (None, 0.0):
        delta_pct = (delta / baseline_value) * 100.0
    n_verified_true, n_verified_false = scan_verification(successful_outputs)
    return HypothesisEvidence(
        metric_value=metric_value,
        baseline_value=baseline_value,
        delta=delta,
        delta_pct=delta_pct,
        p_value=p_value,
        effect_size=effect_size,
        confidence_interval=ci,
        n_tool_steps=n_tool_steps,
        n_metric_steps=metric_steps,
        relevance_score=float(relevance_score),
        verdict_reason="",
        verified_true_steps=n_verified_true,
        verified_false_steps=n_verified_false,
    )


def attach_permutation_null_to_evidence(
    evidence_obj: dict[str, Any],
    sig_call_arrays: list[tuple[list[float], list[float]]],
) -> dict[str, Any]:
    """Compute a real label-permutation null from captured two-group arrays and
    attach ``permutation_p`` + ``n_samples`` to ``evidence_obj`` (D2). Mutates and
    returns ``evidence_obj``.

    ``sig_call_arrays`` holds only pairs where BOTH real outcome arrays were
    present at a successful significance-tool call (see ``extract_two_group_arrays``).
    This is the ONLY place a generic (non-LOFO) statistical result acquires the
    adversarial null that ``artifact_verification._survives_permutation`` requires.

    Integrity invariants:
      * Fail-closed — if ``sig_call_arrays`` is empty, or the arrays can't ground a
        null (``compute_label_permutation_null`` returns ``None``), NOTHING is
        attached: ``permutation_p`` stays absent and the verdict pipeline correctly
        keeps the result inconclusive. No synthesized/self-reported null is ever
        emitted.
      * The p-value is a deterministic function of the SAME real arrays the observed
        effect was measured on, under a fixed seed.
      * The caller's ``stat_input_provenance`` field (if any) is left untouched —
        an ``agent_literal`` tag on untrusted inputs survives so a later gate can
        reject the verdict. Attaching a real null never launders that tag.
      * Group-mean metric fields are filled ONLY when no real metric was found, and
        never overwrite an existing measured ``metric_value``.
    """
    if not sig_call_arrays:
        return evidence_obj

    # Largest captured comparison (most observations -> most trustworthy null);
    # ties keep the earliest call.
    ra, rb = max(sig_call_arrays, key=lambda pair: len(pair[0]) + len(pair[1]))
    perm = compute_label_permutation_null(ra, rb)
    if perm is None:
        return evidence_obj

    fields = perm.to_evidence_fields()
    if evidence_obj.get("n_samples") is None:
        evidence_obj["n_samples"] = fields["n_samples"]
    evidence_obj["permutation_p"] = fields["permutation_p"]
    evidence_obj["permutation_null_n_permutations"] = fields["permutation_null_n_permutations"]
    evidence_obj["permutation_null_observed_stat"] = fields["permutation_null_observed_stat"]
    evidence_obj["permutation_null_group_sizes"] = fields["permutation_null_group_sizes"]

    # A pure two-group outcome experiment may carry a significance p_value but no
    # accuracy-style metric_value, so _build_evidence leaves metric fields unset
    # and the evidence classifies as "unknown" (never reaching the artifact gate).
    # The honest metric here is the group means of the SAME real arrays the null
    # was computed from. Fill ONLY when no real metric was found; never overwrite
    # a measured metric, never invent direction beyond the observed means.
    if evidence_obj.get("metric_value") is None:
        mean_a = sum(ra) / len(ra)
        mean_b = sum(rb) / len(rb)
        evidence_obj["metric_value"] = float(mean_a)
        evidence_obj["baseline_value"] = float(mean_b)
        evidence_obj["delta"] = float(mean_a - mean_b)
        if mean_b != 0.0:
            evidence_obj["delta_pct"] = float((mean_a - mean_b) / mean_b * 100.0)
        evidence_obj["n_metric_steps"] = max(1, int(evidence_obj.get("n_metric_steps") or 0))
        evidence_obj["metric_from_permutation_groups"] = True

    return evidence_obj


def _build_mandrake_evidence(
    *,
    output: dict[str, Any],
    verdict: str,
    reason: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Structured evidence for campaign tree diagnostics and paper trace."""
    metric_value = output.get("mean_r2", output.get("lofo_r2"))
    if metric_value is not None:
        metric_value = float(metric_value)
    p_value = output.get("permutation_p")
    if p_value is not None:
        p_value = float(p_value)
    baseline_value = baseline.get("value")
    if baseline_value is None:
        fam = output.get("family_baseline_r2")
        baseline_value = float(fam) if fam is not None else None
    lofo_gap = output.get("lofo_gap")
    effect_size = float(lofo_gap) if lofo_gap is not None else None
    evidence_obj: dict[str, Any] = {
        "metric_value": metric_value,
        "baseline_value": baseline_value,
        "p_value": p_value,
        "effect_size": effect_size,
        "lofo_r2": metric_value,
        "lofo_gap": effect_size,
        "n_samples": output.get("n_samples"),
        "n_families": output.get("n_families"),
        "methodology": output.get("methodology") or "LOFO",
        "feature_subset": output.get("feature_subset"),
        "n_tool_steps": 1,
        "n_metric_steps": 1 if metric_value is not None else 0,
        "verdict_reason": reason,
        "verification_tool": "mandrake_verification",
        "label_shuffle_permutation_p": output.get("label_shuffle_permutation_p"),
        "label_shuffle_null_p95": output.get("label_shuffle_null_p95"),
    }
    if verdict == "confirmed":
        evidence_obj["verified_true_steps"] = 1
    elif verdict == "refuted":
        evidence_obj["verified_false_steps"] = 1
    if metric_value is not None and baseline_value is not None:
        evidence_obj["delta"] = metric_value - float(baseline_value)
    if output.get("label_shuffle_permutation_p") is not None:
        lp = float(output["label_shuffle_permutation_p"])
        lofo = metric_value
        p95 = output.get("label_shuffle_null_p95")
        if lofo is not None and p95 is not None and float(lofo) > float(p95) and lp < 0.05:
            evidence_obj["ood_passed"] = True
            evidence_obj["ood_reason"] = f"label-shuffle LOFO={float(lofo):.3f} p={lp:.3f}"
        else:
            evidence_obj["ood_passed"] = False
            evidence_obj["ood_reason"] = f"LOFO OOD failed lofo={lofo} p95={p95} label_p={lp}"

    from propab.scoped_claim import (
        check_scope_executed_integrity,
        extract_executed_ood_from_experiment,
        parse_scope_from_methodology,
    )
    executed = extract_executed_ood_from_experiment(output, code=str(output.get("executed_code") or ""))
    if executed:
        evidence_obj["executed_ood"] = executed.to_dict()
    return evidence_obj


def _build_materials_evidence(
    *,
    output: dict[str, Any],
    verdict: str,
    reason: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    metric_value = output.get("mean_r2", output.get("lofo_r2"))
    if metric_value is not None:
        metric_value = float(metric_value)
    p_value = output.get("permutation_p")
    if p_value is not None:
        p_value = float(p_value)
    baseline_value = baseline.get("value")
    if baseline_value is None:
        fam = output.get("family_baseline_r2")
        baseline_value = float(fam) if fam is not None else None
    lofo_gap = output.get("lofo_gap")
    effect_size = float(lofo_gap) if lofo_gap is not None else None
    evidence_obj: dict[str, Any] = {
        "metric_value": metric_value,
        "baseline_value": baseline_value,
        "p_value": p_value,
        "effect_size": effect_size,
        "lofo_r2": metric_value,
        "lofo_gap": effect_size,
        "n_samples": output.get("n_samples"),
        "n_families": output.get("n_families"),
        "group_column": output.get("group_column") or "crystal_system",
        "methodology": output.get("methodology") or "LOFO",
        "feature_subset": output.get("feature_subset"),
        "n_tool_steps": 1,
        "n_metric_steps": 1 if metric_value is not None else 0,
        "verdict_reason": reason,
        "verification_tool": "materials_verification",
        "label_shuffle_permutation_p": output.get("label_shuffle_permutation_p"),
        "label_shuffle_null_p95": output.get("label_shuffle_null_p95"),
        "family_leakage_confirmed": output.get("family_leakage_confirmed"),
    }
    if verdict == "confirmed":
        evidence_obj["verified_true_steps"] = 1
    elif verdict == "refuted":
        evidence_obj["verified_false_steps"] = 1
    if metric_value is not None and baseline_value is not None:
        evidence_obj["delta"] = metric_value - float(baseline_value)
    return evidence_obj


def attach_scope_integrity(
    evidence_obj: dict[str, Any],
    *,
    hypothesis_text: str,
    test_methodology: str,
    experiment_output: dict[str, Any] | None,
    question: str = "",
    code: str = "",
) -> dict[str, Any]:
    """P0 — verify declared OOD ≈ executed OOD; attach to evidence."""
    from propab.scoped_claim import (
        check_scope_executed_integrity,
        extract_executed_ood_from_experiment,
        parse_scope_from_methodology,
    )

    scope = parse_scope_from_methodology(hypothesis_text, test_methodology)
    executed = extract_executed_ood_from_experiment(experiment_output or {}, code=code)
    integrity = check_scope_executed_integrity(scope, executed, question=question)
    out = dict(evidence_obj)
    out["scope_integrity"] = integrity.to_dict()
    out["scope_gate_result"] = integrity.scope_gate_result
    if integrity.scope_gate_result == "FAIL" and out.get("verdict_reason"):
        out["verdict_reason"] = f"{out['verdict_reason']}; scope integrity: {integrity.reason}"
    return out


def _compute_confidence(evidence: HypothesisEvidence) -> float:
    logger.info(
        "compute_confidence called: n_metric_steps=%s, p_value=%s, effect_size=%s, delta_pct=%s",
        evidence.get("n_metric_steps"),
        evidence.get("p_value"),
        evidence.get("effect_size"),
        evidence.get("delta_pct"),
    )
    score = 0.0
    if evidence["metric_value"] is not None:
        score += 0.20
    if evidence["baseline_value"] is not None:
        score += 0.20
    p = evidence["p_value"]
    if p is not None and p < 0.05:
        score += 0.25
    es = evidence["effect_size"]
    if es is not None and abs(es) > 0.2:
        score += 0.15
    if evidence["n_metric_steps"] >= 3:
        score += 0.10
    if evidence["relevance_score"] > 0.30:
        score += 0.10
    return min(max(score, 0.0), 0.95)


def _hypothesis_relevance_score(hypothesis_text: str, successful_outputs: list[dict]) -> float:
    if not successful_outputs:
        return 0.0
    hyp_toks = _tokens(hypothesis_text)
    if not hyp_toks:
        return 0.0
    blob = json.dumps(successful_outputs, ensure_ascii=False).lower()
    out_toks = _tokens(blob)
    overlap = len(hyp_toks & out_toks) / float(len(hyp_toks))
    evidence_keys = ("conclusion", "verdict", "significant", "p_value", "improvement",
                     "confidence_interval", "summary")
    key_bonus = 0.02 * sum(1 for k in evidence_keys if k in blob)
    return float(overlap + key_bonus)


async def _update_hypothesis(
    session_factory: async_sessionmaker,
    hypothesis_id: str,
    *,
    status: str,
    verdict: str | None = None,
    confidence: float | None = None,
    evidence_summary: str | None = None,
    key_finding: str | None = None,
    tool_trace_id: str | None = None,
) -> None:
    fields: list[str] = ["status = :status"]
    params: dict = {"id": hypothesis_id, "status": status}
    if verdict is not None:
        fields.append("verdict = :verdict")
        params["verdict"] = verdict
    if confidence is not None:
        fields.append("confidence = :confidence")
        params["confidence"] = confidence
    if evidence_summary is not None:
        fields.append("evidence_summary = :evidence_summary")
        params["evidence_summary"] = evidence_summary
    if key_finding is not None:
        fields.append("key_finding = :key_finding")
        params["key_finding"] = key_finding
    if tool_trace_id is not None:
        fields.append("tool_trace_id = :tool_trace_id")
        params["tool_trace_id"] = tool_trace_id
    query = text(f"UPDATE hypotheses SET {', '.join(fields)} WHERE id = :id")
    async with session_factory() as session:
        await session.execute(query, params)
        await session.commit()


async def _insert_experiment_step_tool(
    session_factory: async_sessionmaker,
    *,
    step_id: str,
    hypothesis_id: str,
    step_index: int,
    tool_name: str,
    params: dict,
    result_output: Any,
    result_error: Any,
    duration_ms: int,
) -> None:
    async with session_factory() as session:
        await session.execute(
            text("""
                INSERT INTO experiment_steps
                    (id, hypothesis_id, step_type, step_index, input_json, output_json, error_json, duration_ms)
                VALUES
                    (:id, :hypothesis_id, 'tool_call', :step_index,
                     CAST(:input_json AS jsonb), CAST(:output_json AS jsonb),
                     CAST(:error_json AS jsonb), :duration_ms)
            """),
            {
                "id": step_id,
                "hypothesis_id": hypothesis_id,
                "step_index": step_index,
                "input_json": json.dumps({"tool": tool_name, "params": params}),
                "output_json": json.dumps(result_output) if result_output is not None else "null",
                "error_json": json.dumps(result_error) if result_error is not None else "null",
                "duration_ms": duration_ms,
            },
        )
        await session.execute(
            text("""
                INSERT INTO tool_calls
                    (id, step_id, hypothesis_id, tool_name, domain, params_json, result_json, success, duration_ms)
                VALUES
                    (:id, :step_id, :hypothesis_id, :tool_name, :domain,
                     CAST(:params_json AS jsonb), CAST(:result_json AS jsonb), :success, :duration_ms)
            """),
            {
                "id": str(uuid4()),
                "step_id": step_id,
                "hypothesis_id": hypothesis_id,
                "tool_name": tool_name,
                "domain": "unknown",
                "params_json": json.dumps(params),
                "result_json": json.dumps(result_output) if result_output is not None else "null",
                "success": result_error is None,
                "duration_ms": duration_ms,
            },
        )
        await session.commit()


async def _insert_experiment_step_code(
    session_factory: async_sessionmaker,
    *,
    step_id: str,
    hypothesis_id: str,
    step_index: int,
    code: str,
    parsed_output: Any,
    sandbox_out: dict,
    duration_ms: int,
    memory_mb: int,
    timeout_sec: int,
) -> None:
    async with session_factory() as session:
        await session.execute(
            text("""
                INSERT INTO experiment_steps
                    (id, hypothesis_id, step_type, step_index, input_json, output_json, error_json,
                     duration_ms, memory_mb, timeout_sec)
                VALUES
                    (:id, :hypothesis_id, 'code_exec', :step_index,
                     CAST(:input_json AS jsonb), CAST(:output_json AS jsonb),
                     CAST(:error_json AS jsonb), :duration_ms, :memory_mb, :timeout_sec)
            """),
            {
                "id": step_id,
                "hypothesis_id": hypothesis_id,
                "step_index": step_index,
                "input_json": json.dumps({"code": code}),
                "output_json": json.dumps({"parsed": parsed_output, "stdout": sandbox_out.get("stdout")}),
                "error_json": json.dumps(sandbox_out) if not sandbox_out.get("ok") else "null",
                "duration_ms": duration_ms,
                "memory_mb": memory_mb,
                "timeout_sec": timeout_sec,
            },
        )
        await session.commit()


async def _hydrate_hypothesis_from_campaign(
    hypothesis: dict,
    *,
    session_id: str,
    campaign_node_id: str | None,
    session_factory: async_sessionmaker,
) -> dict:
    """Load test_methodology / feature_subset from campaign tree node (fixes.md P2)."""
    if not campaign_node_id:
        return hypothesis
    try:
        from propab.campaign_db import db_load_campaign

        campaign = await db_load_campaign(session_id, session_factory)
        node = campaign.hypothesis_tree.nodes.get(campaign_node_id)
        if not node:
            return hypothesis
        h = dict(hypothesis)
        if node.test_methodology:
            h["test_methodology"] = node.test_methodology
        if node.feature_subset:
            h["feature_subset"] = list(node.feature_subset)
        if node.mechanism_id:
            h["mechanism_id"] = node.mechanism_id
        return h
    except Exception as exc:  # noqa: BLE001
        logger.warning("Campaign node hydration failed: %s", exc)
        return hypothesis


async def _mandrake_verification_path(
    *,
    payload: dict,
    hypothesis: dict,
    hypothesis_id: str,
    campaign_node_id: str | None,
    session_id: str,
    question: str,
    session_factory: async_sessionmaker,
    emitter: EventEmitter,
    registry: ToolRegistry,
    trace_pointer: str,
    started: float,
    baseline: dict,
) -> dict:
    """Run real LOFO via mandrake_verification; skip generic stats (fixes.md P3/P4)."""
    from propab.domain_adapters.mandrake_adapter import (
        MandrakeExperimentSpec,
        classify_mandrake_verdict,
    )

    spec = MandrakeExperimentSpec.from_hypothesis(hypothesis, question=question)
    art_dir = f"/data/propab/sessions/{session_id}/mandrake/{hypothesis_id}"
    tool_result = registry.call(
        "mandrake_verification",
        {**spec.to_tool_params(), "artifacts_dir": art_dir},
    )

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.TOOL_CALLED,
        step=f"experiment.{hypothesis_id}.mandrake",
        payload={"tool": "mandrake_verification", "features": spec.feature_subset},
        hypothesis_id=hypothesis_id,
    )

    if not tool_result.success or not isinstance(tool_result.output, dict):
        err = tool_result.error.message if tool_result.error else "mandrake_verification failed"
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.TOOL_ERROR,
            step=f"experiment.{hypothesis_id}.mandrake",
            payload={"tool": "mandrake_verification", "error": err},
            hypothesis_id=hypothesis_id,
        )
        verdict, reason, confidence = "inconclusive", f"LOFO experiment failed: {err}", 0.0
        output: dict = {}
        gate = None
    else:
        output = tool_result.output
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.TOOL_RESULT,
            step=f"experiment.{hypothesis_id}.mandrake",
            payload={"tool": "mandrake_verification", "output": output},
            hypothesis_id=hypothesis_id,
        )
        verdict, reason, confidence = classify_mandrake_verdict(str(hypothesis.get("text") or ""), output)
        from propab.artifact_verification import (
            apply_artifact_gate_override,
            evidence_context_from_hypothesis,
            merge_artifact_into_evidence,
        )
        ctx = evidence_context_from_hypothesis(
            str(hypothesis.get("text") or ""),
            {
                "metric_value": output.get("mean_r2"),
                "lofo_r2": output.get("mean_r2"),
                "lofo_gap": output.get("lofo_gap"),
                "p_value": output.get("permutation_p"),
                "n_samples": output.get("n_samples"),
                "n_families": output.get("n_families"),
                "methodology": "LOFO",
                "feature_subset": output.get("feature_subset"),
                "label_shuffle_permutation_p": output.get("label_shuffle_permutation_p"),
                "label_shuffle_null_p95": output.get("label_shuffle_null_p95"),
            },
            methodology="LOFO",
        )
        verdict, reason, confidence, gate = apply_artifact_gate_override(
            verdict, reason, confidence, ctx, output,
        )

    evidence_obj = _build_mandrake_evidence(
        output=output,
        verdict=verdict,
        reason=reason,
        baseline=baseline,
    )
    evidence_obj = attach_scope_integrity(
        evidence_obj,
        hypothesis_text=str(hypothesis.get("text") or ""),
        test_methodology=str(hypothesis.get("test_methodology") or spec.methodology),
        experiment_output=output,
        question=question,
        code=str(output.get("executed_code") or ""),
    )
    if gate is not None:
        evidence_obj = merge_artifact_into_evidence(evidence_obj, gate)
    from propab.scoped_claim import apply_ood_gate_to_verdict
    verdict, reason = apply_ood_gate_to_verdict(
        verdict, reason, evidence_obj,
        hypothesis_text=str(hypothesis.get("text") or ""),
        test_methodology=str(hypothesis.get("test_methodology") or spec.methodology),
    )
    if evidence_obj.get("scope_gate_result") == "FAIL" and verdict == "confirmed":
        verdict = "inconclusive"
        reason = f"scope integrity fail: {evidence_obj.get('scope_integrity', {}).get('reason', '?')}"
    evidence_obj["verdict_reason"] = reason
    evidence = (
        f"evidence={json.dumps(evidence_obj, ensure_ascii=False)}; "
        f"mandrake_verification; LOFO={output.get('mean_r2')}; "
        f"gap={output.get('lofo_gap')}; p_perm={output.get('permutation_p')}; "
        f"features={output.get('feature_subset', spec.feature_subset)}"
    )
    key_finding = None
    if verdict == "confirmed":
        key_finding = str(hypothesis.get("text") or "")[:300]

    duration_ms = int((time.perf_counter() - started) * 1000)
    await _insert_experiment_step_tool(
        session_factory,
        step_id=str(uuid4()),
        hypothesis_id=hypothesis_id,
        step_index=0,
        tool_name="mandrake_verification",
        params={**spec.to_tool_params(), "artifacts_dir": art_dir},
        result_output=output if output else None,
        result_error=(
            {"message": tool_result.error.message}
            if not tool_result.success and tool_result.error
            else None
        ),
        duration_ms=duration_ms,
    )

    await _update_hypothesis(
        session_factory,
        hypothesis_id,
        status="completed",
        verdict=verdict,
        confidence=confidence,
        evidence_summary=evidence,
        key_finding=key_finding,
        tool_trace_id=trace_pointer,
    )
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.AGENT_COMPLETED,
        step=f"experiment.{hypothesis_id}.complete",
        payload={
            "verdict": verdict,
            "confidence": confidence,
            "sig_gate_passed": verdict in {"confirmed", "refuted"},
            "mandrake": True,
            "mean_r2": output.get("mean_r2"),
        },
        hypothesis_id=hypothesis_id,
    )

    metric_key = str(baseline.get("metric_name") or "lofo_r2").strip()
    result_dict: dict[str, Any] = {
        "hypothesis_id": hypothesis_id,
        "campaign_node_id": campaign_node_id,
        "verdict": verdict,
        "confidence": confidence,
        "evidence_summary": evidence,
        "key_finding": key_finding,
        "tool_trace_id": trace_pointer,
        "figures": [],
        "duration_sec": round(time.perf_counter() - started, 3),
        "failure_reason": None if verdict != "inconclusive" else reason,
        "learned": reason,
        "metric_value": output.get("mean_r2"),
        "mandrake_artifacts_dir": art_dir,
    }
    if output.get("mean_r2") is not None:
        result_dict[metric_key] = output.get("mean_r2")
    return result_dict


async def _plugin_verification_path(
    *,
    payload: dict,
    hypothesis: dict,
    hypothesis_id: str,
    campaign_node_id: str | None,
    session_id: str,
    question: str,
    session_factory: async_sessionmaker,
    emitter: EventEmitter,
    registry: ToolRegistry,
    trace_pointer: str,
    started: float,
    baseline: dict,
    domain_plugin: Any,
) -> dict:
    """Generic worker path: delegate experiment + verdict to a DomainPlugin."""
    import asyncio
    import json

    from propab.verdict_pipeline import (
        artifact_gate_stage,
        classify_evidence_type,
        ood_gate_stage,
        run_verdict_pipeline,
        scope_integrity_stage,
    )

    domain_id = domain_plugin.domain_id
    hypothesis_text = str(hypothesis.get("text") or "")
    test_methodology = str(hypothesis.get("test_methodology") or "")
    hyp_dict = {
        "text": hypothesis_text,
        "statement": hypothesis_text,
        "test_methodology": test_methodology,
        "feature_subset": list(hypothesis.get("feature_subset") or []),
    }
    features = list(hypothesis.get("feature_subset") or [])

    # DOM4 honesty gate: before running the domain's fixed-default verification,
    # confirm the hypothesis is actually ON-TOPIC for this domain. Without this,
    # an off-topic / misrouted hypothesis is verified against the domain's default
    # feature pair (e.g. graph_invariants' spectral_gap->clustering) and can yield
    # a "confirmed" verdict decoupled from the real claim. ``hypothesis_on_topic``
    # is defined on the DomainPlugin base (default: accept all), so every plugin
    # has it; we still guard with getattr + a fail-OPEN except so a plugin that
    # lacks or breaks the method is verified exactly as before (never a false
    # refusal).
    on_topic_check = getattr(domain_plugin, "hypothesis_on_topic", None)
    if callable(on_topic_check):
        try:
            on_topic = bool(on_topic_check(hypothesis_text, methodology=test_methodology))
        except Exception:  # noqa: BLE001 — a broken on-topic check must not block verification
            on_topic = True
        if not on_topic:
            reason = "hypothesis_off_topic_for_domain"
            off_topic_evidence = json.dumps({"reason": reason, "domain": domain_id})
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.AGENT_COMPLETED,
                step=f"experiment.{hypothesis_id}.complete",
                payload={
                    "verdict": "inconclusive",
                    "confidence": 0.0,
                    "domain": domain_id,
                    "off_topic": True,
                    "reason": reason,
                },
                hypothesis_id=hypothesis_id,
            )
            await _update_hypothesis(
                session_factory,
                hypothesis_id,
                status="completed",
                verdict="inconclusive",
                confidence=0.0,
                evidence_summary=off_topic_evidence,
                key_finding=None,
                tool_trace_id=trace_pointer,
            )
            return {
                "hypothesis_id": hypothesis_id,
                "campaign_node_id": campaign_node_id,
                "verdict": "inconclusive",
                "confidence": 0.0,
                "evidence_summary": off_topic_evidence,
                "key_finding": None,
                "tool_trace_id": trace_pointer,
                "figures": [],
                "duration_sec": round(time.perf_counter() - started, 3),
                "failure_reason": reason,
                "learned": reason,
            }

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.TOOL_CALLED,
        step=f"experiment.{hypothesis_id}.{domain_id}",
        payload={"tool": f"{domain_id}_verification", "domain": domain_id, "features": features},
        hypothesis_id=hypothesis_id,
    )

    try:
        output = await asyncio.to_thread(
            domain_plugin.run_verification,
            hyp_dict,
            None,
            features or None,
        )
    except Exception as exc:  # noqa: BLE001
        err = str(exc)[:500]
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.TOOL_ERROR,
            step=f"experiment.{hypothesis_id}.{domain_id}",
            payload={"tool": f"{domain_id}_verification", "error": err},
            hypothesis_id=hypothesis_id,
        )
        return {
            "hypothesis_id": hypothesis_id,
            "campaign_node_id": campaign_node_id,
            "verdict": "inconclusive",
            "confidence": 0.0,
            "evidence_summary": json.dumps({"error": err, "domain": domain_id}),
            "key_finding": err,
            "tool_trace_id": trace_pointer,
            "figures": [],
            "duration_sec": round(time.perf_counter() - started, 3),
            "failure_reason": err,
            "learned": err,
        }

    if not isinstance(output, dict):
        output = {"raw": output}

    # DOM2 honesty: stamp synthetic-data provenance onto the evidence so the
    # verdict/paper pipeline can label a finding backed by a seed-generated
    # (illustrative) dataset — never presenting it as a real-world result. The
    # adapters record ``synthetic: True`` in their cache meta; the plugin surfaces
    # that via ``uses_synthetic_data()``. Real-data domains report False and this
    # field is omitted.
    try:
        if domain_plugin.uses_synthetic_data():
            output["data_provenance"] = "synthetic"
    except Exception:  # noqa: BLE001 — provenance labelling must never break a run
        pass

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.TOOL_RESULT,
        step=f"experiment.{hypothesis_id}.{domain_id}",
        payload={"tool": f"{domain_id}_verification", "output": output, "deterministic": output.get("deterministic")},
        hypothesis_id=hypothesis_id,
    )

    criteria = domain_plugin.confirmation_criteria()
    min_steps = int(criteria.get("min_metric_steps_for_confirm") or 1)
    try:
        verdict, reason, confidence = domain_plugin.classify_verdict(
            str(hypothesis.get("text") or ""), output,
        )
    except Exception:
        verdict, confidence, reason = run_verdict_pipeline(
            output,
            hypothesis=hypothesis,
            campaign_context={"min_metric_steps": min_steps},
        )

    # F1 honesty gate: a DomainPlugin's own classify_verdict is not adversarially
    # gated on its own. Historically only the materials/mandrake worker paths ran
    # the artifact gate, so every plugin-path domain (math, graph, genomics,
    # enzyme, ...) could emit a "confirmed" verdict that never faced a
    # permutation / label-shuffle null. Route a plugin "confirmed" through the same
    # shape-aware artifact gate the except-branch (run_verdict_pipeline) already
    # applies. This stage is DOMAIN-GENERAL: it keys on evidence SHAPE via
    # classify_evidence_type, never on domain id — a deterministic proof passes
    # through untouched, a lofo/statistical confirm must survive a real adversarial
    # null (label-shuffle / permutation), and shapeless "unknown" evidence cannot
    # confirm. Downgrade-only; no-ops unless verdict == "confirmed".
    #
    # For DISTRIBUTIONAL (lofo/statistical) plugin evidence we ALSO chain the OOD
    # gate + scope-integrity stage that the generic/mandrake/materials paths run —
    # the artifact gate alone does not catch a scope-inflated confirm. A
    # deterministic proof (math/coding_theory) is left untouched: OOD wrongly
    # collapses a proof to inconclusive. Keyed on evidence SHAPE via
    # classify_evidence_type, never on domain id.
    #
    # FAIL-CLOSED contract (matches the generic run_verdict_pipeline path, which has
    # no try/except): the gated verdict is applied to (verdict, confidence, reason)
    # BEFORE the best-effort downgrade notification is emitted, and ANY exception in
    # the gate downgrades to "inconclusive". Previously the emit fired first and the
    # assignment second, both inside one broad ``try/except: pass`` — so a transient
    # redis/DB failure during the emit (which fires ONLY when a confirm is being
    # REJECTED) swallowed the error and left the rejected verdict standing as
    # "confirmed". A gate error, or a failing emit, must never leave a confirm.
    if verdict == "confirmed":
        _orig_verdict = verdict
        try:
            _gv, _gc, _gr = artifact_gate_stage(
                output, verdict, confidence, reason,
                campaign_context={"min_metric_steps": min_steps},
            )
            if _gv == "confirmed" and classify_evidence_type(output) in ("lofo", "statistical"):
                _gv, _gc, _gr = ood_gate_stage(
                    output, _gv, _gc, _gr,
                    hypothesis=hypothesis,
                    campaign_context={
                        "hyp_text": hypothesis_text,
                        "test_methodology": test_methodology,
                    },
                )
                if _gv == "confirmed":
                    _scoped = attach_scope_integrity(
                        output,
                        hypothesis_text=hypothesis_text,
                        test_methodology=test_methodology,
                        experiment_output=output,
                        question=question,
                        code=str(output.get("executed_code") or ""),
                    )
                    output["scope_integrity"] = _scoped.get("scope_integrity")
                    output["scope_gate_result"] = _scoped.get("scope_gate_result")
                    _gv, _gc, _gr = scope_integrity_stage(
                        output, _gv, _gc, _gr, hypothesis=hypothesis,
                    )
        except Exception:  # noqa: BLE001 — FAIL CLOSED: never leave a confirm standing on gate error
            _gv, _gc, _gr = (
                "inconclusive",
                0.0,
                "artifact/scope gate raised; failing closed (verdict downgraded to inconclusive)",
            )
        # Apply the (possibly downgraded) verdict BEFORE any I/O so a failing emit
        # can never resurrect a rejected "confirmed".
        _downgraded = _gv != _orig_verdict
        verdict, confidence, reason = _gv, _gc, _gr
        if _downgraded:
            try:
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.TOOL_RESULT,
                    step=f"experiment.{hypothesis_id}.artifact_gate",
                    payload={
                        "domain": domain_id,
                        "gated_from": _orig_verdict,
                        "gated_to": _gv,
                        "reason": str(_gr)[:300],
                    },
                    hypothesis_id=hypothesis_id,
                )
            except Exception:  # noqa: BLE001 — emit failure must not undo the downgrade already applied
                pass

    evidence = json.dumps(output, default=str)
    key_finding = str(output.get("notes") or reason or "")[:500] if verdict == "confirmed" else None

    duration_ms = int((time.perf_counter() - started) * 1000)
    await _insert_experiment_step_tool(
        session_factory,
        step_id=str(uuid4()),
        hypothesis_id=hypothesis_id,
        step_index=0,
        tool_name=f"{domain_id}_verification",
        params={"domain": domain_id, "features": features},
        result_output=output,
        result_error=None,
        duration_ms=duration_ms,
    )

    await _update_hypothesis(
        session_factory,
        hypothesis_id,
        status="completed",
        verdict=verdict,
        confidence=confidence,
        evidence_summary=evidence,
        key_finding=key_finding,
        tool_trace_id=trace_pointer,
    )
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.AGENT_COMPLETED,
        step=f"experiment.{hypothesis_id}.complete",
        payload={
            "verdict": verdict,
            "confidence": confidence,
            "domain": domain_id,
            "deterministic": output.get("deterministic"),
            "metric_value": output.get("metric_value"),
        },
        hypothesis_id=hypothesis_id,
    )

    return {
        "hypothesis_id": hypothesis_id,
        "campaign_node_id": campaign_node_id,
        "verdict": verdict,
        "confidence": confidence,
        "evidence_summary": evidence,
        "key_finding": key_finding,
        "tool_trace_id": trace_pointer,
        "figures": [],
        "duration_sec": round(time.perf_counter() - started, 3),
        "failure_reason": None if verdict != "inconclusive" else reason,
        "learned": reason,
        "metric_value": output.get("metric_value"),
    }


async def _materials_verification_path(
    *,
    payload: dict,
    hypothesis: dict,
    hypothesis_id: str,
    campaign_node_id: str | None,
    session_id: str,
    question: str,
    session_factory: async_sessionmaker,
    emitter: EventEmitter,
    registry: ToolRegistry,
    trace_pointer: str,
    started: float,
    baseline: dict,
) -> dict:
    """Run crystal-system LOFO via materials_verification on matbench dielectric."""
    from propab.domain_adapters.materials_adapter import (
        MaterialsExperimentSpec,
        classify_materials_verdict,
    )
    from propab.artifact_verification import (
        evidence_context_from_hypothesis,
        merge_artifact_into_evidence,
        run_artifact_gate,
    )

    spec = MaterialsExperimentSpec.from_hypothesis(hypothesis, question=question)
    art_dir = f"/data/propab/sessions/{session_id}/materials/{hypothesis_id}"
    tool_result = registry.call(
        "materials_verification",
        {**spec.to_tool_params(), "artifacts_dir": art_dir},
    )

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.TOOL_CALLED,
        step=f"experiment.{hypothesis_id}.materials",
        payload={"tool": "materials_verification", "features": spec.feature_subset},
        hypothesis_id=hypothesis_id,
    )

    gate = None
    if not tool_result.success or not isinstance(tool_result.output, dict):
        err = tool_result.error.message if tool_result.error else "materials_verification failed"
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.TOOL_ERROR,
            step=f"experiment.{hypothesis_id}.materials",
            payload={"tool": "materials_verification", "error": err},
            hypothesis_id=hypothesis_id,
        )
        verdict, reason, confidence = "inconclusive", f"LOFO experiment failed: {err}", 0.0
        output: dict = {}
    else:
        output = tool_result.output
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.TOOL_RESULT,
            step=f"experiment.{hypothesis_id}.materials",
            payload={"tool": "materials_verification", "output": output},
            hypothesis_id=hypothesis_id,
        )
        verdict, reason, confidence = classify_materials_verdict(str(hypothesis.get("text") or ""), output)
        ctx = evidence_context_from_hypothesis(
            str(hypothesis.get("text") or ""),
            {
                "metric_value": output.get("mean_r2"),
                "lofo_r2": output.get("lofo_r2"),
                "lofo_gap": output.get("lofo_gap"),
                "p_value": output.get("permutation_p"),
                "n_samples": output.get("n_samples"),
                "n_families": output.get("n_families"),
                "group_column": output.get("group_column"),
                "methodology": "LOFO",
                "feature_subset": output.get("feature_subset"),
                "label_shuffle_permutation_p": output.get("label_shuffle_permutation_p"),
                "label_shuffle_null_p95": output.get("label_shuffle_null_p95"),
            },
            methodology="LOFO",
            domain_bucket="materials",
        )
        gate = run_artifact_gate(
            ctx, output, question=question, payload=payload if isinstance(payload, dict) else None,
        )
        if verdict == "confirmed" and gate.verdict != "confirmed":
            verdict, reason, confidence = gate.verdict, gate.verdict_reason, gate.confidence
        elif verdict == "confirmed":
            reason = gate.verdict_reason

    evidence_obj = _build_materials_evidence(
        output=output, verdict=verdict, reason=reason, baseline=baseline,
    )
    evidence_obj = attach_scope_integrity(
        evidence_obj,
        hypothesis_text=str(hypothesis.get("text") or ""),
        test_methodology=str(hypothesis.get("test_methodology") or spec.methodology),
        experiment_output=output,
        question=question,
        code=str(output.get("executed_code") or ""),
    )
    if gate is not None:
        evidence_obj = merge_artifact_into_evidence(evidence_obj, gate)
    from propab.scoped_claim import apply_ood_gate_to_verdict
    verdict, reason = apply_ood_gate_to_verdict(
        verdict, reason, evidence_obj,
        hypothesis_text=str(hypothesis.get("text") or ""),
        test_methodology=str(hypothesis.get("test_methodology") or spec.methodology),
    )
    if evidence_obj.get("scope_gate_result") == "FAIL" and verdict == "confirmed":
        verdict = "inconclusive"
        reason = f"scope integrity fail: {evidence_obj.get('scope_integrity', {}).get('reason', '?')}"
    evidence_obj["verdict_reason"] = reason
    evidence = (
        f"evidence={json.dumps(evidence_obj, ensure_ascii=False)}; "
        f"materials_verification; LOFO={output.get('lofo_r2')}; "
        f"gap={output.get('lofo_gap')}; label_p95={output.get('label_shuffle_null_p95')}; "
        f"features={output.get('feature_subset', spec.feature_subset)}"
    )
    key_finding = str(hypothesis.get("text") or "")[:300] if verdict == "confirmed" else None

    duration_ms = int((time.perf_counter() - started) * 1000)
    await _insert_experiment_step_tool(
        session_factory,
        step_id=str(uuid4()),
        hypothesis_id=hypothesis_id,
        step_index=0,
        tool_name="materials_verification",
        params={**spec.to_tool_params(), "artifacts_dir": art_dir},
        result_output=output if output else None,
        result_error=(
            {"message": tool_result.error.message}
            if not tool_result.success and tool_result.error
            else None
        ),
        duration_ms=duration_ms,
    )

    await _update_hypothesis(
        session_factory,
        hypothesis_id,
        status="completed",
        verdict=verdict,
        confidence=confidence,
        evidence_summary=evidence,
        key_finding=key_finding,
        tool_trace_id=trace_pointer,
    )
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.AGENT_COMPLETED,
        step=f"experiment.{hypothesis_id}.complete",
        payload={
            "verdict": verdict,
            "confidence": confidence,
            "sig_gate_passed": verdict in {"confirmed", "refuted"},
            "materials": True,
            "lofo_r2": output.get("lofo_r2"),
        },
        hypothesis_id=hypothesis_id,
    )

    metric_key = str(baseline.get("metric_name") or "lofo_r2").strip()
    result_dict: dict[str, Any] = {
        "hypothesis_id": hypothesis_id,
        "campaign_node_id": campaign_node_id,
        "verdict": verdict,
        "confidence": confidence,
        "evidence_summary": evidence,
        "key_finding": key_finding,
        "tool_trace_id": trace_pointer,
        "figures": [],
        "duration_sec": round(time.perf_counter() - started, 3),
        "failure_reason": None if verdict != "inconclusive" else reason,
        "learned": reason,
        "metric_value": output.get("lofo_r2"),
        "materials_artifacts_dir": art_dir,
    }
    if output.get("lofo_r2") is not None:
        result_dict[metric_key] = output.get("lofo_r2")
    return result_dict


async def run_sub_agent_async(payload: dict) -> dict:
    """
    Execute sub-agent trace with think-act or heuristic execution.

    - think-act (SUB_AGENT_PLAN_SOURCE=llm or hybrid): LLM decides each tool call
      after observing accumulated results. Significance gate enforced before verdict.
    - heuristic (SUB_AGENT_PLAN_SOURCE=heuristic): static multi-round tool plan,
      with a significance recovery step appended when no stat tool ran.

    Tool failures are non-fatal. Verdict requires significance evidence.
    """
    session_id: str = payload["session_id"]
    hypothesis_id: str = payload["hypothesis_id"]
    campaign_node_id: str | None = payload.get("campaign_node_id")
    hypothesis: dict = payload["hypothesis"]
    question: str = str(payload.get("question") or "")
    peer_findings: list[dict] = payload.get("peer_findings") or []
    learned_from: str | None = payload.get("learned_from") or None
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    baseline_lit_compare_safe = bool(baseline.get("lit_compare_safe", False))
    baseline_value = (
        float(baseline.get("metric_value"))
        if isinstance(baseline.get("metric_value"), (int, float))
        else None
    )

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    redis = await create_redis(settings.redis_url)
    emitter = EventEmitter(source="worker", redis=redis, session_factory=session_factory)
    llm = LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.llm_api_secret,
        emitter=emitter,
        session_factory=session_factory,
    )
    registry = ToolRegistry()
    trace_pointer = str(uuid4())
    started = time.perf_counter()
    last_tool_name: str | None = None
    last_tool_step_index: int | None = None

    try:
        await _update_hypothesis(session_factory, hypothesis_id, status="running")

        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_STARTED,
            step=f"experiment.{hypothesis_id}",
            payload={"hypothesis_id": hypothesis_id, "text": hypothesis.get("text")},
            hypothesis_id=hypothesis_id,
        )

        hypothesis = await _hydrate_hypothesis_from_campaign(
            hypothesis,
            session_id=session_id,
            campaign_node_id=campaign_node_id,
            session_factory=session_factory,
        )

        # Domain routing goes through the plugin registry — the worker never
        # inspects the question for domain keywords itself. Plugins own their own
        # detection (DomainPlugin.matches). This local table binds a resolved
        # plugin to the worker-side verification coroutine, which needs worker-only
        # objects (emitter, redis, registry) and therefore cannot live in core.
        from propab.domain_modules.registry import resolve_domain_plugin

        _domain_plugin = resolve_domain_plugin(question=question, payload=payload)
        _worker_verification_paths = {
            "materials": _materials_verification_path,
            "mandrake": _mandrake_verification_path,
        }
        _worker_path = (
            _worker_verification_paths.get(_domain_plugin.domain_id)
            if _domain_plugin is not None
            else None
        )
        if _worker_path is None and _domain_plugin is not None:
            _worker_path = _plugin_verification_path
        if _worker_path is not None:
            if _worker_path is _plugin_verification_path:
                result = await _worker_path(
                    payload=payload,
                    hypothesis=hypothesis,
                    hypothesis_id=hypothesis_id,
                    campaign_node_id=campaign_node_id,
                    session_id=session_id,
                    question=question,
                    session_factory=session_factory,
                    emitter=emitter,
                    registry=registry,
                    trace_pointer=trace_pointer,
                    started=started,
                    baseline=baseline,
                    domain_plugin=_domain_plugin,
                )
            else:
                result = await _worker_path(
                    payload=payload,
                    hypothesis=hypothesis,
                    hypothesis_id=hypothesis_id,
                    campaign_node_id=campaign_node_id,
                    session_id=session_id,
                    question=question,
                    session_factory=session_factory,
                    emitter=emitter,
                    registry=registry,
                    trace_pointer=trace_pointer,
                    started=started,
                    baseline=baseline,
                )
            await redis.close()
            await engine.dispose()
            return result

        logger.info(
            "[sub_agent] startup session_id=%s hypothesis_id=%s PROPAB_PROFILE=%s "
            "sandbox_code_max_retries=%s sandbox_after_timeout_llm_rewrite=%s",
            session_id,
            hypothesis_id,
            (os.environ.get("PROPAB_PROFILE") or "").strip(),
            int(getattr(settings, "sandbox_code_max_retries", 1)),
            bool(getattr(settings, "sandbox_after_timeout_llm_rewrite", True)),
        )

        fast_path = str(payload.get("fast_path") or "").strip().lower()
        if fast_path == "baseline_measurement":
            bm_cfg = payload.get("baseline_measurement") if isinstance(payload.get("baseline_measurement"), dict) else {}
            ds = str(bm_cfg.get("dataset") or "mnist").strip() or "mnist"
            n_bm = max(
                20,
                min(
                    int(bm_cfg.get("n_steps") or settings.campaign_baseline_max_train_steps),
                    int(getattr(settings, "campaign_baseline_max_train_steps", 150)),
                ),
            )
            metric_key = str(baseline.get("metric_name") or "val_accuracy").strip() or "val_accuracy"
            params = {
                "model_id": "auto",
                "dataset": ds,
                "n_steps": n_bm,
                "task": "classification",
            }
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.TOOL_CALLED,
                step=f"experiment.{hypothesis_id}.baseline_fast",
                payload={"tool": "train_model", "params": params, "fast_path": True},
                hypothesis_id=hypothesis_id,
            )
            result = registry.call("train_model", params)
            if result.success and isinstance(result.output, dict):
                metric_val = _primary_metric_from_tool_output(result.output)
                evidence = (
                    "fast baseline measurement via train_model; "
                    f"dataset={ds}; n_steps={n_bm}; metric={metric_key}; "
                    f"metric_value={metric_val}."
                )
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.TOOL_RESULT,
                    step=f"experiment.{hypothesis_id}.baseline_fast",
                    payload={"tool": "train_model", "output": result.output, "fast_path": True},
                    hypothesis_id=hypothesis_id,
                )
                await _update_hypothesis(
                    session_factory,
                    hypothesis_id,
                    status="completed",
                    verdict="inconclusive",
                    confidence=0.0,
                    evidence_summary=evidence,
                    key_finding=None,
                    tool_trace_id=trace_pointer,
                )
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_COMPLETED,
                    step=f"experiment.{hypothesis_id}.complete",
                    payload={"verdict": "inconclusive", "confidence": 0.0, "fast_path": fast_path},
                    hypothesis_id=hypothesis_id,
                )
                await redis.close()
                await engine.dispose()
                out = {
                    "hypothesis_id": hypothesis_id,
                    "campaign_node_id": campaign_node_id,
                    "verdict": "inconclusive",
                    "confidence": 0.0,
                    "evidence_summary": evidence,
                    "key_finding": None,
                    "tool_trace_id": trace_pointer,
                    "figures": [],
                    "duration_sec": round(time.perf_counter() - started, 3),
                    "failure_reason": None,
                    "learned": "Fast baseline measured with train_model.",
                    "metric_value": metric_val,
                    "baseline_value": baseline_value,
                }
                if metric_val is not None:
                    out[metric_key] = metric_val
                return out

            err = result.error.to_dict() if result.error else {"type": "tool_error", "message": "unknown"}
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.TOOL_ERROR,
                step=f"experiment.{hypothesis_id}.baseline_fast",
                payload={"tool": "train_model", "failure_kind": "fast_baseline_failed", "error": err},
                hypothesis_id=hypothesis_id,
            )
            raise RuntimeError(f"fast baseline measurement failed: {err}")

        hyp_text = str(hypothesis.get("text", ""))
        plan_source = (settings.sub_agent_plan_source or "heuristic").strip().lower()
        payload_domain = str(payload.get("domain") or "").strip()
        if plan_source == "heuristic" and payload_domain:
            domain = coerce_routed_domain(payload_domain)
            domain_reason = "Using orchestrator-provided domain in heuristic smoke mode."
        else:
            domain, domain_reason = await route_domain(
                hypothesis_text=hyp_text,
                llm=llm,
                session_id=session_id,
                hypothesis_id=hypothesis_id,
                question=question,
                payload=payload,
            )
        sandbox_timeout_sec = effective_sandbox_timeout_sec(
            domain,
            settings.sandbox_timeout_sec,
            use_domain_floor=bool(getattr(settings, "sandbox_use_domain_timeout_floor", True)),
        )

        await emitter.emit(
            session_id=session_id,
            event_type=EventType.TOOL_SELECTED,
            step=f"experiment.{hypothesis_id}.domain",
            payload={"domain": domain, "reason": domain_reason},
            hypothesis_id=hypothesis_id,
        )

        # Materials/mandrake campaigns take their dedicated verification path above
        # and return before reaching here; every remaining campaign uses the routed
        # domain's tool cluster (plus significance-capable tools).
        specs = registry.get_cluster_with_significance(domain)
        if not specs:
            specs = registry.get_cluster_with_significance("general_computation")
        available_tools = [str(s["name"]) for s in specs]
        resource_limits = {
            "memory_mb": settings.sandbox_memory_mb,
            "timeout_sec": sandbox_timeout_sec,
            "domain": domain,
        }

        can_llm = settings.llm_provider.strip().lower() == "ollama" or bool(settings.llm_api_secret.strip())
        use_think_act = plan_source in ("llm", "hybrid") and can_llm

        _alimits = payload.get("agent_limits") if isinstance(payload.get("agent_limits"), dict) else {}

        def _agent_lim(name: str, fallback: int) -> int:
            raw = _alimits.get(name)
            if raw is None:
                return int(fallback)
            try:
                return int(raw)
            except (TypeError, ValueError):
                return int(fallback)

        agent_max_seconds_eff = max(30, _agent_lim("max_seconds", int(settings.agent_max_seconds)))
        agent_max_steps = max(_agent_lim("max_steps", int(settings.agent_max_steps)), 5)
        agent_min_steps = max(1, min(int(settings.agent_min_steps), agent_max_steps - 1))
        max_tc = max(0, _agent_lim("max_tool_calls", int(getattr(settings, "agent_max_tool_calls", 0) or 0)))

        logger.info(
            "[sub_agent] hypothesis_id=%s sandbox_code_max_retries=%s sandbox_after_timeout_llm_rewrite=%s "
            "agent_max_steps=%s agent_max_seconds=%s agent_max_tool_calls=%s",
            hypothesis_id,
            int(getattr(settings, "sandbox_code_max_retries", 1)),
            bool(getattr(settings, "sandbox_after_timeout_llm_rewrite", True)),
            agent_max_steps,
            agent_max_seconds_eff,
            max_tc,
        )

        # Build initial tool steps (heuristic plan used as starting point for both paths)
        heuristic_steps = _heuristic_tool_plan_merged(
            specs, hypothesis_text=hyp_text, hypothesis=hypothesis,
        )
        plan_origin = "think_act" if use_think_act else "heuristic"

        if not use_think_act and plan_source in ("llm", "hybrid") and can_llm:
            max_llm = max(1, min(int(settings.sub_agent_max_planned_steps), 12))
            planned = await build_tool_plan_via_llm(
                llm=llm,
                session_id=session_id,
                hypothesis_id=hypothesis_id,
                hypothesis_text=hyp_text,
                specs=specs,
                max_steps=max_llm,
                emitter=emitter,
            )
            if planned is not None and len(planned) >= 1:
                heuristic_steps = _extend_plan_with_heuristic_rounds(
                    list(planned), specs, hypothesis_text=hyp_text, hypothesis=hypothesis,
                )
                plan_origin = "llm"

        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_PLAN_CREATED,
            step=f"experiment.{hypothesis_id}.plan",
            payload={
                "domain": domain,
                "sandbox_timeout_sec": sandbox_timeout_sec,
                "available_tools": available_tools,
                "resource_limits": resource_limits,
                "plan_origin": plan_origin,
                "think_act_enabled": use_think_act,
                "agent_max_steps": agent_max_steps,
                "agent_max_seconds": agent_max_seconds_eff,
                "agent_max_tool_calls": max_tc,
                "heuristic_steps": [tn for tn, _ in heuristic_steps],
            },
            hypothesis_id=hypothesis_id,
        )

        sandbox_ok = False
        any_tool_success = False
        successful_tool_outputs: list[dict[str, Any]] = []
        successful_tool_names: list[str] = []
        last_code_output: list[dict[str, Any]] = []
        step_counter = 0
        tool_calls_done = 0
        err_box: list[Any] = [None]
        # Provenance verdicts for each significance-tool call, in call order. Each
        # entry is "computed" | "agent_literal" | "unknown" (see
        # _classify_stat_input_provenance). Recorded from the arrays available
        # BEFORE the significance tool's own output is appended.
        sig_input_provenance: list[str] = []
        # Two-group outcome arrays captured from successful significance-tool calls
        # (results_a/results_b or treatment/baseline), in call order. Used AFTER
        # the loop to compute a genuine label-permutation null from the SAME data
        # the observed effect was measured on (D2). Only pairs where BOTH real
        # arrays were present are recorded — see extract_two_group_arrays.
        sig_call_arrays: list[tuple[list[float], list[float]]] = []

        # ── Shared tool execution helper (inline, no external function needed) ─

        async def run_tool_step(tool_name: str, params: dict, step_index: int) -> bool:
            nonlocal any_tool_success, last_tool_name, last_tool_step_index, tool_calls_done
            err_box[0] = None
            last_tool_name = tool_name
            last_tool_step_index = step_index
            step_id = str(uuid4())
            t0 = time.perf_counter()

            if max_tc > 0 and tool_calls_done >= max_tc:
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.TOOL_ERROR,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={
                        "tool": tool_name,
                        "failure_kind": "agent_max_tool_calls",
                        "error": {
                            "type": "budget_exceeded",
                            "detail": (
                                f"agent_max_tool_calls={max_tc} exhausted; refusing further tools."
                            ),
                        },
                    },
                    hypothesis_id=hypothesis_id,
                )
                return False
            tool_calls_done += 1

            call_params = dict(params or {})
            n_steps_cap = max(0, int(getattr(settings, "agent_tool_n_steps_cap", 0) or 0))
            if n_steps_cap > 0:
                for k in ("n_steps", "epochs", "num_steps", "num_epochs", "steps", "n_epochs", "training_steps"):
                    if k in call_params:
                        try:
                            call_params[k] = min(int(call_params[k]), n_steps_cap)
                        except (TypeError, ValueError):
                            pass
            if (
                tool_name == "literature_baseline_compare"
                and call_params.get("baseline_value") is None
                and baseline_lit_compare_safe
                and baseline_value is not None
                and abs(float(baseline_value)) >= 1e-12
            ):
                call_params["baseline_value"] = float(baseline_value)
            if tool_name == "literature_baseline_compare" and call_params.get("baseline_value") is not None:
                try:
                    if abs(float(call_params["baseline_value"])) < 1e-12:
                        call_params.pop("baseline_value", None)
                except (TypeError, ValueError):
                    pass

            # Record significance-input provenance BEFORE the call, while
            # successful_tool_outputs holds only PRIOR real outputs (this tool's
            # own output has not been appended yet).
            if tool_name in _SIGNIFICANCE_TOOL_NAMES:
                prov = _classify_stat_input_provenance(call_params, successful_tool_outputs)
                sig_input_provenance.append(prov)
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.TOOL_CALLED,
                    step=f"experiment.{hypothesis_id}.step_{step_index}.provenance",
                    payload={"tool": tool_name, "stat_input_provenance": prov},
                    hypothesis_id=hypothesis_id,
                )

            await emitter.emit(
                session_id=session_id,
                event_type=EventType.TOOL_CALLED,
                step=f"experiment.{hypothesis_id}.step_{step_index}",
                payload={"tool": tool_name, "params": call_params},
                hypothesis_id=hypothesis_id,
            )
            try:
                result = registry.call(tool_name, call_params)
            except TypeError as exc:
                result = ToolResult(
                    success=False,
                    error=ToolError(type="validation_error", message=str(exc)),
                )
            duration_ms = int((time.perf_counter() - t0) * 1000)

            if result.success:
                any_tool_success = True
                successful_tool_names.append(tool_name)
                if isinstance(result.output, dict):
                    successful_tool_outputs.append(result.output)
                # Capture the two real outcome arrays this significance test ran on,
                # so a genuine label-permutation null can be computed post-loop from
                # the SAME data as the observed effect (D2). The significance tool
                # output does NOT echo its inputs back, so we must read them from the
                # call params here. Fail-closed: only recorded when BOTH arrays exist.
                if tool_name in _SIGNIFICANCE_TOOL_NAMES:
                    _two_arrays = extract_two_group_arrays(call_params)
                    if _two_arrays is not None:
                        sig_call_arrays.append(_two_arrays)
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.TOOL_RESULT,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"tool": tool_name, "output": result.output},
                    hypothesis_id=hypothesis_id,
                )
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_COMPLETED,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"tool": tool_name},
                    hypothesis_id=hypothesis_id,
                )
            else:
                err_d = result.error.to_dict() if result.error else {}
                err_box[0] = err_d
                fk = "tool_execution_error"
                et = str((err_d or {}).get("type") or "")
                if et in ("validation_error", "missing_dependency", "auto_build_error", "zero_variance"):
                    fk = et
                elif "timeout" in str((err_d or {}).get("message", "")).lower():
                    fk = "tool_timeout_or_resource"
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.TOOL_ERROR,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"tool": tool_name, "failure_kind": fk, "error": err_d},
                    hypothesis_id=hypothesis_id,
                )
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_FAILED,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"tool": tool_name, "non_fatal": True},
                    hypothesis_id=hypothesis_id,
                )

            await _insert_experiment_step_tool(
                session_factory,
                step_id=step_id,
                hypothesis_id=hypothesis_id,
                step_index=step_index,
                tool_name=tool_name,
                params=call_params,
                result_output=result.output,
                result_error=result.error.to_dict() if result.error else None,
                duration_ms=duration_ms,
            )
            return bool(result.success)

        async def run_code_step(
            code: str,
            step_index: int,
            *,
            force_inline_trusted: bool = False,
        ) -> bool:
            nonlocal sandbox_ok
            del last_code_output[:]
            step_id = str(uuid4())
            t0 = time.perf_counter()
            code_cur = code
            parsed = None
            sandbox_out: dict = {}
            rewrite_used = False

            await emitter.emit(
                session_id=session_id,
                event_type=EventType.CODE_GENERATED,
                step=f"experiment.{hypothesis_id}.step_{step_index}",
                payload={"code": code_cur, "rewrite_after_timeout": False},
                hypothesis_id=hypothesis_id,
            )

            sandbox_ok = False
            trust_heuristic = _is_trusted_inline_sandbox_code(code_cur)
            use_inline = bool(force_inline_trusted) or trust_heuristic
            if use_inline:
                forced = bool(force_inline_trusted) and not trust_heuristic
                exec_label = "inline_stub_forced" if forced else "inline_stub"
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.CODE_SUBMITTED,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={
                        "memory_mb": settings.sandbox_memory_mb,
                        "timeout_sec": sandbox_timeout_sec,
                        "domain": domain,
                        "attempt": 1,
                        "max_attempts": 1,
                        "llm_rewrite_slot": False,
                        "execution": exec_label,
                        "note": (
                            "Programmatic stub — in-process (no Docker); forced because "
                            "description tripped substring heuristics or bypasses comment injection."
                            if forced
                            else "Trusted agent stub — in-process (no Docker) to avoid spurious sandbox timeouts."
                        ),
                    },
                    hypothesis_id=hypothesis_id,
                )
                sandbox_out = await asyncio.to_thread(_run_inline_trusted_sandbox_code, code_cur)
                parsed = sandbox_out.get("parsed") if isinstance(sandbox_out, dict) else None
                ok_run = bool(
                    sandbox_out.get("ok")
                    and isinstance(parsed, dict)
                    and parsed.get("sandbox") == "ok"
                )
                if ok_run:
                    sandbox_ok = True
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.CODE_RESULT,
                        step=f"experiment.{hypothesis_id}.step_{step_index}",
                        payload={
                            "stdout_json": parsed,
                            "stdout": sandbox_out.get("stdout"),
                            "attempt": 1,
                            "rewrite_after_timeout": False,
                            "execution": exec_label,
                        },
                        hypothesis_id=hypothesis_id,
                    )
                else:
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.CODE_ERROR,
                        step=f"experiment.{hypothesis_id}.step_{step_index}",
                        payload={
                            "failure_kind": "inline_stub_error",
                            "error": sandbox_out,
                            "attempt": 1,
                            "max_attempts": 1,
                        },
                        hypothesis_id=hypothesis_id,
                    )
            else:
                # Never burn N × sandbox wall on the same source: the old
                # ``while inv < hard_cap`` loop replayed identical code up to
                # ``sandbox_code_max_retries`` times after each timeout (45 submits /
                # 15 generations in campaigns). We allow at most:
                #   1) one Docker run of the model's code, then
                #   2) one optional second run *only* after an LLM rewrite (different source).
                # Docker path: total executions per ``code.generated`` are capped by
                # ``sandbox_code_max_retries`` (interpreted as max Docker runs, minimum 1).
                # When 1, skip the post-timeout rewrite second run entirely.
                allow_rewrite = bool(getattr(settings, "sandbox_after_timeout_llm_rewrite", True)) and can_llm
                max_docker = max(1, int(getattr(settings, "sandbox_code_max_retries", 1) or 1))
                allow_second = max_docker >= 2 and allow_rewrite
                planned_max = min(max_docker, 1 + (1 if allow_second else 0))
                for exec_n in range(1, planned_max + 1):
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.CODE_SUBMITTED,
                        step=f"experiment.{hypothesis_id}.step_{step_index}",
                        payload={
                            "memory_mb": settings.sandbox_memory_mb,
                            "timeout_sec": sandbox_timeout_sec,
                            "domain": domain,
                            "attempt": exec_n,
                            "max_attempts": planned_max,
                            "llm_rewrite_slot": allow_second,
                        },
                        hypothesis_id=hypothesis_id,
                    )
                    sandbox_out = await asyncio.to_thread(
                        run_sandboxed_python,
                        code_cur,
                        timeout_sec=sandbox_timeout_sec,
                        memory_mb=settings.sandbox_memory_mb,
                    )
                    parsed = sandbox_out.get("parsed") if isinstance(sandbox_out, dict) else None
                    ok_run = bool(
                        sandbox_out.get("ok")
                        and isinstance(parsed, dict)
                        and parsed.get("sandbox") == "ok"
                    )
                    if ok_run:
                        sandbox_ok = True
                        await emitter.emit(
                            session_id=session_id,
                            event_type=EventType.CODE_RESULT,
                            step=f"experiment.{hypothesis_id}.step_{step_index}",
                            payload={
                                "stdout_json": parsed,
                                "stdout": sandbox_out.get("stdout"),
                                "attempt": exec_n,
                                "rewrite_after_timeout": rewrite_used,
                            },
                            hypothesis_id=hypothesis_id,
                        )
                        break
                    is_timeout = _is_sandbox_wall_timeout(sandbox_out)
                    ev = EventType.CODE_TIMEOUT if is_timeout else EventType.CODE_ERROR
                    await emitter.emit(
                        session_id=session_id,
                        event_type=ev,
                        step=f"experiment.{hypothesis_id}.step_{step_index}",
                        payload={
                            "failure_kind": "sandbox_timeout" if is_timeout else "sandbox_error",
                            "timeout_sec": sandbox_timeout_sec,
                            "memory_mb": settings.sandbox_memory_mb,
                            "error": sandbox_out,
                            "attempt": exec_n,
                            "max_attempts": planned_max,
                            **_sandbox_diag_tails(sandbox_out),
                        },
                        hypothesis_id=hypothesis_id,
                    )
                    if (
                        exec_n < planned_max
                        and is_timeout
                        and allow_second
                        and looks_like_heavy_training_code(code_cur)
                    ):
                        new_code = await rewrite_sandbox_code_after_timeout(
                            llm,
                            session_id=session_id,
                            hypothesis_id=hypothesis_id,
                            code=code_cur,
                            sandbox_timeout_sec=int(sandbox_timeout_sec),
                            domain=str(domain or ""),
                        )
                        if new_code and new_code.strip() != code_cur.strip():
                            code_cur = new_code
                            rewrite_used = True
                            await emitter.emit(
                                session_id=session_id,
                                event_type=EventType.CODE_GENERATED,
                                step=f"experiment.{hypothesis_id}.step_{step_index}",
                                payload={
                                    "code": code_cur,
                                    "rewrite_after_timeout": True,
                                    "after_attempt": exec_n,
                                },
                                hypothesis_id=hypothesis_id,
                            )
                            continue
                    break

            duration_ms = int((time.perf_counter() - t0) * 1000)
            await _insert_experiment_step_code(
                session_factory,
                step_id=step_id,
                hypothesis_id=hypothesis_id,
                step_index=step_index,
                code=code_cur,
                parsed_output=parsed,
                sandbox_out=sandbox_out,
                duration_ms=duration_ms,
                memory_mb=settings.sandbox_memory_mb,
                timeout_sec=sandbox_timeout_sec,
            )
            if sandbox_ok and isinstance(parsed, dict):
                last_code_output.append(parsed)
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.AGENT_STEP_COMPLETED,
                step=f"experiment.{hypothesis_id}.step_{step_index}",
                payload={"sandbox_ok": sandbox_ok},
                hypothesis_id=hypothesis_id,
            )
            return sandbox_ok

        # ─────────────────────────────────────────────────────────────────────
        # THINK-ACT PATH: LLM decides each next action from accumulated context
        # ─────────────────────────────────────────────────────────────────────
        if use_think_act:
            deadline = time.monotonic() + max(30.0, float(agent_max_seconds_eff))
            agent_ctx = AgentContext(
                hypothesis_text=hyp_text,
                test_methodology=str(hypothesis.get("test_methodology") or ""),
                learned_from=learned_from,
                peer_findings=peer_findings,
                results_so_far=[],
                tool_names_run=[],
                steps_taken=0,
                max_steps=agent_max_steps,
                min_steps=agent_min_steps,
                deadline_monotonic=deadline,
                tool_failures=[],
            )

            # Run the first heuristic step immediately to seed the agent with data
            if heuristic_steps:
                first_tool, first_params = heuristic_steps[0]
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_STARTED,
                    step=f"experiment.{hypothesis_id}.step_{step_counter}",
                    payload={"tool": first_tool, "source": "heuristic_seed"},
                    hypothesis_id=hypothesis_id,
                )
                ok = await run_tool_step(first_tool, dict(first_params), step_counter)
                if ok and successful_tool_outputs:
                    agent_ctx.results_so_far.append(successful_tool_outputs[-1])
                    agent_ctx.tool_names_run.append(first_tool)
                step_counter += 1
                agent_ctx.steps_taken += 1

            # Think-act loop
            while not should_stop(agent_ctx):
                if max_tc > 0 and tool_calls_done >= max_tc:
                    logger.info(
                        "Agent %s hit agent_max_tool_calls=%s before next think step.",
                        hypothesis_id,
                        max_tc,
                    )
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.AGENT_STEP_COMPLETED,
                        step=f"experiment.{hypothesis_id}.budget",
                        payload={"reason": "agent_max_tool_calls", "limit": max_tc},
                        hypothesis_id=hypothesis_id,
                    )
                    break
                # Poll peer channel non-blocking — inject new findings into context
                try:
                    new_peers = await poll_peer_findings(redis, hypothesis_id=hypothesis_id)
                    if new_peers:
                        agent_ctx.peer_findings.extend(new_peers)
                        logger.debug(
                            "Agent %s received %d peer finding(s).", hypothesis_id, len(new_peers)
                        )
                except Exception:
                    pass  # peer polling is always best-effort

                action = await decide_next_action(
                    context=agent_ctx,
                    specs=specs,
                    llm=llm,
                    session_id=session_id,
                    hypothesis_id=hypothesis_id,
                    sandbox_timeout_sec=int(sandbox_timeout_sec),
                    agent_wall_budget_sec=int(agent_max_seconds_eff),
                )

                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_STARTED,
                    step=f"experiment.{hypothesis_id}.step_{step_counter}",
                    payload={"action": action.action_type, "reasoning": action.reasoning,
                             "expected_outcome": action.expected_outcome},
                    hypothesis_id=hypothesis_id,
                )

                if action.action_type == "stop":
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.AGENT_STEP_COMPLETED,
                        step=f"experiment.{hypothesis_id}.step_{step_counter}",
                        payload={"action": "stop", "reasoning": action.reasoning},
                        hypothesis_id=hypothesis_id,
                    )
                    break

                step_ok = False
                if action.action_type == "tool" and action.tool_name:
                    tool_name_ta = action.tool_name
                    # Tolerate unknown tool names: skip and continue
                    try:
                        step_ok = await run_tool_step(tool_name_ta, action.params, step_counter)
                    except KeyError:
                        logger.warning("Think-act chose unknown tool %s; skipping.", tool_name_ta)
                        await emitter.emit(
                            session_id=session_id,
                            event_type=EventType.TOOL_ERROR,
                            step=f"experiment.{hypothesis_id}.step_{step_counter}",
                            payload={
                                "tool": tool_name_ta,
                                "failure_kind": "unknown_tool_name",
                                "error": {
                                    "type": "unknown_tool",
                                    "detail": "Tool name not in registry (bad plan / typo).",
                                },
                            },
                            hypothesis_id=hypothesis_id,
                        )
                    if step_ok and successful_tool_outputs:
                        agent_ctx.results_so_far.append(successful_tool_outputs[-1])
                        agent_ctx.tool_names_run.append(tool_name_ta)
                    elif not step_ok and err_box[0]:
                        agent_ctx.tool_failures.append({"tool": tool_name_ta, "error": err_box[0]})
                        if len(agent_ctx.tool_failures) > 12:
                            agent_ctx.tool_failures.pop(0)

                elif action.action_type == "code":
                    real_code = (action.code or "").strip()
                    if real_code:
                        # Real LLM-authored program → execute in the Docker sandbox.
                        step_ok = await run_code_step(real_code, step_counter)
                        if step_ok and last_code_output:
                            # Feed real computed output back so the LLM can chain on it and
                            # the significance/evidence layer can use real measurements.
                            out = last_code_output[-1]
                            agent_ctx.results_so_far.append(out)
                            successful_tool_outputs.append(out)
                            successful_tool_names.append("__code__")
                    else:
                        # No source supplied (description only) → deterministic no-op stub so the
                        # step is recorded without burning a Docker wall on an empty computation.
                        code_desc = action.code_description or "custom computation"
                        step_ok = await run_code_step(
                            _think_act_stub_code(str(code_desc)),
                            step_counter,
                            force_inline_trusted=True,
                        )
                    if step_ok:
                        agent_ctx.tool_names_run.append("__code__")

                step_counter += 1
                agent_ctx.steps_taken += 1

            if agent_ctx.time_budget_exceeded:
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_TIME_BUDGET_EXCEEDED,
                    step=f"experiment.{hypothesis_id}.time_budget",
                    payload={
                        "agent_max_seconds": int(agent_max_seconds_eff),
                        "steps_taken": agent_ctx.steps_taken,
                        "tools_tail": agent_ctx.tool_names_run[-12:],
                        "reason": "Wall clock cap (settings.agent_max_seconds / PROPAB_PROFILE).",
                    },
                    hypothesis_id=hypothesis_id,
                )

        # ─────────────────────────────────────────────────────────────────────
        # HEURISTIC PATH: static multi-round plan + significance recovery
        # ─────────────────────────────────────────────────────────────────────
        else:
            plan_steps: list[dict] = [
                {"type": "tool", "tool": tn, "params": dict(pr)} for tn, pr in heuristic_steps
            ] + [{
                "type": "code",
                "code": (
                    "import json,sys\n"
                    f"print(json.dumps({{\"sandbox\":\"ok\",\"hypothesis_rank\":{json.dumps(hypothesis.get('rank'))}}}))  \n"
                ),
            }]
            if not plan_steps:
                raise RuntimeError(
                    f"Empty execution plan for hypothesis {hypothesis_id}; refusing zero-step trace."
                )

            prev_out: dict | None = None
            for step_index, step in enumerate(plan_steps):
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_STARTED,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"step": step},
                    hypothesis_id=hypothesis_id,
                )

                if step.get("type") == "code":
                    await run_code_step(
                        str(step.get("code", "")),
                        step_index,
                        force_inline_trusted=True,
                    )
                else:
                    tool_name = step["tool"]
                    params = step["params"]
                    # Chain prior output into next step if applicable
                    if prev_out is not None and step_index > 0:
                        step = refine_next_tool_step(
                            plan_steps[step_index - 1].get("tool", ""),
                            prev_out,
                            step,
                        )
                        params = step["params"]
                    await run_tool_step(tool_name, params, step_index)
                    prev_out = successful_tool_outputs[-1] if successful_tool_outputs else None

                step_counter = step_index + 1

            # Significance recovery: if no significance tool ran, force one now
            if not any_significance_tool_ran(successful_tool_names) and successful_tool_outputs:
                sig_step_index = step_counter
                logger.info(
                    "Significance recovery: no stat tool ran for %s. Attempting bootstrap_confidence.",
                    hypothesis_id,
                )
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_STARTED,
                    step=f"experiment.{hypothesis_id}.step_{sig_step_index}",
                    payload={"note": "significance_recovery", "tool": "bootstrap_confidence"},
                    hypothesis_id=hypothesis_id,
                )
                # Extract numeric values from outputs
                values: list[float] = []
                for out in successful_tool_outputs:
                    for v in _walk_numeric_values(out).values():
                        values.append(v)
                        if len(values) >= 20:
                            break
                    if len(values) >= 20:
                        break

                if len(values) >= 2:
                    try:
                        await run_tool_step(
                            "bootstrap_confidence",
                            {"values": values[:20]},
                            sig_step_index,
                        )
                        step_counter += 1
                    except Exception as exc:
                        logger.warning("Significance recovery failed: %s", exc)
                else:
                    logger.info("Significance recovery skipped: not enough numeric values.")

        # ─────────────────────────────────────────────────────────────────────
        # VERDICT COMPUTATION (shared by both paths)
        # ─────────────────────────────────────────────────────────────────────
        relevance_score = _hypothesis_relevance_score(hyp_text, successful_tool_outputs)
        substantive_hit = any(n in SUBSTANTIVE_TOOL_NAMES for n in successful_tool_names)
        if substantive_hit:
            relevance_score = min(1.0, relevance_score + 0.06)

        n_tool_steps = sum(1 for n in successful_tool_names if n not in _SIGNIFICANCE_TOOL_NAMES)
        evidence_obj = _build_evidence(
            successful_outputs=successful_tool_outputs,
            relevance_score=relevance_score,
            n_tool_steps=n_tool_steps,
            baseline_value=baseline_value,
        )

        # ── Significance-input provenance (audit signal) ──────────────────────
        # Aggregate per-call verdicts into a single evidence-level signal. This
        # is only meaningful for the think-act / heuristic worker path (where
        # sig_input_provenance was populated by run_tool_step); the dedicated
        # domain-verification paths (mandrake/materials/plugin) return earlier
        # and never reach here.
        #
        # Aggregation rule (conservative): if ANY significance call rested on
        # agent-typed literals, the evidence is flagged agent_literal — one
        # fabricated comparison taints the verdict. "computed" requires that at
        # least one significance call ran AND every such call traced to prior
        # real outputs. "unknown" when we could not decide (no numeric-array
        # inputs / no significance call).
        if sig_input_provenance:
            if any(p == "agent_literal" for p in sig_input_provenance):
                agg_provenance = "agent_literal"
            elif any(p == "computed" for p in sig_input_provenance):
                agg_provenance = "computed"
            else:
                agg_provenance = "unknown"
        else:
            agg_provenance = "unknown"
        evidence_obj["stat_input_provenance"] = agg_provenance
        evidence_obj["stat_input_provenance_calls"] = list(sig_input_provenance)
        evidence_obj["inputs_from_sandbox"] = agg_provenance == "computed"

        # ── Real label-permutation null (D2) ──────────────────────────────────
        # If the agent ran a two-group significance comparison on real outcome
        # arrays, compute a genuine label-permutation null from that SAME data and
        # attach `permutation_p` + `n_samples` so artifact_verification.
        # _survives_permutation (read-only, core) can evaluate a real adversarial
        # null. Without this, a generic statistical result carries no null and the
        # verdict pipeline correctly keeps it inconclusive.
        #
        # Fail-closed integrity:
        #   * `sig_call_arrays` only holds pairs where BOTH real arrays were present
        #     (extract_two_group_arrays); a lone p-value or single array never lands
        #     here, so `permutation_p` stays absent and the result stays inconclusive.
        #   * The p-value is a deterministic function of the actual arrays under a
        #     fixed seed — nothing is self-reported. We never write `permutation_p`
        #     unless compute_label_permutation_null returned a real result.
        #   * We do NOT strip or alter `stat_input_provenance`: if the significance
        #     inputs were agent-typed (`agent_literal`), the null is still computed
        #     on those (untrusted) numbers but the provenance tag remains on the
        #     evidence for a later gate to reject. Attaching a real null never
        #     launders untrusted inputs.
        _had_perm_before = evidence_obj.get("permutation_p")
        attach_permutation_null_to_evidence(evidence_obj, sig_call_arrays)
        if evidence_obj.get("permutation_p") is not None and _had_perm_before is None:
            logger.info(
                "label-permutation null computed for %s: permutation_p=%.4f "
                "n_samples=%s (n_perm=%s, provenance=%s)",
                hypothesis_id,
                float(evidence_obj["permutation_p"]),
                evidence_obj.get("n_samples"),
                evidence_obj.get("permutation_null_n_permutations"),
                agg_provenance,
            )

        bm_cfg = payload.get("baseline_measurement")
        if isinstance(bm_cfg, dict) and evidence_obj.get("metric_value") is None:
            ds = str(bm_cfg.get("dataset") or "mnist").strip() or "mnist"
            n_bm = max(50, min(int(bm_cfg.get("n_steps") or 150), 500))
            try:
                r_bm = registry.call(
                    "train_model",
                    {
                        "model_id": "auto",
                        "dataset": ds,
                        "n_steps": n_bm,
                        "task": "classification",
                    },
                )
                if r_bm.success and isinstance(r_bm.output, dict):
                    cand = _primary_metric_from_tool_output(r_bm.output)
                    if cand is not None:
                        fv_c = float(cand)
                        evidence_obj["metric_value"] = fv_c
                        evidence_obj["n_metric_steps"] = max(1, int(evidence_obj.get("n_metric_steps") or 0))
                        if baseline_value is not None:
                            evidence_obj["delta"] = fv_c - float(baseline_value)
                        logger.info(
                            "baseline_measurement fallback train_model(dataset=%s, n_steps=%s) "
                            "-> val_accuracy=%.4f",
                            ds,
                            n_bm,
                            float(cand),
                        )
            except Exception as exc_bm:
                logger.warning("baseline_measurement train_model fallback failed: %s", exc_bm)

        # Use significance module for the gate check
        sig_result = check_significance(successful_tool_outputs)

        _vesc = payload.get("verification_escalation") if isinstance(payload.get("verification_escalation"), dict) else {}
        # Per-domain confirmation threshold: read the base from the resolved domain
        # plugin's confirmation_criteria() (ownership contract), falling back to the
        # global setting when no scientific-domain plugin owns this campaign.
        _base_metric_steps = int(getattr(settings, "min_metric_steps_for_confirm", 2))
        if _domain_plugin is not None:
            try:
                _crit = _domain_plugin.confirmation_criteria()
                _base_metric_steps = int(_crit.get("min_metric_steps_for_confirm") or _base_metric_steps)
            except Exception:
                pass
        min_metric_steps = int(_vesc.get("min_metric_steps") or _base_metric_steps)
        if _vesc.get("require_replication"):
            min_metric_steps = max(min_metric_steps, 3)
        if _vesc.get("extra_significance_rounds"):
            min_metric_steps = max(min_metric_steps, 2)

        from propab.verdict_pipeline import run_verdict_pipeline

        campaign_context = {
            "question": question,
            "payload": payload if isinstance(payload, dict) else None,
            "hyp_text": str(hyp_text or ""),
            "tools_used": successful_tool_names,
            "domain_bucket": str(payload.get("domain_bucket") or payload.get("domain") or ""),
            "min_metric_steps": min_metric_steps,
            "sig_result": sig_result,
            "test_methodology": str(hypothesis.get("test_methodology") or ""),
        }
        verdict, confidence, verdict_reason = run_verdict_pipeline(
            evidence_obj,
            hypothesis=hypothesis,
            campaign_context=campaign_context,
        )
        evidence_obj = attach_scope_integrity(
            evidence_obj,
            hypothesis_text=str(hyp_text or ""),
            test_methodology=str(hypothesis.get("test_methodology") or ""),
            experiment_output=None,
            question=question,
        )
        if evidence_obj.get("scope_gate_result") == "FAIL" and verdict == "confirmed":
            verdict = "inconclusive"
            verdict_reason = f"scope integrity fail: {evidence_obj.get('scope_integrity', {}).get('reason', '?')}"
        evidence_obj["verdict_reason"] = verdict_reason
        claim_type = classify_claim_type(evidence_obj, verdict, hypothesis_text=str(hyp_text or ""))
        if claim_type:
            evidence_obj["claim_type"] = claim_type
        if int(evidence_obj.get("verified_true_steps") or 0) > 0 and verdict == "confirmed":
            confidence = max(confidence, 0.95)
        elif int(evidence_obj.get("verified_false_steps") or 0) > 0 and verdict == "refuted":
            confidence = max(confidence, 0.95)
        elif confidence == 0.0 and evidence_obj.get("n_metric_steps", 0) > 0:
            confidence = _compute_confidence(evidence_obj)

        sig_summary = {
            "gate_passed": sig_result.gate_passed,
            "p_value": sig_result.p_value,
            "effect_size": sig_result.effect_size,
            "method": sig_result.method,
        }
        evidence = (
            f"evidence={json.dumps(evidence_obj, ensure_ascii=False)}; "
            f"significance={json.dumps(sig_summary)}; "
            f"plan_origin={plan_origin}; "
            f"any_success={any_tool_success}; sandbox_ok={sandbox_ok}; "
            f"steps={step_counter}."
        )
        # The key finding must be the actual claim that was supported — not a generic
        # "significance gate passed" line — or the paper reads like an internal log.
        if verdict == "confirmed":
            claim = re.split(r"\s*\(Question:", str(hyp_text or "").strip())[0].strip()
            claim = re.sub(r"^Hypothesis\s+\d+\s*:\s*", "", claim).strip()
            key_finding = claim or "The hypothesis was supported by statistically significant evidence."
        elif sandbox_ok:
            key_finding = "Sandbox probe completed."
        else:
            key_finding = None

        await _update_hypothesis(
            session_factory,
            hypothesis_id,
            status="completed",
            verdict=verdict,
            confidence=confidence,
            evidence_summary=evidence,
            key_finding=key_finding,
            tool_trace_id=trace_pointer,
        )

        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_COMPLETED,
            step=f"experiment.{hypothesis_id}.complete",
            payload={"verdict": verdict, "confidence": confidence, "sig_gate_passed": sig_result.gate_passed},
            hypothesis_id=hypothesis_id,
        )

        duration_sec = time.perf_counter() - started
        metric_out = evidence_obj.get("metric_value")
        metric_key = str(baseline.get("metric_name") or "").strip()
        result_dict: dict[str, Any] = {
            "hypothesis_id": hypothesis_id,
            "campaign_node_id": campaign_node_id,
            "verdict": verdict,
            "confidence": confidence,
            "evidence_summary": evidence,
            "key_finding": key_finding,
            "tool_trace_id": trace_pointer,
            "figures": [],
            "duration_sec": round(duration_sec, 3),
            "failure_reason": None if verdict == "confirmed" else evidence_obj["verdict_reason"],
            "learned": (
                f"Tools run: {', '.join(successful_tool_names[:10])}. "
                f"Significance: {sig_result.method or 'none'}. "
                f"Verdict: {verdict}."
            ),
            "metric_value": metric_out,
            "baseline_value": evidence_obj.get("baseline_value"),
        }
        if metric_key and metric_out is not None:
            result_dict[metric_key] = metric_out

        await redis.close()
        await engine.dispose()
        return result_dict

    except Exception as exc:
        detail = classify_exception(
            exc,
            hint_tool=last_tool_name,
            hint_step_index=last_tool_step_index,
        )
        await _update_hypothesis(
            session_factory,
            hypothesis_id,
            status="failed",
            verdict="inconclusive",
            confidence=0.0,
            evidence_summary=json.dumps(detail, ensure_ascii=False)[:8000],
            key_finding=None,
            tool_trace_id=trace_pointer,
        )
        fail_payload = {
            **detail,
            "error": str(exc)[:1500],
        }
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_FAILED,
            step=f"experiment.{hypothesis_id}.failed",
            payload=fail_payload,
            hypothesis_id=hypothesis_id,
        )
        await redis.close()
        await engine.dispose()
        return {
            "hypothesis_id": hypothesis_id,
            "campaign_node_id": campaign_node_id,
            "verdict": "inconclusive",
            "confidence": 0.0,
            "evidence_summary": str(exc),
            "key_finding": None,
            "tool_trace_id": trace_pointer,
            "figures": [],
            "duration_sec": round(time.perf_counter() - started, 3),
            "failure_reason": str(exc),
            "learned": None,
        }
