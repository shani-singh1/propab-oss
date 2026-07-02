#!/usr/bin/env python3
"""
Contagion refuted-batch A/B harness (fixes.md).

Compares orchestrator role variants on the same 8 labeled contagion failures.
Scores meta-diagnosis *shape* (scope/significance/simulator-class failures),
not exact vocabulary tokens from the prompt.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.belief_state import CampaignBeliefState
from propab.campaign_synthesis import parse_synthesis_response
from propab.prompt_composer import compose_synthesis_prompt, load_prompt

from synthesis_redesign_preflight import (  # noqa: E402
    CONTAGION_QUESTION,
    _failures_to_tree,
    _load_api_key,
    _load_contagion_failures,
    _load_model,
    _score_beliefs,
    _score_critical_experiment,
)

OUT = ROOT / "artifacts" / "synthesis_contagion_failure_mode_ab.json"
MERGED_PASSED = ROOT / "prompts" / "orchestrator_role.merged_passed.md"
CANDIDATE_ROLE = ROOT / "prompts" / "orchestrator_role.md"

# Semantic shapes — independent of exact prompt tokens
META_SHAPES: dict[str, re.Pattern[str]] = {
    "scope_inflation": re.compile(
        r"scope.?inflat|overly scoped|single.?context|single topology|one (?:network|topology|model)|"
        r"do(?:es)? not generalize|fail(?:s|ed)? to generalize|narrow(?:ed)? (?:test|scope)|"
        r"without (?:ood|transfer|held.?out|cross.?topology)|limited to (?:a |one )",
        re.I,
    ),
    "significance_only": re.compile(
        r"significance.?only|p.?value|statistical significance|low p|"
        r"without (?:replication|replicat|robust)|replicat(?:ion|e)|effect size|"
        r"sole criterion|insufficient (?:validation|replication)",
        re.I,
    ),
    "simulator_artifact": re.compile(
        r"simulat(?:or|ion).{0,20}artifact|simulator.?artifact|procedural|algorithmic|"
        r"setup.?specific|method rather than|simulation (?:method|environment|setup)|"
        r"artifact of the (?:simulation|experiment|setup)",
        re.I,
    ),
    "topology_dependence": re.compile(
        r"topology.?depend|cross.?topology|graph family|network type|"
        r"does not (?:hold|transfer) across|specific (?:topolog|network model)",
        re.I,
    ),
    "failure_class_diagnosis": re.compile(
        r"why (?:attempts|hypotheses|nodes|these).{0,30}fail|failed as a class|"
        r"failure pattern|share[sd]? a (?:common )?(?:failure|testing)|"
        r"explain why|attempts are failing|batch.{0,20}fail",
        re.I,
    ),
}
STRUCTURAL_CLAIM_BIAS = re.compile(
    r"stronger determinant|more robust predictor|dominates|predicts.{0,20}better|"
    r"is a stronger|wins over|outperforms.{0,15}(metric|predictor|coefficient)|"
    r"better than (?:mean degree|global|average)",
    re.I,
)


async def _gemini(prompt: str, model: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(url, params={"key": api_key}, json=payload)
    r.raise_for_status()
    parts = (((r.json().get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


def _score_contagion_meta(beliefs: list[dict[str, Any]]) -> dict[str, Any]:
    stmts = [str(b.get("statement") or "") for b in beliefs if b.get("statement")]
    blob = " ".join(stmts)
    shape_hits: dict[str, list[str]] = {}
    per_belief_shapes: list[dict[str, Any]] = []
    for i, s in enumerate(stmts):
        matched = [name for name, pat in META_SHAPES.items() if pat.search(s)]
        per_belief_shapes.append({"idx": i, "shapes": matched, "statement": s[:240]})
        for name in matched:
            shape_hits.setdefault(name, []).append(i)

    n_meta_beliefs = sum(1 for pb in per_belief_shapes if pb["shapes"])
    structural = bool(STRUCTURAL_CLAIM_BIAS.search(blob))
    core_shapes = {"scope_inflation", "significance_only", "simulator_artifact"}
    core_hit = core_shapes & set(shape_hits)
    meta_diagnosis_present = (
        len(core_hit) >= 2
        or (len(core_hit) >= 1 and n_meta_beliefs >= 2)
        or ("failure_class_diagnosis" in shape_hits and n_meta_beliefs >= 2 and not structural)
    )
    return {
        "statements": stmts,
        "n_beliefs": len(stmts),
        "shape_hits": {k: v for k, v in shape_hits.items()},
        "core_shapes_hit": sorted(core_hit),
        "per_belief_shapes": per_belief_shapes,
        "n_meta_beliefs": n_meta_beliefs,
        "structural_claim_hits": [m.group(0) for m in STRUCTURAL_CLAIM_BIAS.finditer(blob)],
        "fresh_structural_claim_bias": structural and n_meta_beliefs == 0,
        "meta_diagnosis_present": meta_diagnosis_present,
    }


async def _run_arm(
    *,
    label: str,
    role_text: str,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    failures = _load_contagion_failures(8)
    tree = _failures_to_tree(failures)
    bs = CampaignBeliefState()
    prompt = compose_synthesis_prompt(
        question=CONTAGION_QUESTION,
        belief_state=bs,
        tree=tree,
        role_text=role_text,
    )
    raw = await _gemini(prompt, model, api_key)
    parsed = parse_synthesis_response(raw)
    beliefs = parsed.get("beliefs") or []
    crit = parsed.get("critical_experiment")
    meta = _score_contagion_meta(beliefs if isinstance(beliefs, list) else [])
    return {
        "label": label,
        "role_chars": len(role_text),
        "prompt_chars": len(prompt),
        "parsed_ok": not parsed.get("_parse_error"),
        "beliefs_raw": beliefs,
        "meta_score": meta,
        "belief_score": _score_beliefs(beliefs if isinstance(beliefs, list) else [], domain="contagion"),
        "critical_experiment": crit,
        "critical_experiment_score": _score_critical_experiment(
            crit if isinstance(crit, dict) else None,
            beliefs if isinstance(beliefs, list) else [],
        ),
        "frontier_candidates": parsed.get("frontier_candidates"),
    }


def _passes(meta: dict[str, Any]) -> bool:
    return (
        meta.get("meta_diagnosis_present")
        and not meta.get("fresh_structural_claim_bias")
        and meta.get("n_beliefs", 0) <= 3
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Contagion failure-mode orchestrator A/B")
    parser.add_argument(
        "--reference-role",
        default=str(MERGED_PASSED),
        help="Previously passing merged prompt (default: orchestrator_role.merged_passed.md)",
    )
    parser.add_argument(
        "--candidate-role",
        default=str(CANDIDATE_ROLE),
        help="Candidate prompt to validate before replacing reference (default: orchestrator_role.md)",
    )
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    args = parser.parse_args()

    api_key = _load_api_key()
    if not api_key:
        print("GOOGLE_API_KEY / GEMINI_API_KEY required", file=sys.stderr)
        return 1

    model = _load_model()
    reference_text = Path(args.reference_role).read_text(encoding="utf-8")
    candidate_text = Path(args.candidate_role).read_text(encoding="utf-8")

    report: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model": model,
        "n_failures": 8,
        "reference_role_path": args.reference_role,
        "candidate_role_path": args.candidate_role,
        "scoring": "semantic meta-diagnosis shapes (not exact vocab tokens)",
    }

    if not args.candidate_only:
        report["reference_merged_passed"] = await _run_arm(
            label="merged_passed",
            role_text=reference_text,
            model=model,
            api_key=api_key,
        )
    if not args.reference_only:
        report["candidate_revision"] = await _run_arm(
            label="candidate_revision",
            role_text=candidate_text,
            model=model,
            api_key=api_key,
        )

    ref = report.get("reference_merged_passed")
    cand = report.get("candidate_revision")
    ref_pass = _passes(ref["meta_score"]) if ref else None
    cand_pass = _passes(cand["meta_score"]) if cand else None

    if cand_pass and ref_pass:
        verdict = "GENERALIZATION_HELD — candidate matches reference meta-diagnosis quality"
        recommendation = "safe_to_replace_merged_prompt"
    elif cand_pass and ref_pass is False:
        verdict = "CANDIDATE_IMPROVES — candidate passes where reference failed (lucky run? recheck)"
        recommendation = "safe_to_replace_merged_prompt"
    elif cand_pass:
        verdict = "CANDIDATE_OK — meta-diagnosis present (no reference arm run)"
        recommendation = "safe_to_replace_merged_prompt"
    elif not cand_pass and ref_pass:
        verdict = "REGRESSION — candidate lost meta-diagnosis; keep merged_passed prompt"
        recommendation = "keep_merged_passed"
    else:
        verdict = "FAIL — neither arm produces meta-diagnosis on this run"
        recommendation = "do_not_replace"

    report["verdict"] = verdict
    report["recommendation"] = recommendation
    report["reference_pass"] = ref_pass
    report["candidate_pass"] = cand_pass
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "verdict": verdict,
        "recommendation": recommendation,
        "reference_pass": ref_pass,
        "candidate_pass": cand_pass,
        "candidate_core_shapes": (cand or {}).get("meta_score", {}).get("core_shapes_hit"),
        "candidate_beliefs": (cand or {}).get("meta_score", {}).get("statements"),
        "written": str(OUT),
    }
    print(json.dumps(summary, indent=2))
    return 0 if recommendation == "safe_to_replace_merged_prompt" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
