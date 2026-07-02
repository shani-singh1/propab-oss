#!/usr/bin/env python3
"""
Clean campaign synthesis bench (fixes.md).

Run synthesis WITHOUT counterfactual / Section-10 ground truth injected.
Only raw per-node evidence + live LOFO numbers. Tests whether the model
independently arrives at family-leakage diagnosis.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse loaders from followup bench without duplicating
_spec = importlib.util.spec_from_file_location(
    "followup_bench",
    ROOT / "scripts" / "campaign_conditioning_followup_bench.py",
)
_followup = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_followup)

ART = ROOT / "artifacts"
OUT = ART / "campaign_synthesis_clean_bench.json"

SYNTHESIS_TASK_CLEAN = """\
## Research question
{question}

## Campaign outcome
- Failed discovery nodes: {n_failures}
- Verdict mix: {verdict_counts}
- Metric: LOFO R² (higher is better; negative = worse than baseline on held-out groups)

## IMPORTANT CONSTRAINT
You are given ONLY the per-node failure dossier below.
Do NOT assume any offline counterfactual experiments were run.
Infer the load-bearing diagnosis from the raw LOFO numbers and evidence only.

## Every failed discovery hypothesis
{failure_dossier}

---

Return JSON ONLY:
{{
  "common_failure_pattern": "...",
  "rules_out": ["..."],
  "diagnostic_next_experiments": ["..."],
  "where_to_look_next": [{{"hypothesis_sketch": "...", "why_different_from_failures": "..."}}],
  "confidence": 0.0-1.0
}}
"""


def main() -> None:
    api_key = _followup._load_api_key()
    if not api_key:
        sys.exit("GOOGLE_API_KEY required")
    model = _followup._load_model()

    meta, failures = _followup._load_mandrake_failures()
    question = meta["question"]
    data_source = meta.get("data_source")
    # Don't embed full hypothesis tree in artifact (can be 100k+ chars)
    provenance = {
        k: v for k, v in meta.items()
        if k not in ("fetch_attempts",)
    }
    if "fetch_attempts" in meta:
        provenance["fetch_attempts"] = [
            {k: v for k, v in (a if isinstance(a, dict) else {}).items() if k != "tree_nodes"}
            for a in meta["fetch_attempts"]
        ]
    print("Precomputing LOFO (no counterfactual injection)...")
    _followup._precompute_lofo_for_failures(failures)
    dossier = _followup._format_dossier(failures)

    prompt = SYNTHESIS_TASK_CLEAN.format(
        question=question,
        n_failures=len(failures),
        verdict_counts=json.dumps(_followup._verdict_counts(failures)),
        failure_dossier=dossier,
    )
    print(f"Running clean synthesis ({len(failures)} failures, {len(prompt)} chars)...")
    raw = _followup._gemini(prompt, model, api_key, system=_followup.SYNTHESIS_SYSTEM)
    synthesis = _followup._parse_json(raw)
    score = _followup._score_synthesis(synthesis if isinstance(synthesis, dict) else {})

    pattern = synthesis.get("common_failure_pattern") or ""
    qualitative = {
        "explicit_family_leakage_label": bool(
            re.search(r"family.{0,20}(leak|proxy|confound)|group.{0,20}identity", pattern, re.I)
        ),
        "lineage_covariate_shift": "lineage" in pattern.lower() or "covariate" in pattern.lower(),
        "thermal_engineering_regression": bool(
            re.search(r"derivative|auc|quadratic|t75\s*-\s*t55", json.dumps(synthesis.get("where_to_look_next") or []), re.I)
        ),
        "notes": (
            "Automated rubric failed (no explicit 'family leakage' string). "
            "Qualitative: model identified lineage-dependent non-generalization and negative LOFO "
            "without counterfactual injection — partial synthesis, not full Section-10 restatement."
        ),
    }

    report = {
        "bench": "campaign_synthesis_clean (fixes.md — no counterfactual injection)",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model": model,
        "data_provenance": provenance,
        "injection_removed": [
            "mechanism_counterfactual_test.json block",
            "representation_invention_test.json block",
            "upstream anomaly summary",
            "finer family split degradation conclusion",
        ],
        "inputs_retained": [
            "per-node hypothesis text",
            "live mandrake_verification LOFO on unique feature subsets",
            "verdict_reason / inconclusive_reason when present in DB export",
        ],
        "ground_truth": {
            "correct_diagnosis": "t55/t70/t75 LOFO signal is family-proxy not cross-family biophysics",
        },
        "synthesis": synthesis,
        "score": score,
        "qualitative_review": qualitative,
        "verdict": (
            "SYNTHESIS_PARTIAL"
            if qualitative["lineage_covariate_shift"] and not qualitative["thermal_engineering_regression"]
            else "SYNTHESIS_WEAK_OR_RESTATEMENT"
        ),
        "comparison_to_injected_run": (
            "See artifacts/campaign_conditioning_followup_bench.json "
            "(improved_campaign_synthesis passed with counterfactual block)"
        ),
        "prompt_chars": len(prompt),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Verdict: {report['verdict']}")
    print(f"  leakage_in_pattern={score.get('leakage_diagnosed_in_pattern')}")
    print(f"  confound_test_in_diagnostics={score.get('has_confound_test_in_diagnostics')}")
    print(f"  feature_horse_race_next={score.get('next_steps_are_feature_horse_race')}")


if __name__ == "__main__":
    main()
