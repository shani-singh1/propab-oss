#!/usr/bin/env python3
"""
Dead-finding artifact diagnosis bench (fixes.md).

Give Propab's generate_artifact_models only claim + evidence (no labels).
Compare top artifact diagnosis vs hand-classified ground truth.
Optional LLM diagnosis arm for comparison.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.artifact_verification import (
    ARTIFACT_FAMILY_LEAKAGE,
    ARTIFACT_SIGNIFICANCE_ONLY,
    ARTIFACT_SIMULATOR,
    ARTIFACT_TOPOLOGY_DEPENDENCE,
    evidence_context_from_hypothesis,
    generate_artifact_models,
    rank_artifact_models,
)

ART = ROOT / "artifacts"
OUT = ART / "dead_finding_diagnosis_bench.json"
REPORT = ART / "dead_finding_diagnosis_report.json"

# Map human failure labels → artifact_ids the generator can emit
FAILURE_TO_ARTIFACT: dict[str, set[str]] = {
    "distribution_leakage": {ARTIFACT_FAMILY_LEAKAGE},
    "scope_inflation": {ARTIFACT_TOPOLOGY_DEPENDENCE, ARTIFACT_SIGNIFICANCE_ONLY},
    "single_context": {ARTIFACT_TOPOLOGY_DEPENDENCE, ARTIFACT_SIMULATOR},
    "simulator_artifact": {ARTIFACT_SIMULATOR, ARTIFACT_TOPOLOGY_DEPENDENCE},
    "significance_only": {ARTIFACT_SIGNIFICANCE_ONLY},
    "sample_size": {"sample_size_artifact", ARTIFACT_SIGNIFICANCE_ONLY},
    "overfitting": {"overfitting", ARTIFACT_SIGNIFICANCE_ONLY},
}

# 10 cases: mix of scope_inflation, topology-ish, distribution_leakage, simulator
SELECTED_IDS = [
    "10c7ce0f-3dad-42c5-9790-0534dd55b970",  # scope + sig + single
    "1779cb4a-ee69-4d7e-b7de-ac3675e1bd32",  # scope + single
    "6f18f40c-2603-49b4-92a5-084831f3b58a",  # simulator + single
    "4e3f9550-1c0c-4b6c-977a-d97a461814e5",  # simulator + scope
    "764a324a-eb45-444a-aa79-71cced568ceb",  # sample + overfit + sig
    "231422d8-ab7a-4e4c-9d04-dc4aefb8cca5",  # simulator + single
    "f7020abb-71c0-4edd-8bcf-483f84826a00",  # scope + sig + single
    "8d2473eb-e716-51bf-9aed-00644c9189e5",  # distribution_leakage mandrake
    "9429430f-201d-5fcf-b68a-2858d62f3306",  # distribution_leakage mandrake
    "4b3ba17d-fdab-4762-a4b6-690c4ccce546",  # scope + sig
]

LLM_PROMPT = """You are an artifact-model generator for scientific findings.

A finding looked confirmed in-campaign but may be fake. Given ONLY the claim and evidence below,
list the top 3 plausible ARTIFACT explanations (why it might look real but not generalize).

Do NOT use labels like "scope inflation" or "distribution leakage" — describe concrete artifact mechanisms.

Return JSON ONLY:
{{"artifacts": [
  {{"artifact_id": "snake_case_id", "description": "...", "why_plausible": "...", "proposed_test": "..."}}
]}}

Use artifact_id from: family_leakage, topology_dependence, significance_only, simulator_specific,
sample_size_artifact, overfitting, measurement_bias, feature_redundancy, distribution_shift.

Claim:
{claim}

Evidence:
{evidence}
"""


def _load_api_key() -> str:
    key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _gemini(prompt: str, model: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, params={"key": api_key}, json=payload)
    r.raise_for_status()
    parts = (((r.json().get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


def _parse_llm_artifacts(raw: str) -> list[dict]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return []
        data = json.loads(m.group(0))
    return [a for a in (data.get("artifacts") or []) if isinstance(a, dict)]


def _expected_artifacts(failure_types: list[str]) -> set[str]:
    out: set[str] = set()
    for ft in failure_types:
        out |= FAILURE_TO_ARTIFACT.get(ft, set())
    return out


def _score(top_ids: list[str], expected: set[str]) -> dict:
    top1 = top_ids[0] if top_ids else None
    top3 = set(top_ids[:3])
    return {
        "top1_id": top1,
        "top3_ids": top_ids[:3],
        "top1_correct": top1 in expected if top1 else False,
        "top3_hit": bool(top3 & expected),
        "expected": sorted(expected),
    }


def _build_cases() -> list[dict]:
    dead = {f["id"]: f for f in json.loads((ART / "dead_findings_classification.json").read_text(encoding="utf-8"))["findings"]}
    contagion = {r["id"]: r for r in json.loads((ART / "contagion_14_confirmed.json").read_text(encoding="utf-8"))}
    mandrake = {r["hypothesis_id"]: r for r in json.loads((ART / "confirmed_findings_audit.json").read_text(encoding="utf-8"))["findings"]}
    perm = {r["hypothesis_id"]: r for r in json.loads((ART / "contagion_confirmed_permutation_audit.json").read_text(encoding="utf-8"))["findings"]}

    cases = []
    for fid in SELECTED_IDS:
        gt = dead.get(fid)
        if not gt:
            continue
        domain = gt["domain"]
        if domain == "mandrake":
            row = mandrake.get(fid, {})
            claim = row.get("text") or ""
            evidence = {
                "lofo_r2": row.get("lofo_r2"),
                "lofo_gap": row.get("lofo_gap"),
                "features": row.get("features"),
                "methodology": "LOFO",
                "family_column": "rt_family",
                "n_families": 7,
                "n_features": len(row.get("features") or []),
            }
            domain_bucket = None
            tools = ["mandrake_verification"]
        else:
            row = contagion.get(fid, {})
            claim = row.get("text") or ""
            p_row = perm.get(fid, {})
            evidence = {
                "p_value": row.get("p_value"),
                "metric_value": row.get("metric_value"),
                "effect_size": row.get("effect_size"),
                "verdict_reason": row.get("verdict_reason"),
                "lofo_r2": p_row.get("observed_lofo_r2"),
                "methodology": "sandbox_simulation",
            }
            domain_bucket = "graphs"
            tools = ["__code__", "statistical_significance"]
        cases.append({
            "id": fid,
            "domain": domain,
            "ground_truth_failure_types": gt["failure_types"],
            "ground_truth_note": gt["note"],
            "claim": claim,
            "evidence": evidence,
            "domain_bucket": domain_bucket,
            "tools_used": tools,
        })
    return cases


def run(*, use_llm: bool = True) -> dict:
    cases = _build_cases()
    api_key = _load_api_key() if use_llm else ""
    model = "gemini-3-flash-preview"
    results = []

    for case in cases:
        expected = _expected_artifacts(case["ground_truth_failure_types"])
        ctx = evidence_context_from_hypothesis(
            case["claim"],
            case["evidence"],
            tools_used=case["tools_used"],
            methodology=case["evidence"].get("methodology") or "",
            domain_bucket=case["domain_bucket"],
        )
        models = generate_artifact_models(ctx)
        ranked = rank_artifact_models(models, top_k=3)
        propab_ids = [m.artifact_id for m in ranked]

        entry = {
            "id": case["id"],
            "domain": case["domain"],
            "claim_snippet": case["claim"][:160],
            "ground_truth_failure_types": case["ground_truth_failure_types"],
            "ground_truth_note": case["ground_truth_note"],
            "propab_generator": {
                "ranked": [{"artifact_id": m.artifact_id, "description": m.description, "why_plausible": m.why_plausible} for m in ranked],
                **_score(propab_ids, expected),
            },
        }

        if use_llm and api_key:
            prompt = LLM_PROMPT.format(
                claim=case["claim"],
                evidence=json.dumps(case["evidence"], indent=2),
            )
            try:
                raw = _gemini(prompt, model, api_key)
                llm_arts = _parse_llm_artifacts(raw)
                llm_ids = [str(a.get("artifact_id") or "") for a in llm_arts]
                entry["llm_flash"] = {
                    "artifacts": llm_arts[:3],
                    **_score(llm_ids, expected),
                }
            except Exception as exc:  # noqa: BLE001
                entry["llm_flash"] = {"error": str(exc)}
            time.sleep(0.4)

        results.append(entry)

    n = len(results)
    propab_top1 = sum(1 for r in results if r["propab_generator"]["top1_correct"])
    propab_top3 = sum(1 for r in results if r["propab_generator"]["top3_hit"])
    llm_top1 = sum(1 for r in results if r.get("llm_flash", {}).get("top1_correct"))
    llm_top3 = sum(1 for r in results if r.get("llm_flash", {}).get("top3_hit"))

    topo_default = sum(
        1 for c in results
        if c["domain"] == "contagion" and c["propab_generator"]["top1_id"] == ARTIFACT_TOPOLOGY_DEPENDENCE
    )
    contagion_n = sum(1 for c in results if c["domain"] == "contagion")
    unique_top1 = sorted({c["propab_generator"]["top1_id"] for c in results})

    report = {
        "bench": "dead_finding_diagnosis",
        "n_cases": n,
        "propab_generate_artifact_models": {
            "top1_accuracy": round(propab_top1 / n, 3) if n else 0,
            "top3_recall": round(propab_top3 / n, 3) if n else 0,
            "top1_correct": propab_top1,
            "top3_hits": propab_top3,
            "contagion_topology_top1_rate": f"{topo_default}/{contagion_n}",
            "unique_top1_labels": unique_top1,
            "specificity_warning": (
                "Contagion findings almost always get topology_dependence as #1 — "
                "high accuracy may reflect a generic default, not case-specific diagnosis."
                if topo_default >= contagion_n - 1 else None
            ),
        },
        "llm_flash_unprompted": {
            "model": model if use_llm else None,
            "top1_accuracy": round(llm_top1 / n, 3) if n and use_llm else None,
            "top3_recall": round(llm_top3 / n, 3) if n and use_llm else None,
            "top1_correct": llm_top1,
            "top3_hits": llm_top3,
        },
        "verdict": "",
        "cases": results,
    }

    if propab_top1 >= n * 0.6 and topo_default < contagion_n:
        report["verdict"] = (
            "Propab artifact generator correctly diagnoses most dead findings unprompted — "
            "useful for pre-audit flagging on future campaigns."
        )
    elif propab_top1 >= n * 0.6:
        report["verdict"] = (
            f"Propab artifact generator hits {propab_top1}/{n} top-1 labels but "
            f"{topo_default}/{contagion_n} contagion cases get topology_dependence by default — "
            "correct bucket, not specific diagnosis. Useful as a coarse 'needs OOD holdout' flag, "
            "not as unprompted identification of simulator vs sample-size vs leakage failure modes. "
            "Mandrake family_leakage detection works when LOFO metadata present."
        )
    elif propab_top3 >= n * 0.7:
        report["verdict"] = (
            "Top-1 often wrong but top-3 usually contains the right artifact — "
            "generator is useful as a checklist, not a single-label classifier."
        )
    else:
        report["verdict"] = (
            "Propab artifact generator does NOT reliably identify actual failure modes unprompted — "
            "produces generic topology/significance diagnoses. Same shallow pattern-matching as other layers; "
            "artifact generator needs upstream fix, not more downstream gates."
        )

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "cases"}
    REPORT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    llm = "--no-llm" not in sys.argv
    r = run(use_llm=llm)
    print(json.dumps({k: r[k] for k in r if k != "cases"}, indent=2))
