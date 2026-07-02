#!/usr/bin/env python3
"""
Part B Failure Interpreter bias bench (fixes.md step 1).

Run production build_failure_interpret_prompt on hand-labeled dead findings
(Section 7 classification). No ground-truth labels in the prompt.

Compare against Section 12 artifact-generator results on the same cases.
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

from propab.artifact_verification import (  # noqa: E402
    ARTIFACT_FAMILY_LEAKAGE,
    ARTIFACT_SIGNIFICANCE_ONLY,
    ARTIFACT_SIMULATOR,
    ARTIFACT_TOPOLOGY_DEPENDENCE,
    evidence_context_from_hypothesis,
    generate_artifact_models,
    rank_artifact_models,
)
from propab.hypothesis_tree import HypothesisNode, HypothesisTree  # noqa: E402

ART = ROOT / "artifacts"
OUT = ART / "interpreter_bias_bench.json"
REPORT = ART / "interpreter_bias_report.json"

# Same 10-case mix as dead_finding_diagnosis_bench.py
SELECTED_IDS = [
    "10c7ce0f-3dad-42c5-9790-0534dd55b970",
    "1779cb4a-ee69-4d7e-b7de-ac3675e1bd32",
    "6f18f40c-2603-49b4-92a5-084831f3b58a",
    "4e3f9550-1c0c-4b6c-977a-d97a461814e5",
    "764a324a-eb45-444a-aa79-71cced568ceb",
    "231422d8-ab7a-4e4c-9d04-dc4aefb8cca5",
    "f7020abb-71c0-4edd-8bcf-483f84826a00",
    "8d2473eb-e716-51bf-9aed-00644c9189e5",
    "9429430f-201d-5fcf-b68a-2858d62f3306",
    "4b3ba17d-fdab-4762-a4b6-690c4ccce546",
]

STRICT_GT = frozenset({
    "distribution_leakage",
    "simulator_artifact",
    "significance_only",
    "sample_size",
})

SCOPE_GT = frozenset({"scope_inflation", "single_context"})

GT_ACCEPTABLE: dict[str, set[str]] = {
    "distribution_leakage": {"distribution_leakage"},
    "simulator_artifact": {"simulator_artifact"},
    "significance_only": {"significance_only"},
    "sample_size": {"sample_size", "significance_only"},
    "overfitting": {"overfitting", "sample_size", "significance_only"},
    "scope_inflation": {"scope_inflation", "single_context", "topology_dependence"},
    "single_context": {"single_context", "scope_inflation", "topology_dependence"},
}

PRIORITY = [
    "distribution_leakage",
    "simulator_artifact",
    "significance_only",
    "sample_size",
    "overfitting",
    "scope_inflation",
    "single_context",
]

CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "distribution_leakage": re.compile(
        r"family.{0,25}(leak|proxy|surrogate|identity|specific)|"
        r"lofo.{0,20}(gap|negative|degrad|hold|collapse)|"
        r"group.{0,15}identity|distribution.{0,12}leak|lineage|rt_family",
        re.I,
    ),
    "simulator_artifact": re.compile(
        r"simulator|simulation.{0,20}(path|specific|artifact|engine)|"
        r"implementation|procedural|rewiring|sandbox|sobel|mediation.{0,15}artifact|"
        r"bespoke.{0,15}(subgraph|pipeline)",
        re.I,
    ),
    "significance_only": re.compile(
        r"p.?value|significance.{0,20}(only|gate|without|driven)|"
        r"replic|unreplic|p.?hack|noise.{0,12}robust|statistical.{0,12}fluke",
        re.I,
    ),
    "sample_size": re.compile(
        r"sample.?size|underpower|n=\s*1000|trials.{0,15}variance|power.{0,12}analysis",
        re.I,
    ),
    "topology_dependence": re.compile(
        r"topology|graph.?family|cross.?family|ood|hold.?out|transfer|generali|"
        r"single.?context|one.?family|modular.{0,12}only|ws/er|non.?modular",
        re.I,
    ),
    "scope_inflation": re.compile(
        r"scope.{0,12}inflat|over.?general|claims.{0,15}universal|without.{0,12}ood",
        re.I,
    ),
}

CONTAGION_QUESTION = (
    "Investigate which structural properties of complex networks most strongly "
    "determine the speed and extent of contagion spreading under competing diffusion models."
)
MANDRAKE_QUESTION = (
    "Which biophysical properties predict RT activity independently of evolutionary family membership?"
)


def _primary_gt(failure_types: list[str]) -> str:
    for label in PRIORITY:
        if label in failure_types:
            return label
    return failure_types[0] if failure_types else "unknown"


def _load_api_key() -> str:
    key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GOOGLE_API_KEY=") or line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _load_model() -> str:
    for key in ("GEMINI_MODEL", "GOOGLE_MODEL"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_MODEL="):
                return line.split("=", 1)[1].strip()
    return "gemini-2.5-flash"


def _gemini(prompt: str, model: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, params={"key": api_key}, json=payload)
    r.raise_for_status()
    parts = (((r.json().get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


def _parse_interpreter(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        data = json.loads(m.group(0))
    return data if isinstance(data, dict) else {}


def _interpreter_text(interp: dict) -> str:
    parts = [
        str(interp.get("confound_hypothesis") or ""),
        str(interp.get("failure_mechanism") or ""),
        str(interp.get("diagnostic_experiment") or ""),
        " ".join(interp.get("rules_out") or []),
        " ".join(interp.get("implies") or []),
    ]
    return " ".join(parts)


def _detect_categories(text: str) -> set[str]:
    found: set[str] = set()
    for label, pat in CATEGORY_PATTERNS.items():
        if pat.search(text):
            found.add(label)
    return found


def _score_interpreter(detected: set[str], failure_types: list[str]) -> dict:
    primary = _primary_gt(failure_types)
    acceptable = GT_ACCEPTABLE.get(primary, {primary})
    hit = bool(detected & acceptable)
    strict = primary in STRICT_GT
    scope_ok = primary in SCOPE_GT and bool(detected & {"topology_dependence", "scope_inflation", "single_context"})
    topology_only = (
        "topology_dependence" in detected
        and not (detected & {"simulator_artifact", "significance_only", "sample_size", "distribution_leakage"})
    )
    misleading = strict and topology_only and primary not in SCOPE_GT
    return {
        "primary_ground_truth": primary,
        "detected_categories": sorted(detected),
        "acceptable_for_primary": sorted(acceptable),
        "primary_correct": hit or scope_ok,
        "strict_primary_correct": hit if strict else scope_ok,
        "topology_only_diagnosis": topology_only,
        "misleading_topology_default": misleading,
    }


def _build_cases() -> list[dict]:
    dead = {
        f["id"]: f
        for f in json.loads((ART / "dead_findings_classification.json").read_text(encoding="utf-8"))["findings"]
    }
    contagion = {r["id"]: r for r in json.loads((ART / "contagion_14_confirmed.json").read_text(encoding="utf-8"))}
    mandrake = {
        r["hypothesis_id"]: r
        for r in json.loads((ART / "confirmed_findings_audit.json").read_text(encoding="utf-8"))["findings"]
    }
    perm = {
        r["hypothesis_id"]: r
        for r in json.loads((ART / "contagion_confirmed_permutation_audit.json").read_text(encoding="utf-8"))["findings"]
    }

    cases: list[dict] = []
    for fid in SELECTED_IDS:
        gt = dead.get(fid)
        if not gt:
            continue
        domain = gt["domain"]
        if domain == "mandrake":
            row = mandrake.get(fid, {})
            question = MANDRAKE_QUESTION
            evidence_obj = {
                "verdict_reason": row.get("rationale") or "LOFO audit failed — family surrogate",
                "lofo_r2": row.get("lofo_r2"),
                "lofo_gap": row.get("lofo_gap"),
                "features": row.get("features"),
                "methodology": "LOFO cross-validation",
                "family_column": "rt_family",
                "n_families": 7,
                "artifact_gate": {
                    "verdict": "likely_artifact",
                    "verdict_reason": "negative LOFO with large gap — family leakage",
                    "ranked_artifacts": [{"artifact_id": ARTIFACT_FAMILY_LEAKAGE, "score": 0.9}],
                },
            }
            inconclusive_reason = "replication_failed"
            failure_signature = f"lofo_r2={row.get('lofo_r2')} gap={row.get('lofo_gap')}"
        else:
            row = contagion.get(fid, {})
            p_row = perm.get(fid, {})
            question = CONTAGION_QUESTION
            evidence_obj = {
                "verdict_reason": row.get("verdict_reason") or "significance gate passed in-campaign",
                "p_value": row.get("p_value"),
                "metric_value": row.get("metric_value"),
                "effect_size": row.get("effect_size"),
                "lofo_r2": p_row.get("observed_lofo_r2"),
                "permutation_verdict": p_row.get("verdict"),
                "methodology": "sandbox_simulation",
            }
            inconclusive_reason = "replication_failed"
            failure_signature = gt["note"][:200]

        cases.append({
            "id": fid,
            "domain": domain,
            "question": question,
            "ground_truth_failure_types": gt["failure_types"],
            "ground_truth_note": gt["note"],
            "claim": row.get("text") or "",
            "confidence": float(row.get("confidence") or 0.9),
            "evidence_summary": json.dumps(evidence_obj, ensure_ascii=False),
            "inconclusive_reason": inconclusive_reason,
            "failure_signature": failure_signature,
        })
    return cases


def _artifact_top1(case: dict) -> str | None:
    domain_bucket = None if case["domain"] == "mandrake" else "graphs"
    tools = ["mandrake_verification"] if case["domain"] == "mandrake" else ["__code__", "statistical_significance"]
    ev = json.loads(case["evidence_summary"])
    ctx = evidence_context_from_hypothesis(
        case["claim"],
        ev,
        tools_used=tools,
        methodology=ev.get("methodology") or "",
        domain_bucket=domain_bucket,
    )
    ranked = rank_artifact_models(generate_artifact_models(ctx), top_k=1)
    return ranked[0].artifact_id if ranked else None


def run(*, use_llm: bool = True) -> dict:
    api_key = _load_api_key() if use_llm else ""
    model = _load_model()
    if use_llm and not api_key:
        print("ERROR: GOOGLE_API_KEY / GEMINI_API_KEY required", file=sys.stderr)
        sys.exit(1)

    cases = _build_cases()
    results: list[dict] = []

    for case in cases:
        node = HypothesisNode(
            id=case["id"],
            text=case["claim"],
            parent_id=None,
            depth=1,
            verdict="refuted",
            confidence=case["confidence"],
            evidence_summary=case["evidence_summary"],
            inconclusive_reason=case["inconclusive_reason"],
            failure_signature=case["failure_signature"],
            node_role="DISCOVERY",
        )
        tree = HypothesisTree()
        tree.nodes[node.id] = node
        prompt = tree.build_failure_interpret_prompt(node.id, question=case["question"])
        if not prompt:
            results.append({"id": case["id"], "error": "prompt_build_failed"})
            continue

        entry: dict = {
            "id": case["id"],
            "domain": case["domain"],
            "claim_snippet": case["claim"][:160],
            "ground_truth_failure_types": case["ground_truth_failure_types"],
            "ground_truth_note": case["ground_truth_note"],
            "artifact_generator_top1": _artifact_top1(case),
        }

        if use_llm:
            try:
                raw = _gemini(prompt, model, api_key)
                interp = _parse_interpreter(raw)
                text = _interpreter_text(interp)
                detected = _detect_categories(text)
                score = _score_interpreter(detected, case["ground_truth_failure_types"])
                entry["interpreter"] = {
                    "model": model,
                    "parsed": interp,
                    **score,
                }
            except Exception as exc:  # noqa: BLE001
                entry["interpreter"] = {"error": str(exc)}
            time.sleep(0.35)

        results.append(entry)

    n = len([r for r in results if r.get("interpreter") and "error" not in r["interpreter"]])
    primary_ok = sum(1 for r in results if r.get("interpreter", {}).get("primary_correct"))
    strict_ok = sum(1 for r in results if r.get("interpreter", {}).get("strict_primary_correct"))
    misleading = sum(1 for r in results if r.get("interpreter", {}).get("misleading_topology_default"))
    topo_only = sum(1 for r in results if r.get("interpreter", {}).get("topology_only_diagnosis"))

    contagion = [r for r in results if r.get("domain") == "contagion" and r.get("interpreter")]
    mandrake = [r for r in results if r.get("domain") == "mandrake" and r.get("interpreter")]
    contagion_topo = sum(
        1 for r in contagion
        if "topology_dependence" in (r.get("interpreter") or {}).get("detected_categories", [])
    )
    artifact_topo = sum(
        1 for r in contagion
        if r.get("artifact_generator_top1") == ARTIFACT_TOPOLOGY_DEPENDENCE
    )

    strict_cases = [r for r in results if _primary_gt(r["ground_truth_failure_types"]) in STRICT_GT]
    strict_hits = sum(1 for r in strict_cases if r.get("interpreter", {}).get("strict_primary_correct"))

    report = {
        "bench": "interpreter_bias",
        "model": model if use_llm else None,
        "n_cases": len(results),
        "n_scored": n,
        "failure_interpreter": {
            "primary_correct_rate": round(primary_ok / n, 3) if n else 0,
            "strict_primary_correct_rate": round(strict_ok / n, 3) if n else 0,
            "primary_correct": primary_ok,
            "strict_primary_correct": strict_ok,
            "misleading_topology_default": misleading,
            "topology_only_count": topo_only,
            "contagion_topology_detect_rate": f"{contagion_topo}/{len(contagion)}" if contagion else "0/0",
            "mandrake_family_leak_hits": sum(
                1 for r in mandrake
                if "distribution_leakage" in (r.get("interpreter") or {}).get("detected_categories", [])
            ),
        },
        "section_12_comparison": {
            "artifact_generator_contagion_topology_top1": f"{artifact_topo}/{len(contagion)}" if contagion else "0/0",
            "interpreter_contagion_topology_detect": f"{contagion_topo}/{len(contagion)}" if contagion else "0/0",
            "same_dominant_pattern": contagion_topo >= len(contagion) - 1 and artifact_topo >= len(contagion) - 1,
        },
        "strict_type_breakdown": {
            "n_strict_cases": len(strict_cases),
            "strict_correct": strict_hits,
            "strict_accuracy": round(strict_hits / len(strict_cases), 3) if strict_cases else 0,
        },
        "verdict": "",
        "cases": results,
    }

    if strict_hits < len(strict_cases) * 0.5:
        report["verdict"] = (
            f"FAIL — interpreter misses specific failure types on strict cases "
            f"({strict_hits}/{len(strict_cases)}). Same dominant-pattern bias as Section 12 artifact "
            f"generator ({artifact_topo}/{len(contagion)} contagion → topology_dependence). "
            "Part B default-off is correct; prompt tuning will not fix this — diagnosis needs "
            "independent verification, not another single LLM call."
        )
    elif misleading >= 2:
        report["verdict"] = (
            f"PARTIAL FAIL — {misleading} cases got confident topology/LOFO diagnoses on non-topology "
            "ground truth (simulator/significance/sample-size). Coarse scope cases OK but "
            "wrong confident steering on strict types."
        )
    elif primary_ok >= n * 0.7:
        report["verdict"] = (
            f"PASS (coarse) — {primary_ok}/{n} primary-correct across varied failure types. "
            "Re-run matched A/B campaign (fixes.md step 2) before default-on."
        )
    else:
        report["verdict"] = (
            f"MARGINAL — {primary_ok}/{n} primary-correct, {strict_hits}/{len(strict_cases)} on strict types. "
            "Not enough to trust Part B for default-on."
        )

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "cases"}
    REPORT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    r = run(use_llm=not dry)
    print(json.dumps({k: r[k] for k in r if k != "cases"}, indent=2))
