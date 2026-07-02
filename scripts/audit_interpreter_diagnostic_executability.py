#!/usr/bin/env python3
"""Audit diagnostic_experiment executability from interpreter bias bench (fixes.md)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "artifacts" / "interpreter_bias_bench.json"
OUT = ROOT / "artifacts" / "interpreter_diagnostic_executability_audit.json"
REPORT = ROOT / "artifacts" / "interpreter_diagnostic_executability_report.json"

# Existing verification templates (artifact_verification.py + mandrake_adapter)
KNOWN_TEMPLATES = {
    "label_shuffle_lofo": re.compile(
        r"label.{0,12}shuff|shuff.{0,12}label|family.{0,12}label.{0,12}null|null.{0,12}distribution",
        re.I,
    ),
    "lofo_holdout": re.compile(
        r"lofo|leave.{0,5}one.{0,5}(family|group|topology)|held.?out.{0,12}(topology|family|group)",
        re.I,
    ),
    "permutation_null": re.compile(
        r"permutation.{0,12}(test|null|p)|null.{0,12}distribution|bootstrap",
        re.I,
    ),
    "family_classifier": re.compile(
        r"train.{0,20}classifier.{0,30}(family|rt_family|group)|predict.{0,20}family",
        re.I,
    ),
    "topology_holdout": re.compile(
        r"held.?out.{0,12}topology|cross.?topology|leave.{0,5}one.{0,5}topology|ood",
        re.I,
    ),
    "replicate_same_method": re.compile(
        r"replicate.{0,20}(sobel|mediation|analysis)|re.?evaluat.{0,20}same",
        re.I,
    ),
}

VAGUE_MARKERS = re.compile(
    r"systematic(ally)?|broad(ly)?|diverse.{0,12}(set|spectrum|range|ensemble)|"
    r"large.?scale|comprehensive|map.{0,12}boundar|across.{0,12}a.{0,8}(broad|diverse|significantly larger)|"
    r"varying.{0,12}parameters|controlled experiment varying",
    re.I,
)

CONCRETE_MARKERS = re.compile(
    r"\b(lofo|label.?shuff|permutation|held.?out|rt_family|7 families|classifier)\b|"
    r"high accuracy would confirm|negative LOFO|p95|null p",
    re.I,
)


def _classify(text: str, domain: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"tier": "missing", "templates": [], "rationale": "empty field"}

    templates = [name for name, pat in KNOWN_TEMPLATES.items() if pat.search(text)]
    vague = bool(VAGUE_MARKERS.search(text))
    concrete = bool(CONCRETE_MARKERS.search(text))

    # Mandrake family-classifier tests are one-shot runnable
    if domain == "mandrake" and "family_classifier" in templates:
        return {
            "tier": "runnable",
            "templates": templates,
            "existing_machinery": ["mandrake_verification", "label_shuffle_lofo", "lofo_holdout"],
            "rationale": "Names family-membership classifier — maps to mandrake LOFO/label-shuffle null.",
            "gate_buildable": True,
        }

    if "label_shuffle_lofo" in templates and not vague:
        return {
            "tier": "runnable",
            "templates": templates,
            "existing_machinery": ["label_shuffle_lofo"],
            "rationale": "Explicit label-shuffle / LOFO null — scriptable.",
            "gate_buildable": True,
        }

    if templates == ["topology_holdout"] or (
        "topology_holdout" in templates and "held-out topology" in text.lower()
    ):
        if vague:
            return {
                "tier": "partial",
                "templates": templates,
                "existing_machinery": ["held_out_group", "topology_holdout"],
                "rationale": "Names holdout/OOD but not which families/topologies or pass threshold.",
                "gate_buildable": "needs_operationalizer",
            }
        return {
            "tier": "partial",
            "templates": templates,
            "existing_machinery": ["held_out_group"],
            "rationale": "Holdout named; parameters underspecified.",
            "gate_buildable": "needs_operationalizer",
        }

    if "permutation_null" in templates and vague:
        return {
            "tier": "partial",
            "templates": templates,
            "existing_machinery": ["permutation_null"],
            "rationale": "Mentions permutation but embedded in open-ended sweep.",
            "gate_buildable": "needs_operationalizer",
        }

    if "replicate_same_method" in templates:
        return {
            "tier": "partial",
            "templates": templates,
            "existing_machinery": ["alternate_simulator", "held_out_group"],
            "rationale": "Names method (Sobel) + stratified topologies but not fixed pass/fail.",
            "gate_buildable": "needs_operationalizer",
        }

    if vague or len(text.split()) > 35:
        return {
            "tier": "vague",
            "templates": templates,
            "existing_machinery": [],
            "rationale": "Sweep / sensitivity study prose — not a single falsifiable check.",
            "gate_buildable": False,
        }

    if concrete and templates:
        return {
            "tier": "partial",
            "templates": templates,
            "existing_machinery": templates,
            "rationale": "Some concrete keywords but insufficient specification.",
            "gate_buildable": "needs_operationalizer",
        }

    return {
        "tier": "vague",
        "templates": templates,
        "existing_machinery": [],
        "rationale": "No mappable single test.",
        "gate_buildable": False,
    }


def main() -> None:
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    rows = []
    for case in bench.get("cases") or []:
        interp = (case.get("interpreter") or {}).get("parsed") or {}
        diag = interp.get("diagnostic_experiment") or ""
        cls = _classify(diag, case.get("domain") or "")
        rows.append({
            "id": case["id"],
            "domain": case.get("domain"),
            "ground_truth_failure_types": case.get("ground_truth_failure_types"),
            "confound_hypothesis": (interp.get("confound_hypothesis") or "")[:200],
            "diagnostic_experiment": diag,
            **cls,
        })

    tiers = [r["tier"] for r in rows]
    n = len(rows)
    runnable = sum(1 for t in tiers if t == "runnable")
    partial = sum(1 for t in tiers if t == "partial")
    vague = sum(1 for t in tiers if t == "vague")

    contagion = [r for r in rows if r["domain"] == "contagion"]
    mandrake = [r for r in rows if r["domain"] == "mandrake"]

    report = {
        "audit": "interpreter_diagnostic_executability",
        "source": str(BENCH),
        "n_cases": n,
        "tier_counts": {"runnable": runnable, "partial": partial, "vague": vague},
        "rates": {
            "runnable_pct": round(100 * runnable / n, 1) if n else 0,
            "partial_pct": round(100 * partial / n, 1) if n else 0,
            "vague_pct": round(100 * vague / n, 1) if n else 0,
        },
        "by_domain": {
            "contagion": {
                "n": len(contagion),
                "runnable": sum(1 for r in contagion if r["tier"] == "runnable"),
                "partial": sum(1 for r in contagion if r["tier"] == "partial"),
                "vague": sum(1 for r in contagion if r["tier"] == "vague"),
            },
            "mandrake": {
                "n": len(mandrake),
                "runnable": sum(1 for r in mandrake if r["tier"] == "runnable"),
                "partial": sum(1 for r in mandrake if r["tier"] == "partial"),
                "vague": sum(1 for r in mandrake if r["tier"] == "vague"),
            },
        },
        "fixes_md_answer": "",
        "recommendation": "",
        "cases": rows,
    }

    if runnable >= n * 0.6:
        answer = (
            "Most diagnostic_experiment fields are concrete enough to gate expansion "
            "with existing LOFO/permutation machinery."
        )
        rec = "Build interpreter-diagnosis verification gate (Section 6 pattern upstream)."
    elif runnable + partial >= n * 0.5 and runnable >= 2:
        answer = (
            f"Only {runnable}/{n} are directly runnable; {partial}/{n} need operationalization; "
            f"{vague}/{n} are vague sweeps. Gate is buildable for mandrake/LOFO domains only — "
            "contagion diagnostics are mostly non-scriptable prose."
        )
        rec = (
            "Do NOT build a generic auto-gate yet. Either constrain interpreter schema to "
            "emit enum test_id from {label_shuffle_lofo, lofo_holdout, permutation_null, ...} "
            "OR operationalize only for domains with existing verification adapters (mandrake first)."
        )
    else:
        answer = (
            f"Diagnostic experiments are mostly vague ({vague}/{n}). A second LLM call to "
            "operationalize would reintroduce the double-LLM dependency fixes.md warns about."
        )
        rec = "Fix interpreter output schema before building gate — require structured test_id + params."

    report["fixes_md_answer"] = answer
    report["recommendation"] = rec

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "cases"}
    REPORT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
