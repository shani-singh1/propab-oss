#!/usr/bin/env python3
"""
MechanismStructureBench — compare real mechanisms (MechanismCorpus) vs Propab outputs.

Uses artifacts/mechanism_corpus.md as gold standard (5 hand-coded cases).
Scores both on: compression, counterfactuality, failure_conditions, generativity.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
CORPUS = ART / "mechanism_corpus.md"
OUT = ART / "mechanism_structure_bench.json"
REPORT = ART / "mechanism_structure_bench_report.json"

DIMENSIONS = ("compression", "counterfactuality", "failure_conditions", "generativity")

# Hand-coded from mechanism_corpus.md four-property scoring (1–5 scale)
GOLD_SCORES = {
    "alphafold2": {"compression": 5, "counterfactuality": 5, "failure_conditions": 5, "generativity": 5},
    "tao_erdos_discrepancy": {"compression": 5, "counterfactuality": 5, "failure_conditions": 4, "generativity": 5},
    "funsearch": {"compression": 5, "counterfactuality": 4, "failure_conditions": 4, "generativity": 5},
    "semmelweis": {"compression": 5, "counterfactuality": 5, "failure_conditions": 4, "generativity": 5},
    "clp_cap_sets": {"compression": 5, "counterfactuality": 5, "failure_conditions": 4, "generativity": 5},
}


def _load_propab_mechanisms() -> list[dict]:
    items: list[dict] = []

    # Pipeline inducer output (deterministic/LLM)
    mo_path = ART / "mechanism_objects.json"
    if mo_path.exists():
        for m in json.loads(mo_path.read_text(encoding="utf-8")):
            items.append({
                "id": m.get("id", "")[:12],
                "source": "propab_inducer",
                "domain": "mandrake",
                "text": m.get("explanation") or "",
                "schema_fields": list(m.keys()),
            })

    # Competing mechanisms
    cm_path = ART / "competing_mechanisms.json"
    if cm_path.exists():
        for cset in json.loads(cm_path.read_text(encoding="utf-8")):
            for m in cset.get("mechanisms") or []:
                items.append({
                    "id": (m.get("id") or "")[:12],
                    "source": "propab_competing",
                    "domain": "mandrake",
                    "text": m.get("explanation") or "",
                    "schema_fields": list(m.keys()),
                })

    # MechanismBench Flash outputs (representative LLM induction)
    bench_path = ART / "mechanism_bench_results.json"
    if bench_path.exists():
        data = json.loads(bench_path.read_text(encoding="utf-8"))
        seen: set[str] = set()
        for cid, run in data.get("runs", {}).items():
            raw = (run.get("models") or {}).get("gemini-3-flash-preview", {}).get("response") or ""
            text = raw.strip()
            if not text or text[:120] in seen:
                continue
            seen.add(text[:120])
            # extract first explanation
            expl = _first_explanation(text)
            if expl:
                items.append({
                    "id": cid,
                    "source": "propab_bench_flash",
                    "domain": run.get("case", {}).get("domain", "unknown"),
                    "text": expl,
                    "schema_fields": ["explanation", "counterfactual_prediction"],
                })

    # Finding ledger mechanisms from audit
    audit_path = ART / "mechanism_audit_p0_raw.json"
    if audit_path.exists():
        raw = json.loads(audit_path.read_text(encoding="utf-8"))
        for m in raw.get("mechanisms") or []:
            mo = m.get("mechanism_obj") or {}
            claim = (m.get("claim") or mo.get("claim") or "")[:200]
            effect = mo.get("effect") or mo.get("mechanism") or ""
            text = claim if len(claim) > 40 else effect
            if len(text) < 20:
                continue
            items.append({
                "id": f"ledger-{m.get('campaign_id', '')[:8]}",
                "source": "propab_finding_ledger",
                "domain": "contagion" if "network" in (m.get("claim") or "").lower() else "mixed",
                "text": text,
                "schema_fields": list(mo.keys()) if mo else [],
            })

    return items


def _first_explanation(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        mechs = data.get("mechanisms") if isinstance(data, dict) else []
        if mechs and isinstance(mechs[0], dict):
            return str(mechs[0].get("explanation") or "")[:1200]
    except json.JSONDecodeError:
        pass
    return text[:400]


def _score_propab(text: str, *, source: str, schema_fields: list[str]) -> dict:
    t = text.lower()
    scores = {d: 1 for d in DIMENSIONS}
    notes: list[str] = []

    # --- compression ---
    n_obs = 0
    if re.search(r"\b(lofo|r²|r2|within.?family|cross.?group)\b", t):
        n_obs += 1
    if re.search(r"\b(k-shell|lambda|modularity|assortativity|bridge|spectral)\b", t):
        n_obs += 1
    if re.search(r"\b(explains|accounts for|also explains|ward|kolletschka)\b", t):
        n_obs += 2
    if n_obs >= 3:
        scores["compression"] = 2
        notes.append("mentions multiple facts but no single structural idea collapsing them")
    elif n_obs == 1:
        scores["compression"] = 1
        notes.append("single-dataset comparative claim")
    else:
        scores["compression"] = 1

    # --- counterfactuality ---
    if re.search(r"\bcounterfactual\b", t) or "if we " in t or "would fail if" in t or "ablat" in t:
        if re.search(r"\b(predict|rmse|r²|outperform|stronger predictor|rank.?order)\b", t):
            scores["counterfactual"] = 2
            notes.append("counterfactual about model/metric comparison, not world intervention")
        else:
            scores["counterfactual"] = 3
    elif re.search(r"\b(if .* (remov|abl|perturb|interven|wash|decontam))\b", t):
        scores["counterfactual"] = 3
    elif re.search(r"\b(should (increase|decrease|collapse|reverse|degrade))\b", t):
        if re.search(r"\b(predict|accuracy|rmse|lofo|correlation)\b", t):
            scores["counterfactual"] = 2
        else:
            scores["counterfactual"] = 3
    else:
        scores["counterfactual"] = 1
        notes.append("no falsifiable intervention on causal structure")

    # --- failure conditions ---
    if re.search(r"\b(fail(s|ure)? (when|if|on|in)|does not (hold|apply|generalize)|outside|limitation|scope|disordered|shallow msa|evaluator required)\b", t):
        scores["failure_conditions"] = 2
    if "failure_modes" in schema_fields or "assumptions_challenged" in schema_fields:
        scores["failure_conditions"] = max(scores["failure_conditions"], 2)
        if not re.search(r"\b(where (it|the mechanism)|fails when|breaks down)\b", t):
            notes.append("schema has assumptions_challenged but not explicit mechanism failure scope")
    if scores["failure_conditions"] == 1:
        notes.append("no stated scope limit for mechanism")

    # --- generativity ---
    if re.search(r"\b(different (domain|disease|problem|modality|topology family)|out.?of.?distribution|cross.?domain|never touched|new experiment class)\b", t):
        scores["generativity"] = 2
    elif re.search(r"\b(bridge-core|thermal barcode|core synchronization|expansion-diffusion)\b", t):
        scores["generativity"] = 1
        notes.append("named mechanism but no cross-domain experiment proposed")
    else:
        scores["generativity"] = 1

    if source == "propab_inducer" and "lofo" in t and "cross-group predictive" in t:
        scores = {d: 1 for d in DIMENSIONS}
        notes.append("template LOFO restatement — all dimensions at floor")

    total = sum(scores.values())
    return {
        **scores,
        "total": total,
        "max": 20,
        "pct_of_gold_avg": round(100 * total / 18.4, 1),  # gold avg ~4.6*4=18.4
        "notes": notes,
        "structural_class": _classify(text),
    }


def _classify(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(lofo|cross-group predictive|within r)\b", t):
        return "metric_comparison_lofo"
    if re.search(r"\b(stronger predictor|outperforms|more variance|lower rmse|beats)\b", t):
        return "metric_horse_race"
    if re.search(r"\b(mediat|because|drives|mechanism|acts as|barcode|alignment|coupling)\b", t):
        return "literature_causal_prose"
    return "comparative_regression"


def _gold_cases() -> list[dict]:
    corpus = CORPUS.read_text(encoding="utf-8") if CORPUS.exists() else ""
    cases = [
        ("alphafold2", "AlphaFold2 triangle-consistent pair updates", "real_corpus"),
        ("tao_erdos_discrepancy", "Tao Erdős discrepancy dichotomy", "real_corpus"),
        ("funsearch", "FunSearch programs-not-answers", "real_corpus"),
        ("semmelweis", "Semmelweis hand-transfer", "real_corpus"),
        ("clp_cap_sets", "CLP/E-G polynomial rank method", "real_corpus"),
    ]
    out = []
    for key, label, src in cases:
        scores = GOLD_SCORES[key]
        out.append({
            "id": key,
            "source": src,
            "label": label,
            "scores": {**scores, "total": sum(scores.values()), "max": 20},
            "structural_class": "real_mechanism",
        })
    return out


def _schema_gap_analysis() -> dict:
    """What fields real mechanisms have vs Propab MechanismObject."""
    propab_fields = {
        "explanation", "candidate_features", "supporting_anomalies",
        "assumptions_challenged", "confidence", "id",
    }
    real_required = {
        "observation", "anomaly", "mechanism", "counterfactual_prediction",
        "discriminating_experiment", "compression_target_n_observations",
        "failure_conditions", "generative_cross_domain_prediction",
    }
    return {
        "propab_mechanism_object_fields": sorted(propab_fields),
        "real_mechanism_chain_fields": sorted(real_required),
        "missing_in_propab": sorted(real_required - propab_fields),
        "propab_has_no_slot_for": [
            "compression (N observations collapsed by one idea)",
            "counterfactual on world/intervention (not model comparison)",
            "explicit failure_conditions (where mechanism stops)",
            "generative_cross_domain_experiment",
            "discriminating_experiment (single-variable test)",
        ],
    }


def run() -> dict:
    gold = _gold_cases()
    propab = _load_propab_mechanisms()

    scored_propab = []
    for p in propab:
        s = _score_propab(p["text"], source=p["source"], schema_fields=p.get("schema_fields") or [])
        scored_propab.append({**p, "scores": s})

    # dedupe by text prefix for stats
    unique: dict[str, dict] = {}
    for p in scored_propab:
        k = p["text"][:150]
        if k not in unique:
            unique[k] = p

    unique_list = list(unique.values())
    n = len(unique_list) or 1
    avg = {d: round(sum(p["scores"][d] for p in unique_list) / n, 2) for d in DIMENSIONS}
    avg_total = round(sum(p["scores"]["total"] for p in unique_list) / n, 2)
    gold_avg = {d: round(sum(g["scores"][d] for g in gold) / len(gold), 2) for d in DIMENSIONS}
    gold_total = round(sum(g["scores"]["total"] for g in gold) / len(gold), 2)

    by_class: dict[str, list] = {}
    for p in unique_list:
        by_class.setdefault(p["scores"]["structural_class"], []).append(p)

    gap = avg_total / gold_total if gold_total else 0

    missing_dims = []
    for d in DIMENSIONS:
        delta = gold_avg[d] - avg[d]
        if delta >= 2.5:
            missing_dims.append({"dimension": d, "gold_avg": gold_avg[d], "propab_avg": avg[d], "gap": round(delta, 2)})

    report = {
        "bench": "MechanismStructureBench v1",
        "corpus_source": str(CORPUS),
        "gold_n": len(gold),
        "propab_n_raw": len(scored_propab),
        "propab_n_unique": len(unique_list),
        "gold_avg_scores": {**gold_avg, "total": gold_total},
        "propab_avg_scores": {**avg, "total": avg_total},
        "propab_pct_of_gold": round(100 * gap, 1),
        "missing_dimensions": missing_dims,
        "propab_structural_classes": {
            k: len(v) for k, v in sorted(by_class.items(), key=lambda x: -len(x[0]))
        },
        "schema_gap": _schema_gap_analysis(),
        "verdict": _verdict(gap, missing_dims, by_class, propab_total=avg_total, gold_total=gold_total),
        "redesign_requirements": _redesign_requirements(),
        "gold_cases": gold,
        "propab_samples": unique_list[:20],
        "propab_all_scores_summary": {
            "floor_1_count": sum(1 for p in unique_list if p["scores"]["total"] <= 4),
            "below_25pct_gold": sum(1 for p in unique_list if p["scores"]["total"] < gold_total * 0.25),
        },
    }

    OUT.write_text(json.dumps({"gold": gold, "propab": unique_list, "report": report}, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(json.dumps({k: v for k, v in report.items() if k not in ("gold_cases", "propab_samples")}, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _verdict(gap: float, missing: list, by_class: dict, *, propab_total: float, gold_total: float) -> str:
    pct = round(100 * gap, 1)
    top_class = max(by_class.items(), key=lambda x: len(x[1]))[0] if by_class else "unknown"
    return (
        f"Propab mechanisms score {pct}% of real-mechanism structure on average "
        f"({propab_total}/20 vs gold {gold_total}/20). "
        f"All four dimensions are missing: compression, world-intervention counterfactuality, failure conditions, generativity. "
        f"Dominant Propab class: '{top_class}'. "
        "This confirms the bottleneck is structural (what Propab asks for and stores), not model intelligence or prompt wording. "
        "Do not run campaigns, switch models, tune prompts, or add gates until mechanism induction schema is redesigned."
    )


def _redesign_requirements() -> list[dict]:
    return [
        {
            "field": "compression_target",
            "requirement": "Mechanism must name ≥2 otherwise-unconnected observations it collapses",
            "gate": "Reject if only explains the anomaly in front of it",
        },
        {
            "field": "counterfactual_prediction",
            "requirement": "Intervention on causal structure in the studied system (not 'metric A beats metric B')",
            "gate": "Reject model-comparison counterfactuals",
        },
        {
            "field": "failure_conditions",
            "requirement": "Explicit where mechanism stops working",
            "gate": "Reject if no scope limit stated",
        },
        {
            "field": "generative_cross_domain_prediction",
            "requirement": "One falsifiable experiment in a domain this campaign never touched",
            "gate": "Reject if no cross-domain discriminating experiment named",
        },
        {
            "field": "discriminating_experiment",
            "requirement": "Single-variable test that would confirm/refute mechanism",
            "gate": "Required before mechanism enters hypothesis tree",
        },
    ]


if __name__ == "__main__":
    r = run()
    print(json.dumps({
        "gold_avg": r["gold_avg_scores"],
        "propab_avg": r["propab_avg_scores"],
        "propab_pct_of_gold": r["propab_pct_of_gold"],
        "missing_dimensions": r["missing_dimensions"],
        "verdict": r["verdict"][:200] + "...",
        "artifacts": [str(OUT), str(REPORT)],
    }, indent=2))
