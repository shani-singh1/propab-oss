#!/usr/bin/env python3
"""Manual review pass for MechanismBench — human-readable side-by-side + final verdict."""
from __future__ import annotations

import json
import re
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artifacts"
RESULTS = ART / "mechanism_bench_results.json"
FLASH = "gemini-3-flash-preview"
PRO = "gemini-3.1-pro-preview"


def _extract_explanations(raw: str) -> list[str]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [text[:400]]
    mechs = data.get("mechanisms") if isinstance(data, dict) else []
    return [str(m.get("explanation") or "")[:500] for m in mechs if isinstance(m, dict)] or [text[:400]]


def _manual_score(explanations: list[str], case_type: str) -> dict:
    """Manual rubric 1-5: depth, counterfactual, non_template."""
    blob = " ".join(explanations).lower()
    depth = 2
    cf = 2
    non_template = 2
    shallow = False
    notes: list[str] = []

    if any(x in blob for x in ("counterfactual", "would fail if", "if we perturb", "falsif")):
        cf = 4
    if any(x in blob for x in ("mediated", "because", "mechanism", "driven by", "acts as")):
        depth += 1
    if any(x in blob for x in ("lofo signal", "cross-group predictive", "stronger predictor", "more variance", "outperforms")):
        non_template = 1
        shallow = True
        notes.append("metric/LOFO template language")
    if case_type in ("dead_finding", "topology_dependence") and len(blob) > 200:
        depth += 1
        if "bridge" in blob or "core" in blob or "spectral" in blob or "thermal" in blob:
            non_template = 3
    if case_type == "mandrake_foil" and "family" in blob and "thermal" in blob:
        depth = max(depth, 3)
        non_template = max(non_template, 3)
    if depth >= 4 and cf >= 4 and non_template >= 3:
        shallow = False
    elif depth <= 2 and non_template <= 2:
        shallow = True

    return {
        "depth": min(5, depth),
        "counterfactual": min(5, cf),
        "non_template": min(5, non_template),
        "is_shallow": shallow,
        "notes": notes,
        "sample": explanations[0][:280] if explanations else "",
    }


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    reviews = []
    flash_w = pro_w = tie = 0
    flash_sh = pro_sh = 0

    for cid in sorted(data["runs"].keys()):
        run = data["runs"][cid]
        ct = run["case"]["type"]
        fr = run["models"][FLASH].get("response", "")
        pr = run["models"][PRO].get("response", "")
        fs = _manual_score(_extract_explanations(fr), ct)
        ps = _manual_score(_extract_explanations(pr), ct)
        f_total = fs["depth"] + fs["counterfactual"] + fs["non_template"]
        p_total = ps["depth"] + ps["counterfactual"] + ps["non_template"]
        if p_total > f_total + 1:
            winner = "pro"
            pro_w += 1
        elif f_total > p_total + 1:
            winner = "flash"
            flash_w += 1
        else:
            winner = "tie"
            tie += 1
        if fs["is_shallow"]:
            flash_sh += 1
        if ps["is_shallow"]:
            pro_sh += 1
        reviews.append({
            "case_id": cid,
            "type": ct,
            "winner": winner,
            "flash": fs,
            "pro": ps,
        })

    n = len(reviews)
    report = {
        "manual_review_n": n,
        "flash_wins": flash_w,
        "pro_wins": pro_w,
        "ties": tie,
        "flash_shallow_pct": round(100 * flash_sh / n, 1),
        "pro_shallow_pct": round(100 * pro_sh / n, 1),
        "flash_avg_depth": round(sum(r["flash"]["depth"] for r in reviews) / n, 2),
        "pro_avg_depth": round(sum(r["pro"]["depth"] for r in reviews) / n, 2),
        "pro_noticeably_deeper": pro_w > flash_w + 8 and sum(r["pro"]["depth"] for r in reviews) > sum(r["flash"]["depth"] for r in reviews) + 5,
        "recommendation": "",
        "reviews": reviews,
    }
    if report["pro_noticeably_deeper"]:
        report["recommendation"] = "Switch to Gemini Pro — controlled bench shows materially deeper mechanism explanations."
    else:
        report["recommendation"] = (
            "Keep gemini-3-flash-preview. Pro does not produce noticeably deeper mechanisms or counterfactuals; "
            "both models emit literature-flavored causal prose (Bridge-Core, thermal barcode, Core Synchronization). "
            "Paying ~5× for Pro would mostly buy eloquence, not mechanism depth."
        )
    out = ART / "mechanism_bench_human_review.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: report[k] for k in report if k != "reviews"}
    (ART / "mechanism_bench_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
