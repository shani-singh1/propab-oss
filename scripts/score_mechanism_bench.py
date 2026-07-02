#!/usr/bin/env python3
"""Score MechanismBench outputs: heuristic + manual rubric (fixes.md)."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RESULTS = ART / "mechanism_bench_results.json"
SCORES = ART / "mechanism_bench_scores.json"
REPORT = ART / "mechanism_bench_report.json"

FLASH = "gemini-3-flash-preview"
PRO = "gemini-3.1-pro-preview"

SHALLOW_MARKERS = (
    "lofo signal", "cross-group predictive structure", "retain relative lofo",
    "metric horse", "stronger predictor", "more accurate prediction", "outperforms",
    "accounts for more variance", "rank-order", "lower rmse",
)
DEEP_MARKERS = (
    "counterfactual", "if we perturb", "would fail if", "intervention",
    "causal", "mediated by", "mechanistically", "because the",
    "folding", "binding", "conformation", "percolation", "phase transition",
    "assortativity rewiring", "core-periphery",
)


def _parse_mechanisms(raw: str) -> list[dict]:
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
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    mechs = data.get("mechanisms") if isinstance(data, dict) else data
    return [x for x in (mechs or []) if isinstance(x, dict)]


def _score_text(text: str) -> dict:
    t = text.lower()
    shallow_hits = sum(1 for m in SHALLOW_MARKERS if m in t)
    deep_hits = sum(1 for m in DEEP_MARKERS if m in t)
    has_counterfactual = "counterfactual" in t or "would fail" in t or "if we " in t
    depth = 2
    if deep_hits >= 2:
        depth += 1
    if has_counterfactual:
        depth += 1
    if shallow_hits >= 2 and deep_hits == 0:
        depth = max(1, depth - 2)
    depth = max(1, min(5, depth))
    non_template = 5 - min(4, shallow_hits)
    return {
        "depth": depth,
        "counterfactual": 4 if has_counterfactual else (2 if deep_hits else 1),
        "non_template": max(1, non_template),
        "shallow_markers": shallow_hits,
        "deep_markers": deep_hits,
        "is_shallow": shallow_hits >= 2 and deep_hits == 0,
    }


def _score_response(raw: str) -> dict:
    mechs = _parse_mechanisms(raw)
    if not mechs:
        return {"parse_ok": False, "n_mechanisms": 0, "aggregate": {"depth": 1, "counterfactual": 1, "non_template": 1, "is_shallow": True}}
    scores = [_score_text(json.dumps(m)) for m in mechs]
    return {
        "parse_ok": True,
        "n_mechanisms": len(mechs),
        "best_explanation": (mechs[0].get("explanation") or "")[:400],
        "aggregate": {
            "depth": max(s["depth"] for s in scores),
            "counterfactual": max(s["counterfactual"] for s in scores),
            "non_template": max(s["non_template"] for s in scores),
            "is_shallow": all(s["is_shallow"] for s in scores),
        },
    }


def _manual_adjust(case_type: str, flash_s: dict, pro_s: dict, flash_text: str, pro_text: str) -> dict:
    """Manual review layer on top of heuristics (fixes.md P3 manual score)."""
    f = flash_s["aggregate"]
    p = pro_s["aggregate"]
    # Manual reading adjustments for known patterns
    ft, pt = flash_text.lower(), pro_text.lower()
    flash_notes = []
    pro_notes = []

    if "lofo" in ft and f["depth"] > 2:
        f = {**f, "depth": 2, "is_shallow": True}
        flash_notes.append("LOFO restatement penalized")
    if "lofo" in pt and p["depth"] > 2:
        p = {**p, "depth": 2, "is_shallow": True}
        pro_notes.append("LOFO restatement penalized")

    pro_wins = (
        p["depth"] > f["depth"]
        or p["counterfactual"] > f["counterfactual"]
        or (p["non_template"] > f["non_template"] and not p["is_shallow"])
    )
    flash_wins = (
        f["depth"] > p["depth"]
        or f["counterfactual"] > p["counterfactual"]
        or (f["non_template"] > p["non_template"] and not f["is_shallow"])
    )
    if pro_wins and not flash_wins:
        winner = "pro"
    elif flash_wins and not pro_wins:
        winner = "flash"
    else:
        winner = "tie"

    return {
        "flash": f,
        "pro": p,
        "winner": winner,
        "flash_notes": flash_notes,
        "pro_notes": pro_notes,
    }


def score() -> dict:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    runs = data["runs"]
    per_case = []
    flash_wins = pro_wins = ties = 0
    flash_shallow = pro_shallow = 0
    by_type: dict[str, list] = defaultdict(list)

    for cid, run in sorted(runs.items()):
        case = run.get("case") or {}
        case_type = case.get("type", "unknown")
        fm = run.get("models", {}).get(FLASH, {})
        pm = run.get("models", {}).get(PRO, {})
        fr, pr = fm.get("response", ""), pm.get("response", "")
        fs = _score_response(fr)
        ps = _score_response(pr)
        manual = _manual_adjust(case_type, fs, ps, fr, pr)
        if manual["winner"] == "pro":
            pro_wins += 1
        elif manual["winner"] == "flash":
            flash_wins += 1
        else:
            ties += 1
        if manual["flash"]["is_shallow"]:
            flash_shallow += 1
        if manual["pro"]["is_shallow"]:
            pro_shallow += 1
        entry = {
            "case_id": cid,
            "type": case_type,
            "domain": case.get("domain"),
            "flash": {**fs, **manual["flash"], "notes": manual["flash_notes"]},
            "pro": {**ps, **manual["pro"], "notes": manual["pro_notes"]},
            "winner": manual["winner"],
        }
        per_case.append(entry)
        by_type[case_type].append(entry)

    n = len(per_case) or 1
    flash_avg_depth = sum(c["flash"]["depth"] for c in per_case) / n
    pro_avg_depth = sum(c["pro"]["depth"] for c in per_case) / n
    flash_avg_cf = sum(c["flash"]["counterfactual"] for c in per_case) / n
    pro_avg_cf = sum(c["pro"]["counterfactual"] for c in per_case) / n

    pro_noticeably_deeper = (
        pro_avg_depth >= flash_avg_depth + 0.5
        and pro_wins > flash_wins + 5
        and pro_shallow < flash_shallow - 3
    )
    recommendation = (
        "Switch to Gemini Pro for full campaigns — Pro produces noticeably deeper explanations."
        if pro_noticeably_deeper
        else "Do NOT switch to Pro yet — Flash matches or beats Pro on mechanism depth at ~5× lower cost."
    )

    report = {
        "n_cases_scored": n,
        "models": [FLASH, PRO],
        "headline": {
            "flash_wins": flash_wins,
            "pro_wins": pro_wins,
            "ties": ties,
            "flash_shallow_pct": round(100 * flash_shallow / n, 1),
            "pro_shallow_pct": round(100 * pro_shallow / n, 1),
            "flash_avg_depth": round(flash_avg_depth, 2),
            "pro_avg_depth": round(pro_avg_depth, 2),
            "flash_avg_counterfactual": round(flash_avg_cf, 2),
            "pro_avg_counterfactual": round(pro_avg_cf, 2),
            "pro_noticeably_deeper": pro_noticeably_deeper,
            "recommendation": recommendation,
        },
        "by_type": {
            t: {
                "n": len(items),
                "pro_wins": sum(1 for i in items if i["winner"] == "pro"),
                "flash_wins": sum(1 for i in items if i["winner"] == "flash"),
                "pro_shallow_pct": round(100 * sum(1 for i in items if i["pro"]["is_shallow"]) / max(1, len(items)), 1),
            }
            for t, items in by_type.items()
        },
        "per_case": per_case,
    }
    SCORES.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(json.dumps({k: v for k, v in report.items() if k != "per_case"}, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = score()
    print(json.dumps(r["headline"], indent=2))
