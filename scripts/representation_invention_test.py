#!/usr/bin/env python3
"""
Representation invention test (fixes.md).

Ask LLM for derived features on Mandrake; test with LOFO + family-shuffle null.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from demo.mandrake.domain import load_frame
from propab.domain_adapters.mandrake_adapter import (
    _leave_one_family_out_r2,
    _make_model,
    _within_family_r2,
)

ART = ROOT / "artifacts"
OUT = ART / "representation_invention_test.json"

# Import permutation helpers from sibling script
sys.path.insert(0, str(ROOT / "scripts"))

from lofo_family_permutation_test import family_label_permutation_null, summarize_null  # noqa: E402

PROMPT = """You are a feature engineer for a biophysical dataset predicting RT enzyme activity (pe_efficiency_pct).

Context:
- 57 protein sequences across 7 evolutionary families (rt_family).
- {n_features} numeric input columns are available (listed below).
- Leave-one-family-out (LOFO) ridge regression shows that NO raw column or simple 3-feature subset
  survives a family-label shuffle permutation null at p<0.05 — signals are mostly family-specific leakage.

Task: Propose exactly 5 NEW derived features — transformations, ratios, or interactions of EXISTING columns —
that are NOT already in the table as-is. Each must be computable from column names only using +, -, *, /, parentheses.

Do NOT restate raw columns. Do NOT propose features that are trivial renames.

Return JSON ONLY:
{{
  "derived_features": [
    {{
      "name": "snake_case_name",
      "formula": "t75_raw - t70_raw",
      "rationale": "why this might capture cross-family biology not family identity"
    }}
  ]
}}

Available columns:
{column_list}

Best raw LOFO results (for calibration — all failed permutation null):
{baseline_summary}
"""


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    skip = {"rt_name", "rt_family", "active", "pe_efficiency_pct"}
    return [
        c for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]


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


def _gemini(prompt: str, api_key: str, model: str = "gemini-3-flash-preview") -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3}}
    with httpx.Client(timeout=180.0) as client:
        r = client.post(url, params={"key": api_key}, json=payload)
    r.raise_for_status()
    parts = (((r.json().get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


def _eval_formula(df: pd.DataFrame, formula: str, allowed: set[str]) -> pd.Series | None:
    f = formula.strip()
    if not f:
        return None
    # reject dangerous tokens
    if re.search(r"[_\w]+\(", f) and not re.match(r"^[a-zA-Z0-9_\s+\-*/().]+$", f):
        return None
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", f))
    if not tokens <= allowed:
        return None
    try:
        s = df.eval(f)
        if isinstance(s, pd.Series):
            return pd.to_numeric(s, errors="coerce")
    except Exception:  # noqa: BLE001
        return None
    return None


def _is_trivial(name: str, formula: str, raw_cols: set[str]) -> bool:
    f = formula.strip()
    if f in raw_cols or name in raw_cols:
        return True
    if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", f) and f in raw_cols:
        return True
    return False


def _lofo_audit(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_perm: int = 500,
) -> dict:
    sub = df[["pe_efficiency_pct", "rt_family", *feature_cols]].dropna()
    if len(sub) < 10:
        return {"error": "too_few_rows", "n": len(sub)}
    X = sub[feature_cols].to_numpy(dtype=float)
    y = sub["pe_efficiency_pct"].to_numpy(float)
    fam = sub["rt_family"].astype(str).to_numpy()
    model = _make_model("ridge")
    lofo = _leave_one_family_out_r2(X, y, fam, model)
    within = _within_family_r2(X, y, fam, model)
    obs, null = family_label_permutation_null(X, y, fam, model, n_perm=n_perm, seed=42)
    summary = summarize_null(obs, null)
    return {
        "features": feature_cols,
        "n_samples": len(sub),
        "lofo_r2": round(lofo, 4),
        "within_r2": round(within, 4),
        "lofo_gap": round(within - lofo, 4),
        **summary,
    }


def _baseline_scan(df: pd.DataFrame, cols: list[str], *, n_perm: int = 200, top_k: int = 8) -> dict:
    sub = df[["pe_efficiency_pct", "rt_family"] + cols].dropna()
    y = sub["pe_efficiency_pct"].to_numpy(float)
    fam = sub["rt_family"].astype(str).to_numpy()
    model = _make_model("ridge")
    singles = []
    for c in cols:
        X = sub[[c]].to_numpy(float)
        lofo = _leave_one_family_out_r2(X, y, fam, model)
        singles.append({"feature": c, "lofo_r2": round(lofo, 4)})
    singles.sort(key=lambda x: -x["lofo_r2"])

    # best triple from top thermal/geometry mix
    top = [s["feature"] for s in singles[:12]]
    triples = []
    for i, a in enumerate(top[:6]):
        for b in top[i + 1 : 8]:
            for c in top[i + 2 : 9]:
                if len({a, b, c}) < 3:
                    continue
                feat = [a, b, c]
                X = sub[feat].to_numpy(float)
                lofo = _leave_one_family_out_r2(X, y, fam, model)
                triples.append({"features": feat, "lofo_r2": round(lofo, 4)})
    triples.sort(key=lambda x: -x["lofo_r2"])

    best_triple = triples[0] if triples else None
    if best_triple:
        obs, null = family_label_permutation_null(
            sub[best_triple["features"]].to_numpy(float), y, fam, model, n_perm=n_perm,
        )
        best_triple["permutation"] = summarize_null(obs, null)

    return {
        "best_singles": singles[:top_k],
        "best_triple": best_triple,
        "n_singles_survive_p95": sum(
            1 for s in singles
            if _quick_survives(sub, [s["feature"]], fam, y, model, n_perm=100)
        ),
    }


def _quick_survives(sub, feats, fam, y, model, n_perm=100) -> bool:
    X = sub[feats].to_numpy(float)
    obs, null = family_label_permutation_null(X, y, fam, model, n_perm=n_perm, seed=42)
    return obs > float(np.percentile(null, 95))


def _parse_derived(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    return [d for d in (data.get("derived_features") or []) if isinstance(d, dict)]


def run(*, n_perm: int = 500) -> dict:
    df = load_frame()
    cols = _numeric_columns(df)
    raw_set = set(cols)

    baseline = _baseline_scan(df, cols, n_perm=200)
    baseline_txt = json.dumps(
        {"best_singles": baseline["best_singles"][:5], "best_triple": baseline.get("best_triple")},
        indent=2,
    )[:3000]

    api_key = _load_api_key()
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY required")

    prompt = PROMPT.format(
        n_features=len(cols),
        column_list=", ".join(cols),
        baseline_summary=baseline_txt,
    )
    raw_llm = _gemini(prompt, api_key)
    proposals = _parse_derived(raw_llm)

    evaluated = []
    for p in proposals[:5]:
        name = str(p.get("name") or "derived")
        formula = str(p.get("formula") or "")
        trivial = _is_trivial(name, formula, raw_set)
        series = None if trivial else _eval_formula(df, formula, raw_set)
        if series is None and not trivial:
            evaluated.append({
                "name": name,
                "formula": formula,
                "rationale": p.get("rationale"),
                "status": "eval_failed",
                "trivial": trivial,
            })
            continue
        if trivial:
            evaluated.append({
                "name": name,
                "formula": formula,
                "rationale": p.get("rationale"),
                "status": "trivial_restatement",
                "trivial": True,
            })
            continue

        col = f"__derived_{name}"
        work = df.copy()
        work[col] = series
        audit = _lofo_audit(work, [col], n_perm=n_perm)
        evaluated.append({
            "name": name,
            "formula": formula,
            "rationale": p.get("rationale"),
            "status": "evaluated",
            "trivial": False,
            "audit": audit,
            "survives_permutation_null": audit.get("outside_noise_band_p95", False),
        })

    # Also test each derived feature combined with nothing else - and best combo of 3 derived
    derived_cols = [f"__derived_{e['name']}" for e in evaluated if e.get("status") == "evaluated"]
    if len(derived_cols) >= 2:
        work = df.copy()
        for e in evaluated:
            if e.get("status") == "evaluated":
                work[e["name"]] = _eval_formula(df, e["formula"], raw_set)
        combo_cols = [e["name"] for e in evaluated if e.get("status") == "evaluated"][:3]
        if combo_cols:
            combo_audit = _lofo_audit(work, combo_cols, n_perm=n_perm)
        else:
            combo_audit = None
    else:
        combo_audit = None

    n_eval = sum(1 for e in evaluated if e.get("status") == "evaluated")
    n_survive = sum(1 for e in evaluated if e.get("survives_permutation_null"))
    n_trivial = sum(1 for e in evaluated if e.get("trivial") or e.get("status") == "trivial_restatement")

    best_derived_lofo = max(
        (e["audit"]["lofo_r2"] for e in evaluated if e.get("audit")),
        default=None,
    )
    best_raw_lofo = baseline["best_singles"][0]["lofo_r2"] if baseline["best_singles"] else None

    if n_survive > 0:
        verdict = (
            f"Partial success: {n_survive}/{n_eval} derived features survive family-shuffle null — "
            "representation invention may be worth scaling up."
        )
    elif n_eval > 0 and best_derived_lofo is not None and best_raw_lofo is not None and best_derived_lofo > best_raw_lofo + 0.05:
        verdict = (
            "No permutation survival, but derived features beat best raw LOFO — weak signal, not yet cross-family."
        )
    else:
        verdict = (
            "Representation invention failed at smallest scale: derived features do not beat raw columns "
            "on LOFO/permutation null; mostly trivial or family-leakage-shaped. "
            "2-3 weeks of bigger infrastructure unlikely to help without changing what is asked."
        )

    report = {
        "test": "representation_invention_mandrake",
        "n_samples": len(df),
        "n_families": df["rt_family"].nunique(),
        "n_raw_features": len(cols),
        "baseline_raw": baseline,
        "llm_model": "gemini-3-flash-preview",
        "n_proposed": len(proposals),
        "evaluated": evaluated,
        "derived_combo_audit": combo_audit,
        "summary": {
            "n_evaluated": n_eval,
            "n_trivial": n_trivial,
            "n_survive_permutation_p95": n_survive,
            "best_raw_lofo_r2": best_raw_lofo,
            "best_derived_lofo_r2": best_derived_lofo,
        },
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = run()
    print(json.dumps({"summary": r["summary"], "verdict": r["verdict"], "out": str(OUT)}, indent=2))
    for e in r["evaluated"]:
        if e.get("audit"):
            print(f"  {e['name']}: lofo={e['audit']['lofo_r2']} survive={e.get('survives_permutation_null')}")
