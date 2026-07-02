#!/usr/bin/env python3
"""Run MechanismBench: same prompts on Flash vs Pro (fixes.md model comparison)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
BENCH_PATH = ART / "mechanism_bench.json"
OUT_PATH = ART / "mechanism_bench_results.json"

FLASH = "gemini-3-flash-preview"
PRO = "gemini-3.1-pro-preview"


def _gemini_call(prompt: str, model: str, api_key: str, *, timeout: float = 180.0) -> dict:
    mid = model.strip()
    if mid.startswith("models/"):
        mid = mid[len("models/") :]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{mid}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }
    started = time.perf_counter()
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=60.0)) as client:
        resp = client.post(url, params={"key": api_key}, json=payload)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if resp.status_code != 200:
        return {"error": resp.text[:2000], "status_code": resp.status_code, "elapsed_ms": elapsed_ms}
    body = resp.json()
    texts: list[str] = []
    for c in body.get("candidates") or []:
        for part in (c.get("content") or {}).get("parts") or []:
            if isinstance(part, dict) and part.get("text"):
                texts.append(part["text"])
    return {
        "response": "".join(texts),
        "elapsed_ms": elapsed_ms,
        "status_code": resp.status_code,
    }


def run(*, models: list[str] | None = None, limit: int | None = None, resume: bool = True) -> dict:
    api_key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        # load from .env without modifying it
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GOOGLE_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set")

    bench = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    cases = bench["cases"]
    if limit:
        cases = cases[:limit]

    models = models or [FLASH, PRO]
    existing: dict = {}
    if resume and OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))

    results = existing.get("runs") or {}
    for case in cases:
        cid = case["id"]
        if cid not in results:
            results[cid] = {"case": {k: case[k] for k in ("id", "type", "domain", "source_id")}, "models": {}}
        for model in models:
            if model in results[cid]["models"] and not results[cid]["models"][model].get("error"):
                continue
            print(f"  {cid} @ {model}...", flush=True)
            try:
                out = _gemini_call(case["prompt"], model, api_key)
                results[cid]["models"][model] = out
            except Exception as exc:  # noqa: BLE001
                results[cid]["models"][model] = {"error": str(exc)}
            time.sleep(0.5)  # gentle rate limit
            partial = {
                "bench_version": bench.get("version"),
                "models": models,
                "n_cases": len(cases),
                "runs": results,
            }
            OUT_PATH.write_text(json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")

    final = {
        "bench_version": bench.get("version"),
        "models": models,
        "n_cases": len(cases),
        "composition": bench.get("composition"),
        "runs": results,
    }
    OUT_PATH.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    return final


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    r = run(limit=lim)
    ok = sum(
        1 for run in r["runs"].values()
        for m in r["models"]
        if m in run.get("models", {}) and not run["models"][m].get("error")
    )
    print(json.dumps({"out": str(OUT_PATH), "successful_calls": ok, "expected": r["n_cases"] * len(r["models"])}, indent=2))
