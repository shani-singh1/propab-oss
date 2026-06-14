#!/usr/bin/env python3
"""Level 2 quick check: entropy dynamics residuals across runs (fixes.md P1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENTROPY_KEYS = (
    "start_H",
    "growth_rate",
    "saturation_H",
    "cross_H_1_5_at_tested",
    "cross_H_2_0_at_tested",
    "theme_entropy",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        default=str(ROOT / "artifacts" / "policy_residuals.json"),
    )
    args = parser.parse_args()
    data = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    rows = data.get("residuals") or []
    if not rows:
        print("No residuals found.", file=sys.stderr)
        return 1

    per_run = []
    for r in rows:
        res = r.get("residual") or {}
        pred = r.get("prediction") or {}
        act = r.get("actual") or {}
        mode = "dynamics" if any(k in pred for k in ENTROPY_KEYS[:5]) else "legacy"
        key = "saturation_H" if mode == "dynamics" and "saturation_H" in res else "theme_entropy"
        v = res.get(key)
        if v is None and mode == "legacy":
            v = res.get("theme_entropy")
        if v is not None:
            per_run.append({
                "campaign_id": r.get("campaign_id", "")[:8],
                "eval_mode": mode,
                "primary_residual_key": key,
                "residual": float(v),
                "prediction": pred,
                "observed": act,
                "policy_id": r.get("policy_id"),
            })

    if len(per_run) < 2:
        print(json.dumps({"error": "need >=2 runs", "n": len(per_run)}, indent=2))
        return 1

    signs = [1 if x["residual"] > 0 else (-1 if x["residual"] < 0 else 0) for x in per_run]
    nonzero = [s for s in signs if s != 0]
    consistent = len(set(nonzero)) <= 1 if nonzero else False

    report = {
        "n_runs": len(per_run),
        "eval_modes": [x["eval_mode"] for x in per_run],
        "primary_residuals": [x["residual"] for x in per_run],
        "signs": signs,
        "directionally_consistent": consistent,
        "interpretation": (
            "fixable analyst bias on entropy dynamics"
            if consistent and nonzero
            else "deeper understanding problem or noise-dominated (mixed/random signs)"
        ),
        "per_run": per_run,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
