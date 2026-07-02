#!/usr/bin/env python3
"""Summarize scored AstaBench Propab run into artifacts/astabench_propab_results.json."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASTA = ROOT / "asta-bench"


def _hms_from_eval_log(log_dir: Path) -> tuple[dict[str, float], float | None]:
    eval_files = sorted(log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not eval_files:
        return {}, None
    sys.path.insert(0, str(ASTA))
    from inspect_ai.log import read_eval_log

    elog = read_eval_log(str(eval_files[0]))
    per: dict[str, float] = {}
    for s in elog.samples:
        hms = None
        if s.scores:
            for sc in s.scores.values():
                hms = float(sc.value) if sc.value is not None else None
        if hms is not None:
            per[str(s.id)] = hms
    mean = None
    if elog.results and elog.results.scores:
        for sc in elog.results.scores:
            m = (sc.metrics or {}).get("mean")
            if m is not None:
                mean = float(m.value)
                break
    return per, mean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Inspect eval log directory")
    parser.add_argument("--campaign-log", default=str(ROOT / "data" / "astabench" / "_campaign_log.jsonl"))
    parser.add_argument("--option-b", action="store_true", help="Option B fair eval artifact")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"Missing log dir: {log_dir}", file=sys.stderr)
        return 1

    score_out = ""
    env = os.environ.copy()
    env.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    try:
        score_out = subprocess.check_output(
            [sys.executable, str(ROOT / "scripts" / "run_astabench_discovery.py"), "--score-only", str(log_dir)],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        score_out = exc.output or str(exc)

    per_hms, mean_hms = _hms_from_eval_log(log_dir)

    campaigns = []
    solutions_path = log_dir / "option_b_solutions.json"
    if solutions_path.is_file():
        sol = json.loads(solutions_path.read_text(encoding="utf-8"))
        campaigns = list((sol.get("samples") or {}).values())
    else:
        camp_path = Path(args.campaign_log)
        if camp_path.is_file():
            for line in camp_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    campaigns.append(json.loads(line))

    manifest_path = log_dir / "option_b_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mode = "option_b" if args.option_b or manifest.get("mode") == "option_b" else "option_a"
    interpretation = (
        "Option B (decoupled solve-then-score). Campaigns ran outside Inspect with extended "
        "budget; HMS scores reflect hypothesis match under Gemini scorer (not GPT-4o baseline)."
        if mode == "option_b"
        else (
            "Option A (20 min/campaign). Low scores are inconclusive on architecture without "
            "Option B decoupled runs — see fixes.md."
        )
    )

    artifact = {
        "mode": mode,
        "log_dir": str(log_dir.resolve()),
        "hms_mean": mean_hms,
        "per_sample_hms": per_hms,
        "n_samples": len(per_hms),
        "score_output": score_out,
        "manifest": manifest,
        "campaigns": campaigns,
        "interpretation": interpretation,
    }
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / ("astabench_propab_option_b_results.json" if mode == "option_b" else "astabench_propab_results.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    if mean_hms is not None:
        print(f"mean HMS: {mean_hms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
