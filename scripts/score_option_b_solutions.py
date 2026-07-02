#!/usr/bin/env python3
"""Score Option B cached solutions without Inspect/Docker (HMS via DiscoveryBench scorer)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASTA = ROOT / "asta-bench"


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip()


async def _score_all(solutions_path: Path, split: str) -> dict:
    sys.path.insert(0, str(ASTA))
    py_path = os.pathsep.join([str(ROOT), str(ROOT / "packages" / "propab-core")])
    os.environ["PYTHONPATH"] = py_path + (os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")

    from astabench.evals.discoverybench.eval_utils import run_eval_gold_vs_gen_NL_hypo_workflow
    from astabench.evals.discoverybench.lm_utils import discoverybench_scorer_model
    from astabench.evals.discoverybench.task import discoverybench_validation, discoverybench_test

    data = json.loads(solutions_path.read_text(encoding="utf-8"))
    samples_map = data.get("samples") or data
    task = discoverybench_validation() if split == "validation" else discoverybench_test()
    scorer = discoverybench_scorer_model()

    by_id = {str(s.id): s for s in task.dataset}
    per_sample: dict[str, dict] = {}
    hms_values: list[float] = []

    for sid, rec in samples_map.items():
        sample = by_id.get(str(sid))
        if sample is None:
            continue
        answer = rec.get("answer") or {}
        gen_hypo = str(answer.get("hypothesis") or "")
        gen_workflow = str(answer.get("workflow") or "")
        target = sample.target
        gold_hypo = target[0] if isinstance(target, list) else str(target)
        gold_workflow = target[1] if isinstance(target, list) and len(target) > 1 else ""

        eval_result = await run_eval_gold_vs_gen_NL_hypo_workflow(
            query=str((sample.metadata or {}).get("query") or ""),
            gold_hypo=str(gold_hypo),
            gold_workflow=str(gold_workflow),
            gen_hypo=gen_hypo,
            gen_workflow=gen_workflow,
            dataset_meta=(sample.metadata or {}).get("metadata") or {},
            llm_used=scorer,
            use_column_metadata=True,
        )
        hms = float(eval_result.get("HMS", 0.0))
        hms_values.append(hms)
        per_sample[str(sid)] = {
            "hms": hms,
            "context_score": eval_result.get("context_score"),
            "eval_rec": eval_result,
            "extract_audit": rec.get("extract_audit"),
            "fabrication_audit": rec.get("fabrication_audit"),
        }
        print(f"{sid}: HMS={hms}", flush=True)

    mean_hms = sum(hms_values) / len(hms_values) if hms_values else 0.0
    return {
        "mode": "option_b",
        "split": split,
        "scorer_model": scorer,
        "n_samples": len(hms_values),
        "hms_mean": mean_hms,
        "per_sample": per_sample,
        "solutions_path": str(solutions_path.resolve()),
        "budget_hours": (data.get("meta") or {}).get("budget_hours"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "solutions",
        nargs="?",
        default=str(ASTA / "logs" / "astabench-option-b-validation" / "option_b_solutions.json"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "astabench_propab_option_b_results.json"))
    args = parser.parse_args()

    _load_dotenv()
    path = Path(args.solutions)
    if not path.is_file():
        alt = ROOT / "logs" / "astabench-option-b-validation" / "option_b_solutions.json"
        if alt.is_file():
            path = alt
        else:
            print(f"Missing {path}", file=sys.stderr)
            return 1

    result = asyncio.run(_score_all(path, args.split))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nmean HMS: {result['hms_mean']:.4f} (n={result['n_samples']})")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
