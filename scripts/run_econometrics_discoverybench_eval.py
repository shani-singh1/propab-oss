#!/usr/bin/env python3
"""T3-002 exit check: DiscoveryBench econometrics samples with econometrics domain profile."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASTA = ROOT / "asta-bench"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ASTA))


def _ensure_inspect_env() -> None:
    """Re-exec under asta-bench uv env when inspect_ai is not on default python."""
    try:
        import inspect_ai  # noqa: F401
    except ImportError:
        import subprocess

        env = os.environ.copy()
        py_path = os.pathsep.join([str(ROOT), str(ROOT / "packages" / "propab-core"), str(ASTA)])
        env["PYTHONPATH"] = py_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        cmd = ["uv", "run", "python", str(Path(__file__).resolve()), *sys.argv[1:]]
        raise SystemExit(subprocess.call(cmd, cwd=str(ASTA), env=env))


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


def _is_econometrics_sample(sample_id: str) -> bool:
    sid = sample_id.lower()
    return sid.startswith("nls_") or sid.startswith("immigration_offshoring")


async def _score_records(records: dict[str, dict], split: str) -> dict:
    asta = ROOT / "asta-bench"
    sys.path.insert(0, str(asta))
    py_path = os.pathsep.join([str(ROOT), str(ROOT / "packages" / "propab-core")])
    os.environ["PYTHONPATH"] = py_path + (
        os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else ""
    )

    from astabench.evals.discoverybench.eval_utils import run_eval_gold_vs_gen_NL_hypo_workflow
    from astabench.evals.discoverybench.lm_utils import discoverybench_scorer_model
    from astabench.evals.discoverybench.task import discoverybench_validation, discoverybench_test

    task = discoverybench_validation() if split == "validation" else discoverybench_test()
    scorer = discoverybench_scorer_model()
    by_id = {str(s.id): s for s in task.dataset}

    per_sample: dict[str, dict] = {}
    hms_values: list[float] = []
    for sid, rec in records.items():
        sample = by_id.get(str(sid))
        if sample is None:
            continue
        answer = rec.get("answer") or {}
        target = sample.target
        gold_hypo = target[0] if isinstance(target, list) else str(target)
        gold_workflow = target[1] if isinstance(target, list) and len(target) > 1 else ""
        eval_result = await run_eval_gold_vs_gen_NL_hypo_workflow(
            query=str((sample.metadata or {}).get("query") or ""),
            gold_hypo=str(gold_hypo),
            gold_workflow=str(gold_workflow),
            gen_hypo=str(answer.get("hypothesis") or ""),
            gen_workflow=str(answer.get("workflow") or ""),
            dataset_meta=(sample.metadata or {}).get("metadata") or {},
            llm_used=scorer,
            use_column_metadata=True,
        )
        hms = float(eval_result.get("HMS", 0.0))
        hms_values.append(hms)
        per_sample[str(sid)] = {
            "hms": hms,
            "context_score": eval_result.get("context_score"),
            "query": (sample.metadata or {}).get("query"),
            "gen_hypothesis": answer.get("hypothesis"),
            "gold_hypothesis": gold_hypo,
            "extract_audit": rec.get("extract_audit"),
            "campaign_status": rec.get("campaign_status"),
            "stop_reason": rec.get("stop_reason"),
        }
        print(f"{sid}: HMS={hms:.4f}", flush=True)

    mean_hms = sum(hms_values) / len(hms_values) if hms_values else 0.0
    return {
        "domain_profile": "econometrics",
        "split": split,
        "scorer_model": scorer,
        "n_samples": len(hms_values),
        "hms_mean": mean_hms,
        "hms_max": max(hms_values) if hms_values else 0.0,
        "per_sample": per_sample,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    _ensure_inspect_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=3, help="Max econometrics samples to run")
    parser.add_argument("--budget-hours", type=float, default=0.5)
    parser.add_argument("--api", default=os.environ.get("PROPAB_API_URL", "http://localhost:8000"))
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "econometrics_discoverybench_hms.json"),
    )
    parser.add_argument("--score-only", default=None, help="JSON file with prior solve records")
    args = parser.parse_args()

    _load_dotenv()

    if args.score_only:
        data = json.loads(Path(args.score_only).read_text(encoding="utf-8"))
        records = data.get("samples") or data
        result = asyncio.run(_score_records(records, args.split))
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nmean HMS: {result['hms_mean']:.4f} (n={result['n_samples']})")
        print(f"Wrote {args.out}")
        return 0

    from integrations.astabench.discoverybench_solve import iter_samples, sample_files, solve_discoverybench_sample

    records: dict[str, dict] = {}
    pending = [
        s for s in iter_samples(args.split, limit=None)
        if _is_econometrics_sample(str(s.id))
    ][: max(args.limit, 0)]

    if not pending:
        print("No econometrics DiscoveryBench samples matched.", file=sys.stderr)
        return 1

    print(f"Solving {len(pending)} econometrics sample(s) with [domain_profile:econometrics]", flush=True)
    for i, sample in enumerate(pending, 1):
        sid = str(sample.id)
        print(f"[{i}/{len(pending)}] {sid}", flush=True)
        records[sid] = solve_discoverybench_sample(
            sample_id=sid,
            formatted_input=str(sample.input or ""),
            query=str((sample.metadata or {}).get("query") or ""),
            files=sample_files(args.split, sid),
            api_base=args.api,
            budget_hours=args.budget_hours,
            domain_profile="econometrics",
        )

    result = asyncio.run(_score_records(records, args.split))
    result["solve_records"] = {
        sid: {
            "campaign_id": rec.get("campaign_id"),
            "campaign_status": rec.get("campaign_status"),
            "stop_reason": rec.get("stop_reason"),
            "extract_audit": rec.get("extract_audit"),
        }
        for sid, rec in records.items()
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nmean HMS: {result['hms_mean']:.4f} max HMS: {result['hms_max']:.4f} (n={result['n_samples']})")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
