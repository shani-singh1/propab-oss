#!/usr/bin/env python3
"""
Option B: decoupled solve-then-score for fair Propab × DiscoveryBench eval (fixes.md).

Campaigns run outside Inspect (no per-sample harness time ceiling). Completions are
cached, then Inspect scores them via the replay solver + DiscoveryBench HMS.

Examples:
  # Solve validation split (5 samples, 2h budget each) then score:
  python scripts/run_astabench_option_b.py --split validation --limit 5 --budget-hours 2

  # Score-only after solve:
  python scripts/run_astabench_option_b.py --score-only logs/astabench-option-b-validation

  # Resume solve (skip samples already in cache):
  python scripts/run_astabench_option_b.py --solve-only --resume --log-dir logs/astabench-option-b-validation
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASTA = ROOT / "asta-bench"
REPLAY_SOLVER = ROOT / "integrations" / "astabench" / "replay_solver.py"
SOLUTIONS_NAME = "option_b_solutions.json"
MANIFEST_NAME = "option_b_manifest.json"


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k:
            os.environ[k] = v


def _ensure_astabench_ready() -> int:
    import subprocess as sp

    for script in ("patch_astabench_windows.py", "patch_inspect_google_thought_signature.py"):
        rc = sp.call([sys.executable, str(ROOT / "scripts" / script)])
        if rc != 0:
            return rc
    return sp.call([sys.executable, str(ROOT / "scripts" / "prebuild_astabench_sandbox.py")])


def _uv() -> list[str]:
    if shutil.which("uv"):
        return ["uv", "run"]
    return [sys.executable, "-m"]


def _inspect_cmd(extra: list[str]) -> list[str]:
    base = _uv()
    if base[0] == "uv":
        return base + ["inspect"] + extra
    return base + ["inspect_ai"] + extra


def _astabench_cmd(extra: list[str]) -> list[str]:
    base = _uv()
    if base[0] == "uv":
        return base + ["astabench"] + extra
    return base + ["astabench"] + extra


def _model() -> str:
    provider = os.environ.get("LLM_PROVIDER", "google").lower()
    model = os.environ.get("LLM_MODEL", "gemini-3-flash-preview")
    if provider in ("gemini", "google"):
        return f"google/{model}"
    if provider == "openai":
        return f"openai/{model}"
    return model


def _task_id(split: str) -> str:
    return f"astabench/discoverybench_{split}"


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd), env=env)


def _load_solutions(path: Path) -> dict:
    if not path.is_file():
        return {"meta": {}, "samples": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "samples" not in data:
        data = {"meta": {}, "samples": data}
    return data


def _save_solutions(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _solve_phase(
    *,
    log_dir: Path,
    task: str,
    limit: int | None,
    budget_hours: float,
    api_url: str,
    resume: bool,
    force: bool,
    max_hypotheses: int,
    poll_sec: float,
) -> int:
    sys.path.insert(0, str(ROOT))
    from integrations.astabench.discoverybench_solve import (
        append_campaign_sidecar,
        data_root,
        iter_samples,
        sample_files,
        solve_discoverybench_sample,
    )

    cache_path = log_dir / SOLUTIONS_NAME
    cache = _load_solutions(cache_path)
    cache.setdefault("meta", {}).update(
        {
            "mode": "option_b",
            "task": task,
            "budget_hours": budget_hours,
            "model": _model(),
            "started_at": cache.get("meta", {}).get("started_at")
            or datetime.now(timezone.utc).isoformat(),
        }
    )
    samples_map: dict = cache.setdefault("samples", {})

    split_key = "validation" if "validation" in task else "test"
    pending = list(iter_samples(split_key, limit=limit))
    print(f"Solve phase: {len(pending)} sample(s), budget={budget_hours}h each", flush=True)

    for i, sample in enumerate(pending, 1):
        sid = str(sample.id)
        if sid in samples_map and resume and not force:
            print(f"[{i}/{len(pending)}] skip {sid} (cached)", flush=True)
            continue

        print(f"[{i}/{len(pending)}] solving {sid} ...", flush=True)
        t0 = time.monotonic()
        record = solve_discoverybench_sample(
            sample_id=sid,
            formatted_input=str(sample.input or ""),
            query=str((sample.metadata or {}).get("query") or ""),
            files=sample_files(split_key, sid),
            api_base=api_url,
            budget_hours=budget_hours,
            max_hypotheses=max_hypotheses,
            poll_sec=poll_sec,
            dest_root=data_root(),
        )
        elapsed = time.monotonic() - t0
        record["solve_wall_sec"] = round(elapsed, 1)
        samples_map[sid] = record
        append_campaign_sidecar(record, dest_root=data_root())
        _save_solutions(cache_path, cache)
        print(
            f"  done {sid} in {elapsed/60:.1f} min "
            f"status={record.get('campaign_status')} stop={record.get('stop_reason')}",
            flush=True,
        )

    cache["meta"]["solve_completed_at"] = datetime.now(timezone.utc).isoformat()
    _save_solutions(cache_path, cache)
    print(f"Wrote {cache_path}", flush=True)
    return 0


def _score_phase(
    *,
    log_dir: Path,
    task: str,
    limit: int | None,
    display: str,
    env: dict[str, str],
) -> int:
    cache_path = (log_dir / SOLUTIONS_NAME).resolve()
    if not cache_path.is_file():
        print(f"Missing {cache_path} — run solve phase first.", file=sys.stderr)
        return 1

    n_cached = len(_load_solutions(cache_path).get("samples", {}))
    if n_cached == 0:
        print("Solutions cache is empty.", file=sys.stderr)
        return 1

    score_limit = limit if limit is not None else n_cached
    if score_limit > n_cached:
        print(
            f"WARNING: --limit {score_limit} > cached samples ({n_cached}); scoring {n_cached}.",
            flush=True,
        )
        score_limit = n_cached

    solver_ref = f"{REPLAY_SOLVER.resolve()}@propab_replay"
    cmd = _inspect_cmd(
        [
            "eval",
            "--solver",
            solver_ref,
            "--model",
            _model(),
            "-S",
            f"cache_path={cache_path.as_posix()}",
            task,
            f"--log-dir={log_dir.resolve()}",
            f"--display={display}",
            "--max-samples",
            "1",
        ]
    )
    if limit is not None:
        cmd.extend(["--limit", str(score_limit)])

    rc = _run(cmd, cwd=ASTA, env=env)
    if rc != 0:
        return rc
    print(
        f"Scored via Inspect eval log in {log_dir}. "
        f"Summarize: python scripts/summarize_astabench_propab_run.py {log_dir} --option-b",
        flush=True,
    )
    return 0


def _write_manifest(log_dir: Path, *, args: argparse.Namespace, task: str) -> None:
    manifest = {
        "mode": "option_b",
        "task": task,
        "log_dir": str(log_dir.resolve()),
        "solutions": str((log_dir / SOLUTIONS_NAME).resolve()),
        "budget_hours": args.budget_hours,
        "limit": args.limit,
        "model": _model(),
        "note": "Fair eval: campaigns run outside Inspect; scores from replay + HMS.",
    }
    (log_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Propab Option B (decoupled solve-then-score)")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=float(os.environ.get("PROPAB_OPTION_B_BUDGET_HOURS", "2")),
        help="Per-sample Propab campaign budget (default 2h, env PROPAB_OPTION_B_BUDGET_HOURS)",
    )
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--max-hypotheses", type=int, default=120)
    parser.add_argument("--poll-sec", type=float, default=20.0)
    parser.add_argument("--display", default="plain")
    parser.add_argument("--solve-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip samples already in solutions cache")
    parser.add_argument("--force", action="store_true", help="Re-solve even if cached")
    args = parser.parse_args()

    # Re-exec inside asta-bench uv env so `astabench` imports resolve.
    if not os.environ.get("PROPAB_OPTION_B_IN_UV"):
        env = os.environ.copy()
        env["PROPAB_OPTION_B_IN_UV"] = "1"
        py_path = os.pathsep.join(
            p for p in (str(ROOT), str(ROOT / "packages" / "propab-core")) if p
        )
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = py_path + (os.pathsep + existing if existing else "")
        cmd = _uv() + [str(Path(__file__).resolve()), *sys.argv[1:]]
        return subprocess.call(cmd, cwd=str(ASTA), env=env)

    _load_dotenv()
    if not ASTA.is_dir():
        print("asta-bench/ not found.", file=sys.stderr)
        return 1

    if not args.score_only:
        rc = _ensure_astabench_ready()
        if rc != 0:
            return rc

    task = _task_id(args.split)
    log_dir = Path(
        args.log_dir or f"logs/astabench-option-b-{args.split}"
    )
    if not log_dir.is_absolute():
        log_dir = (ROOT / log_dir).resolve()
    else:
        log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_TOKEN", env.get("HF_TOKEN", ""))
    env.setdefault("GOOGLE_API_KEY", env.get("GOOGLE_API_KEY", ""))
    env.setdefault("PROPAB_ASTABENCH_DATA_ROOT", str(ROOT / "data" / "astabench"))
    api_url = (args.api_url or env.get("PROPAB_API_URL") or "http://localhost:8000").rstrip("/")

    do_solve = not args.score_only
    do_score = not args.solve_only

    if do_solve:
        rc = _solve_phase(
            log_dir=log_dir,
            task=task,
            limit=args.limit,
            budget_hours=args.budget_hours,
            api_url=api_url,
            resume=args.resume,
            force=args.force,
            max_hypotheses=args.max_hypotheses,
            poll_sec=args.poll_sec,
        )
        if rc != 0:
            return rc
        _write_manifest(log_dir, args=args, task=task)

    if do_score:
        rc = _score_phase(
            log_dir=log_dir,
            task=task,
            limit=args.limit,
            display=args.display,
            env=env,
        )
        if rc != 0:
            return rc
        print(
            f"\nSummarize: python scripts/summarize_astabench_propab_run.py {log_dir} "
            f"--option-b",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
