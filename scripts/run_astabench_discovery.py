#!/usr/bin/env python3
"""
Run Propab on AstaBench DiscoveryBench (fixes.md Phase 0–2).

Examples:
  # Phase 0 baseline sanity (stock generate solver, no Propab):
  python scripts/run_astabench_discovery.py --phase sanity --limit 1

  # Phase 1 smoke (Propab solver, 1 problem):
  python scripts/run_astabench_discovery.py --phase smoke --limit 1

  # Phase 2 small validation batch:
  python scripts/run_astabench_discovery.py --phase validation --limit 5

  # Score logs:
  python scripts/run_astabench_discovery.py --score-only logs/astabench-propab-validation/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASTA = ROOT / "asta-bench"
SOLVER = ROOT / "integrations" / "astabench" / "propab_solver.py"


def _ensure_astabench_ready() -> int:
    """Apply local patches and ensure pre-built sandbox image exists."""
    import subprocess as sp

    for script in ("patch_astabench_windows.py", "patch_inspect_google_thought_signature.py"):
        rc = sp.call([sys.executable, str(ROOT / "scripts" / script)])
        if rc != 0:
            return rc
    rc = sp.call([sys.executable, str(ROOT / "scripts" / "prebuild_astabench_sandbox.py")])
    return rc


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


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd), env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Propab × AstaBench DiscoveryBench runner")
    parser.add_argument(
        "--phase",
        choices=("sanity", "smoke", "validation", "full-validation"),
        default="smoke",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget-minutes", type=float, default=20.0)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--task", default="astabench/discoverybench_validation")
    parser.add_argument("--score-only", default=None, help="Score an existing log directory")
    parser.add_argument("--display", default="plain")
    args = parser.parse_args()

    _load_dotenv()
    scorer = os.environ.get("ASTABENCH_DISCOVERYBENCH_SCORER_MODEL", "").strip()
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not os.environ.get("OPENAI_API_KEY", "").strip() and not scorer and provider not in ("gemini", "google"):
        print(
            "WARNING: No Gemini scorer configured and OPENAI_API_KEY is empty. "
            "Set LLM_PROVIDER=gemini + LLM_MODEL, or ASTABENCH_DISCOVERYBENCH_SCORER_MODEL.",
            file=sys.stderr,
        )
    if not ASTA.is_dir():
        print("asta-bench/ not found. Clone v0.3.1 first (see fixes.md Phase 0).", file=sys.stderr)
        return 1

    if args.score_only is None:
        setup_rc = _ensure_astabench_ready()
        if setup_rc != 0:
            return setup_rc

    env = os.environ.copy()
    env.setdefault("HF_TOKEN", env.get("HF_TOKEN", ""))
    env.setdefault("GOOGLE_API_KEY", env.get("GOOGLE_API_KEY", ""))
    env.setdefault("PROPAB_ASTABENCH_DATA_ROOT", str(ROOT / "data" / "astabench"))

    if args.score_only:
        log_dir = Path(args.score_only)
        return _run(_astabench_cmd(["score", str(log_dir)]), cwd=ASTA, env=env)

    limits = {
        "sanity": 1,
        "smoke": 1,
        "validation": 5,
        "full-validation": None,
    }
    limit = args.limit if args.limit is not None else limits[args.phase]
    log_dir = args.log_dir or {
        "sanity": "logs/astabench-sanity",
        "smoke": "logs/astabench-propab-smoke",
        "validation": "logs/astabench-propab-validation",
        "full-validation": "logs/astabench-propab-validation-full",
    }[args.phase]
    log_path = (ROOT / log_dir).resolve()
    log_path.mkdir(parents=True, exist_ok=True)

    if args.phase == "sanity":
        cmd = _inspect_cmd(
            [
                "eval",
                "--solver",
                "generate",
                "--model",
                _model(),
                args.task,
                "--limit",
                str(limit or 1),
                f"--log-dir={log_path}",
                f"--display={args.display}",
            ]
        )
        return _run(cmd, cwd=ASTA, env=env)

    solver_ref = f"{SOLVER.resolve()}@propab_campaign"
    cmd = _inspect_cmd(
        [
            "eval",
            "--solver",
            solver_ref,
            "--model",
            _model(),
            "-S",
            f"budget_minutes={args.budget_minutes}",
            "-S",
            f"api_url={env.get('PROPAB_API_URL', 'http://localhost:8000')}",
            args.task,
            f"--log-dir={log_path}",
            f"--display={args.display}",
            "--max-samples",
            "1",
        ]
    )
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    rc = _run(cmd, cwd=ASTA, env=env)
    if rc != 0:
        return rc

    manifest = {
        "phase": args.phase,
        "task": args.task,
        "log_dir": str(log_path),
        "model": _model(),
        "budget_minutes": args.budget_minutes,
        "note": "Run `python scripts/run_astabench_discovery.py --score-only <log_dir>` after eval.",
    }
    manifest_path = log_path / "propab_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sidecar = Path(env.get("PROPAB_ASTABENCH_DATA_ROOT", str(ROOT / "data" / "astabench"))) / "_campaign_log.jsonl"
    if sidecar.is_file():
        records = [json.loads(ln) for ln in sidecar.read_text(encoding="utf-8").splitlines() if ln.strip()]
        (log_path / "propab_campaign_ids.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nWrote {manifest_path}")
    print(f"Score with: python scripts/run_astabench_discovery.py --score-only {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
