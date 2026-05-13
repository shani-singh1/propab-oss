#!/usr/bin/env python3
"""Confirm the worker reads ``sandbox_code_max_retries`` from config (startup log line).

Runs one ``fast_path`` baseline sub-agent (no full think–act loop) with ``--profile campaign``,
which still emits ``[sub_agent] startup ... sandbox_code_max_retries=...`` before the fast exit.

Usage (repo root)::

    python scripts/verify_sub_agent_startup_log.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "propab",
        "agent",
        "--profile",
        "campaign",
        "--fast-baseline",
        "--cleanup",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        print("agent harness failed:", proc.returncode, file=sys.stderr)
        print(merged[-4000:], file=sys.stderr)
        return 1
    if "[sub_agent] startup" not in merged:
        print("MISSING startup log line in subprocess output.", file=sys.stderr)
        print(merged[-6000:], file=sys.stderr)
        return 1
    if "sandbox_code_max_retries=1" not in merged:
        print("Expected sandbox_code_max_retries=1 (campaign profile) in logs.", file=sys.stderr)
        print(merged[-6000:], file=sys.stderr)
        return 1
    print("ok: startup log contains sandbox_code_max_retries=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
