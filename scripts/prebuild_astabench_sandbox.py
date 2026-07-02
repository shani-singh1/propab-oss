#!/usr/bin/env python3
"""One-time build of AstaBench DiscoveryBench sandbox image (reuse on later eval runs)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SANDBOX_DIR = ROOT / "asta-bench" / "astabench" / "util" / "sandbox"
IMAGE = "astabench-sandbox:8g"


def main() -> int:
    compose = ROOT / "asta-bench" / "astabench" / "util" / "sandbox" / "sandbox_compose.yaml"
    if not compose.is_file():
        print(f"Missing {compose}", file=sys.stderr)
        return 1
    # Skip if image already present
    probe = subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
    )
    if probe.returncode == 0:
        print(f"Sandbox image {IMAGE} already exists — skipping build.", flush=True)
        return 0
    print(f"Building sandbox image {IMAGE} (one-time, ~15–45 min)...", flush=True)
    # Build directly — runtime sandbox_compose.yaml is image-only so Inspect won't rebuild.
    rc = subprocess.call(
        ["docker", "build", "-t", IMAGE, "."],
        cwd=str(SANDBOX_DIR),
    )
    if rc != 0:
        return rc
    print(f"Built {IMAGE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
