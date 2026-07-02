#!/usr/bin/env python3
"""Quick check: DiscoveryBench HMS scorer resolves to Gemini from Propab .env."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASTA = ROOT / "asta-bench"
sys.path.insert(0, str(ASTA))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip()

from astabench.evals.discoverybench.lm_utils import discoverybench_scorer_model

print(discoverybench_scorer_model())
