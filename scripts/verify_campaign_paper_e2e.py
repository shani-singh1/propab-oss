#!/usr/bin/env python3
"""Verify campaign paper ↔ ledger alignment without a full Docker/Celery campaign.

Runs automated contract tests that exercise ``build_campaign_synthesis_payload`` and
``generate_prose_sections`` so abstract counts match the hypothesis tree ledger.

For a real stack smoke test after this passes: start a short campaign
(``PROPAB_PROFILE=campaign``, reduced budget), then compare log ``confirmed=N`` to
the generated paper abstract.

Usage (from repo root)::

    python scripts/verify_campaign_paper_e2e.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    tests = [
        "tests/test_campaign_synthesis_builder.py",
        "tests/test_paper_sections.py",
    ]
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", *tests]
    print("Running:", " ".join(cmd), flush=True)
    return int(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    raise SystemExit(main())
