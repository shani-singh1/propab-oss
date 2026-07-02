"""
InspectAI solver: run one Propab campaign per AstaBench problem.

Usage (from repo root, Propab stack running, asta-bench deps installed):

  inspect eval integrations/astabench/propab_solver.py@propab_campaign \\
    --model google/gemini-3-flash-preview \\
    -T budget_minutes=20 \\
    astabench/discoverybench_validation --limit 1 \\
    --log-dir logs/astabench-propab-smoke/

Requires: GOOGLE_API_KEY (Propab + optional), OPENAI_API_KEY (DiscoveryBench HMS scorer),
HF_TOKEN (gated dataset), Propab API at PROPAB_API_URL (default http://localhost:8000).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from inspect_ai.solver import Generate, Solver, solver
from inspect_ai.util import store

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "packages" / "propab-core") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "packages" / "propab-core"))

from integrations.astabench.discoverybench_solve import (
    append_campaign_sidecar,
    data_root,
    sample_files,
    solve_discoverybench_sample,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _validation_file_index() -> dict[str, dict[str, str]]:
    from integrations.astabench.discoverybench_solve import _file_index

    return _file_index("validation")


def _discoverybench_sample_files(sample_id: str) -> dict[str, str]:
    """Resolve DiscoveryBench dataset file paths for a sample id (host paths)."""
    return sample_files("validation", sample_id)


def _data_root() -> Path:
    return data_root()


@solver
def propab_campaign(
    api_url: str | None = None,
    budget_minutes: float = 20.0,
    max_hypotheses: int = 80,
    poll_sec: float = 15.0,
    abstain_if_not_strong: bool = True,
) -> Solver:
    """
    Run a time-bounded Propab campaign for each AstaBench sample (Option A).

    Option A validates adapter wiring only — a low score is ambiguous with the
    harness time ceiling. Use Option B (decoupled solve-then-score) for fair
    architecture evaluation; see fixes.md Phase 1.

    Parameters are Inspect -T overrides, e.g. -T budget_minutes=15.
    """

    async def solve(state, generate: Generate):
        api = (api_url or os.environ.get("PROPAB_API_URL") or "http://localhost:8000").rstrip("/")
        sample_id = str(state.sample_id or state.metadata.get("id") or "unknown")
        query = str((state.metadata or {}).get("query") or "")
        files = _discoverybench_sample_files(sample_id)

        budget_hours = max(0.1, float(budget_minutes) / 60.0)
        record = solve_discoverybench_sample(
            sample_id=sample_id,
            formatted_input=str(state.input or ""),
            query=query,
            files=files,
            api_base=api,
            budget_hours=budget_hours,
            max_hypotheses=max_hypotheses,
            poll_sec=poll_sec,
            abstain_if_not_strong=abstain_if_not_strong,
            dest_root=_data_root(),
        )

        store().set("propab_campaign_id", record["campaign_id"])
        store().set("propab_extract_audit", record["extract_audit"])
        store().set("propab_fabrication_audit", record["fabrication_audit"])
        append_campaign_sidecar(record, dest_root=_data_root())

        state.output.completion = record["completion"]
        return state

    return solve
