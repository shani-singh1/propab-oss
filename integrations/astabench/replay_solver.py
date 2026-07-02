"""
InspectAI solver: inject precomputed DiscoveryBench answers (Option B score phase).

Usage:
  inspect eval integrations/astabench/replay_solver.py@propab_replay \\
    -T cache_path=logs/astabench-option-b-validation/option_b_solutions.json \\
    --model google/gemini-3-flash-preview \\
    astabench/discoverybench_validation --log-dir=logs/astabench-option-b-validation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from inspect_ai.solver import Generate, Solver, solver

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_cache(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = data.get("samples") or data
    if not isinstance(samples, dict):
        raise ValueError("cache must have a 'samples' object mapping sample_id → record")
    return samples


@solver
def propab_replay(cache_path: str) -> Solver:
    """Set state.output.completion from Option B solutions cache (no Propab API calls)."""

    cache = _load_cache(cache_path)

    async def solve(state, generate: Generate):
        sample_id = str(state.sample_id or state.metadata.get("id") or "")
        rec = cache.get(sample_id)
        if rec is None:
            raise KeyError(
                f"No cached solution for sample {sample_id!r} in {cache_path}. "
                "Run the Option B solve phase first."
            )
        completion = rec.get("completion")
        if not completion and rec.get("answer"):
            from integrations.astabench.answer_extract import format_completion

            completion = format_completion(rec["answer"])
        if not completion:
            raise ValueError(f"Cached record for {sample_id} has no completion")
        state.output.completion = str(completion)
        return state

    return solve
