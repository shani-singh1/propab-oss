"""Run one Propab campaign per DiscoveryBench sample (shared by Option A/B)."""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from integrations.astabench.answer_extract import extract_discoverybench_answer, format_completion
from integrations.astabench.audit import audit_campaign_answer
from integrations.astabench.campaign_client import health_check, launch_campaign, wait_for_campaign
from integrations.astabench.data_mount import build_campaign_question, stage_sample_files

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return _REPO_ROOT


def data_root() -> Path:
    env = os.environ.get("PROPAB_ASTABENCH_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    return _REPO_ROOT / "data" / "astabench"


@lru_cache(maxsize=4)
def _file_index(task_key: str) -> dict[str, dict[str, str]]:
    asta = _REPO_ROOT / "asta-bench"
    if task_key in ("validation", "astabench/discoverybench_validation"):
        from astabench.evals.discoverybench.task import discoverybench_validation

        task = discoverybench_validation()
    elif task_key in ("test", "astabench/discoverybench_test"):
        from astabench.evals.discoverybench.task import discoverybench_test

        task = discoverybench_test()
    else:
        raise ValueError(f"Unknown DiscoveryBench task: {task_key}")

    index: dict[str, dict[str, str]] = {}
    for sample in task.dataset:
        resolved: dict[str, str] = {}
        for rel, src in (sample.files or {}).items():
            p = Path(src)
            if not p.is_absolute():
                p = (asta / p).resolve()
            if p.is_file():
                resolved[rel] = str(p)
        index[str(sample.id)] = resolved
    return index


def sample_files(task: str, sample_id: str) -> dict[str, str]:
    return dict(_file_index(task).get(str(sample_id), {}))


def iter_samples(task: str, *, limit: int | None = None) -> Iterator[Any]:
    if task in ("validation", "astabench/discoverybench_validation"):
        from astabench.evals.discoverybench.task import discoverybench_validation

        dataset = discoverybench_validation().dataset
    elif task in ("test", "astabench/discoverybench_test"):
        from astabench.evals.discoverybench.task import discoverybench_test

        dataset = discoverybench_test().dataset
    else:
        raise ValueError(f"Unknown task: {task}")

    n = 0
    for sample in dataset:
        yield sample
        n += 1
        if limit is not None and n >= limit:
            break


def solve_discoverybench_sample(
    *,
    sample_id: str,
    formatted_input: str,
    query: str,
    files: dict[str, str] | None,
    api_base: str,
    budget_hours: float,
    max_hypotheses: int = 80,
    poll_sec: float = 15.0,
    abstain_if_not_strong: bool = True,
    dest_root: Path | None = None,
    domain_profile: str | None = None,
) -> dict[str, Any]:
    """Launch Propab, wait for terminal status, return answer + audit metadata."""
    api = api_base.rstrip("/")
    if not health_check(api):
        raise RuntimeError(
            f"Propab API not reachable at {api}. "
            "Start: docker compose -f docker-compose.yml -f docker-compose.mount-dev.yml "
            "-f docker-compose.astabench.yml up -d"
        )

    root = dest_root or data_root()
    mounted = stage_sample_files(sample_id=sample_id, files=files, dest_root=root)
    question = build_campaign_question(
        formatted_input=formatted_input,
        mounted_paths=mounted,
        query=query,
        domain_profile=domain_profile,
    )

    budget_hours = max(0.1, float(budget_hours))
    campaign_id = launch_campaign(
        api_base=api,
        question=question,
        budget_hours=budget_hours,
        max_hypotheses=max_hypotheses,
    )
    logger.info("Launched Propab campaign %s for sample %s", campaign_id, sample_id)

    # Checkpoint immediately so resume can reattach if the waiter is interrupted.
    checkpoint = root / "_solve_checkpoints.json"
    checkpoints: dict[str, Any] = {}
    if checkpoint.is_file():
        try:
            checkpoints = json.loads(checkpoint.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            checkpoints = {}
    checkpoints[str(sample_id)] = {
        "campaign_id": campaign_id,
        "budget_hours": budget_hours,
        "launched": True,
    }
    checkpoint.write_text(json.dumps(checkpoints, indent=2), encoding="utf-8")

    payload = wait_for_campaign(
        api_base=api,
        campaign_id=campaign_id,
        poll_sec=poll_sec,
        budget_hours=budget_hours,
    )

    answer, extract_audit = extract_discoverybench_answer(
        payload,
        abstain_if_not_strong=abstain_if_not_strong,
    )
    fab_audit = audit_campaign_answer(payload, answer)
    completion = format_completion(answer)

    record: dict[str, Any] = {
        "sample_id": sample_id,
        "campaign_id": campaign_id,
        "answer": answer,
        "completion": completion,
        "extract_audit": extract_audit,
        "fabrication_audit": fab_audit,
        "campaign_status": (payload.get("campaign") or {}).get("status"),
        "stop_reason": (payload.get("campaign") or {}).get("stop_reason"),
    }
    return record


def append_campaign_sidecar(record: dict[str, Any], *, dest_root: Path | None = None) -> None:
    sidecar = (dest_root or data_root()) / "_campaign_log.jsonl"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
