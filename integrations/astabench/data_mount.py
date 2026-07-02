"""Stage DiscoveryBench dataset files for Propab worker sandboxes."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

_INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')


def _safe_dir_name(sample_id: str) -> str:
    """Filesystem-safe directory name (DiscoveryBench ids use ``|``)."""
    return _INVALID_PATH_CHARS.sub("_", str(sample_id))


def stage_sample_files(
    *,
    sample_id: str,
    files: dict[str, str] | None,
    dest_root: Path,
) -> dict[str, str]:
    """
    Copy AstaBench sample files into ``dest_root/{safe_sample_id}/``.

    Returns mapping of sandbox-relative paths → absolute paths inside the Propab
    container (``/app/astabench-data/...``).
    """
    dir_name = _safe_dir_name(sample_id)
    out_dir = dest_root / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    mounted: dict[str, str] = {}
    for rel, src in (files or {}).items():
        src_path = Path(src)
        if not src_path.is_file():
            continue
        rel_norm = rel.replace("\\", "/").lstrip("/")
        target = out_dir / rel_norm
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, target)
        mounted[rel_norm] = f"/app/astabench-data/{dir_name}/{rel_norm}"
    return mounted


def build_campaign_question(
    *,
    formatted_input: str,
    mounted_paths: dict[str, str],
    query: str,
) -> str:
    """Pin AstaBench problem as Propab campaign question (cold start, no seeded beliefs)."""
    paths_block = "\n".join(f"- {p}" for p in mounted_paths.values()) or "(no files staged)"
    return (
        f"{formatted_input.strip()}\n\n"
        "── AstaBench / Propab execution context ──\n"
        "Work cold from the dataset — do not assume any hypothesis before analyzing data.\n"
        "Load and analyze the dataset files from these absolute paths inside the experiment sandbox:\n"
        f"{paths_block}\n\n"
        f"Discovery query to answer: {query.strip()}\n"
        "Goal: derive a single falsifiable scientific hypothesis supported by statistical analysis "
        "of the provided data, with a reproducible workflow summary."
    )
