"""Host-aware data directory resolution for domain adapters.

Adapters ship their datasets under a container path (``/app/...``) inside the
docker image, but the same code must also run on a host / dev checkout where that
absolute path does not exist. Historically each adapter fell back to a *CWD*-relative
path (or, worse, the container path), so a tool invoked from anywhere other than the
repo root resolved to a non-existent ``/app/...`` file and raised ``FileNotFoundError``
even though the data was present in the checkout.

``resolve_data_dir`` fixes that by resolving the fallback relative to the repo root
(derived from this file's location, so it is independent of the current working
directory) and by honoring an explicit environment override. Container behavior is
preserved: when the ``/app/...`` path actually holds the data it still wins.
"""
from __future__ import annotations

import os
from pathlib import Path

# repo root is five parents up:
# packages/propab-core/propab/domain_adapters/_data_paths.py
#   parents[0]=domain_adapters [1]=propab [2]=propab-core [3]=packages [4]=<repo root>
REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_data_dir(
    *,
    sentinel: str,
    rel_path: str,
    env_var: str,
    container_dirs: tuple[Path, ...] = (),
) -> Path:
    """Resolve a dataset directory that works both in-container and on a host checkout.

    Resolution order:
      1. ``$env_var`` if set — explicit operator override, returned as-is (honored
         even when the sentinel is absent so a download/scratch location can be
         targeted).
      2. Any ``container_dirs`` candidate that actually contains ``sentinel``
         (preserves in-docker behavior where ``/app/...`` exists).
      3. The repo-root-relative ``rel_path`` when it contains ``sentinel`` — the
         CWD-independent host location.
      4. The legacy CWD-relative ``rel_path`` when it contains ``sentinel``
         (backward compatibility).
      5. Fallback: the repo-root-relative ``rel_path``. A stable, CWD-independent
         default even when the data file is absent, so callers/tests that probe
         ``<dir>/<sentinel>`` see the repo-relative path (matching the skipif
         guards) and skip cleanly instead of erroring on a container path.
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    repo_rel = REPO_ROOT / rel_path
    cwd_rel = Path(rel_path)
    for candidate in (*container_dirs, repo_rel, cwd_rel):
        if (candidate / sentinel).is_file():
            return candidate
    return repo_rel
