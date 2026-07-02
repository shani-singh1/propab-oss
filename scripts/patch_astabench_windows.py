#!/usr/bin/env python3
"""Apply local patches to cloned asta-bench (Windows HF paths, sandbox mem)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "asta-bench" / "astabench" / "evals" / "discoverybench" / "task.py"
SANDBOX = ROOT / "asta-bench" / "astabench" / "util" / "sandbox" / "sandbox_compose.yaml"

OLD = """        hf_file_path = os.path.join(hf_directory, data_folder, filename)
        destination_path = os.path.join(
            output_base, hf_directory, data_folder, filename
        )"""

NEW = """        hf_file_path = posixpath.join(hf_directory, data_folder, filename)
        destination_path = os.path.join(
            output_base, *posixpath.normpath(posixpath.join(hf_directory, data_folder, filename)).split("/")
        )"""

IMPORT = "import posixpath"


def _patch_hf_paths() -> int:
    if not TASK.is_file():
        print(f"Missing {TASK}", flush=True)
        return 1
    text = TASK.read_text(encoding="utf-8")
    if "posixpath.join(hf_directory" in text:
        print("HF path patch: already applied.", flush=True)
        return 0
    if IMPORT not in text:
        text = text.replace("from typing import Literal", f"from typing import Literal\n{IMPORT}")
    if OLD not in text:
        print("task.py layout changed — manual HF patch required.", flush=True)
        return 1
    TASK.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(f"Patched {TASK}", flush=True)
    return 0


def _patch_sandbox_mem() -> int:
    if not SANDBOX.is_file():
        print(f"Missing {SANDBOX}", flush=True)
        return 1
    text = SANDBOX.read_text(encoding="utf-8")
    changed = False
    if "mem_limit: 48G" in text:
        text = text.replace("mem_limit: 48G", "mem_limit: 8G")
        changed = True
    if "image: astabench-sandbox:8g" not in text:
        if "build: ." in text:
            text = text.replace(
                "  default: \n    build: .",
                "  default:\n    image: astabench-sandbox:8g",
            ).replace(
                "  default:\n    build: .",
                "  default:\n    image: astabench-sandbox:8g",
            )
        else:
            text = text.replace(
                "  default:\n",
                "  default:\n    image: astabench-sandbox:8g\n",
                1,
            )
        changed = True
    # Inspect always rebuilds when compose has `build:` — runtime compose must be image-only.
    if "build: ." in text:
        text = text.replace("\n    build: .", "")
        changed = True
    if not changed and "mem_limit: 8G" in text and "build: ." not in text:
        print("Sandbox patch: already applied.", flush=True)
        return 0
    if changed:
        SANDBOX.write_text(text, encoding="utf-8")
        print(f"Patched {SANDBOX}", flush=True)
        return 0
    print("sandbox_compose.yaml layout changed — manual patch required.", flush=True)
    return 1


def main() -> int:
    rc = _patch_hf_paths()
    if rc != 0:
        return rc
    return _patch_sandbox_mem()


if __name__ == "__main__":
    raise SystemExit(main())
