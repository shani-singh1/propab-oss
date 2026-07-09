"""Run agent Python in an isolated Docker sandbox WITH staged data files.

GeneBench-Pro problems hand the agent messy data files (`.tsv.gz`, `.npz`, `.csv.gz`)
that its code must read. The worker's `run_sandboxed_python` executes inline code but
mounts no data, so this helper stages a problem's `data_files/` into the container via
`put_archive` (robust across Windows/WSL Docker — no host bind-mount path pain) and runs
the agent program against them. Network is disabled, same as the core sandbox.
"""
from __future__ import annotations

import io
import json
import re
import tarfile
from pathlib import Path
from typing import Any

import docker

_WORKDIR = "/work"


def _build_tar(data_dir: Path, code: str) -> bytes:
    """Tar the problem's data files under work/data_files/ plus the agent program."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in sorted(data_dir.rglob("*")):
            if p.is_file():
                arc = "work/data_files/" + str(p.relative_to(data_dir)).replace("\\", "/")
                tar.add(str(p), arcname=arc)
        code_bytes = code.encode("utf-8")
        info = tarfile.TarInfo("work/__run.py")
        info.size = len(code_bytes)
        tar.addfile(info, io.BytesIO(code_bytes))
    return buf.getvalue()


def _parse_last_json(stdout: str) -> Any:
    lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
    for candidate in reversed(lines[-8:]):
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    m = re.search(r"\{[\s\S]*\}\s*$", stdout)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def run_code_with_data(
    code: str,
    data_dir: str | Path,
    *,
    image: str,
    timeout_sec: int = 180,
    memory_mb: int = 4096,
) -> dict[str, Any]:
    """Execute `code` in a network-isolated container with the problem's data staged
    read-only at /work/data_files/. Returns {ok, exit_code, stdout, parsed, error_type}.
    """
    data_dir = Path(data_dir)
    client = docker.from_env(timeout=max(120, timeout_sec + 60))
    try:
        container = client.containers.create(
            image,
            ["python", "/work/__run.py"],
            network_mode="none",
            mem_limit=f"{max(256, memory_mb)}m",
            working_dir=_WORKDIR,
            stdin_open=False,
            tty=False,
        )
    except docker.errors.ImageNotFound as exc:
        return {"ok": False, "error_type": "image_not_found", "message": str(exc), "stdout": ""}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error_type": "create_error", "message": str(exc), "stdout": ""}

    def _logs() -> str:
        try:
            raw = container.logs(stdout=True, stderr=True)
            return raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        except Exception:
            return ""

    try:
        container.put_archive("/", _build_tar(data_dir, code))
        container.start()
        try:
            status = container.wait(timeout=timeout_sec)
        except Exception as exc:  # noqa: BLE001 — read timeout => wall-clock kill
            out = _logs()
            try:
                container.kill()
            except Exception:
                pass
            return {"ok": False, "error_type": "timeout",
                    "message": f"sandbox exceeded {timeout_sec}s", "stdout": out}
        exit_code = int(status.get("StatusCode", 0)) if isinstance(status, dict) else 0
        out = _logs()
        return {
            "ok": exit_code == 0,
            "exit_code": exit_code,
            "stdout": out,
            "parsed": _parse_last_json(out),
            "error_type": None if exit_code == 0 else "container_error",
        }
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass
