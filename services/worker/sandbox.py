from __future__ import annotations

import base64
import json
import re
from typing import Any

import docker

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


_DOCKER_CLIENT = None


def _get_docker_client():
    """Reuse a single Docker client per worker process (avoids per-call connection cold start)."""
    global _DOCKER_CLIENT
    if _DOCKER_CLIENT is None:
        _DOCKER_CLIENT = docker.from_env()
    return _DOCKER_CLIENT


def _sandbox_prepended_budget(timeout_sec: int) -> str:
    """Expose wall budget inside user code so training loops can early-exit before Docker SIGKILL."""
    wall = max(15, int(timeout_sec))
    return (
        "# propab sandbox: bounded wall-clock budget (leave these names for custom training code)\n"
        "import time as _propab_sandbox_time\n"
        f"SANDBOX_WALL_SEC = {wall}\n"
        "_PROPAB_T0_MONO = _propab_sandbox_time.monotonic()\n"
        "def SANDBOX_REMAINING_SEC() -> float:\n"
        "    return SANDBOX_WALL_SEC - (_propab_sandbox_time.monotonic() - _PROPAB_T0_MONO)\n\n"
    )


def run_sandboxed_python(
    code: str,
    *,
    timeout_sec: int,
    memory_mb: int,
    image: str = "python:3.11-alpine",
) -> dict[str, Any]:
    """
    Run Python in an isolated Docker container (no network), per ARCHITECTURE §8.4.
    Code should print JSON to stdout as the last line.

    Prepends SANDBOX_WALL_SEC and SANDBOX_REMAINING_SEC() helpers so codegen can throttle work.
    """
    client = _get_docker_client()
    full_src = _sandbox_prepended_budget(timeout_sec) + code
    payload = base64.b64encode(full_src.encode("utf-8")).decode("ascii")
    inner = (
        "import base64,sys,json;"
        f"exec(compile(base64.b64decode('{payload}').decode('utf-8'), '<sandbox>', 'exec'))"
    )
    command = ["python", "-c", inner]
    mem_limit = f"{max(64, memory_mb)}m"
    try:
        output = client.containers.run(
            image,
            command,
            network_mode="none",
            mem_limit=mem_limit,
            remove=True,
            stdout=True,
            stderr=True,
            timeout=timeout_sec,
        )
    except docker.errors.ContainerError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        return {
            "ok": False,
            "error_type": "container_error",
            "message": stderr or str(exc),
            "stdout": exc.stdout.decode("utf-8", errors="replace") if exc.stdout else "",
            "stderr": stderr,
        }
    except docker.errors.APIError as exc:
        return {"ok": False, "error_type": "docker_api", "message": str(exc), "stdout": "", "stderr": ""}
    except TimeoutError as exc:
        return {
            "ok": False,
            "error_type": "docker_timeout",
            "message": f"sandbox_wall_timeout: {exc!s}",
            "stdout": "",
            "stderr": "",
        }
    except Exception as exc:
        if requests is not None and isinstance(exc, requests.exceptions.ReadTimeout):
            return {
                "ok": False,
                "error_type": "docker_read_timeout",
                "message": f"sandbox_wall_timeout: {exc!s}",
                "stdout": "",
                "stderr": "",
            }
        return {"ok": False, "error_type": "execution_error", "message": str(exc), "stdout": "", "stderr": ""}

    stdout = output.decode("utf-8", errors="replace") if isinstance(output, (bytes, bytearray)) else str(output)
    parsed = _parse_stdout_json(stdout)
    return {"ok": True, "stdout": stdout, "stderr": "", "parsed": parsed}


def _parse_stdout_json(stdout: str) -> Any:
    lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    for candidate in reversed(lines[-5:]):
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    m = re.search(r"\{[\s\S]*\}", stdout)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None
