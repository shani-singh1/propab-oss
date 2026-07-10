from __future__ import annotations

import base64
import json
import re
from typing import Any

import docker

from propab.config import settings

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


_DOCKER_CLIENT = None


def _get_docker_client():
    """Reuse a single Docker client per worker process (avoids per-call connection cold start).

    The client timeout must exceed the longest per-call wait we make (``container.wait``
    with the sandbox wall budget), otherwise the HTTP layer aborts before our budget.
    """
    global _DOCKER_CLIENT
    if _DOCKER_CLIENT is None:
        _DOCKER_CLIENT = docker.from_env(timeout=900)
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
    image: str | None = None,
) -> dict[str, Any]:
    """
    Run Python in an isolated Docker container (no network), per ARCHITECTURE §8.4.
    Code should print JSON to stdout as the last line.

    Prepends SANDBOX_WALL_SEC and SANDBOX_REMAINING_SEC() helpers so codegen can throttle work.
    """
    client = _get_docker_client()
    image = image or str(getattr(settings, "sandbox_image", "") or "python:3.11-alpine")
    full_src = _sandbox_prepended_budget(timeout_sec) + code
    payload = base64.b64encode(full_src.encode("utf-8")).decode("ascii")
    inner = (
        "import base64,sys,json;"
        f"exec(compile(base64.b64decode('{payload}').decode('utf-8'), '<sandbox>', 'exec'))"
    )
    command = ["python", "-c", inner]
    mem_limit = f"{max(64, memory_mb)}m"
    wall = max(5, int(timeout_sec))

    # Detached run + bounded wait + kill: ``containers.run`` has no ``timeout`` kwarg, so a
    # hard wall-clock limit must be enforced via ``container.wait(timeout=...)`` and an
    # explicit kill on expiry. (A prior ``run(..., timeout=...)`` call raised TypeError and
    # made every code step fail instantly, mislabeled as a sandbox timeout.)
    container = None
    try:
        container = client.containers.run(
            image,
            command,
            network_mode="none",
            mem_limit=mem_limit,
            detach=True,
            stdout=True,
            stderr=True,
        )
    except docker.errors.ImageNotFound as exc:
        return {"ok": False, "error_type": "image_not_found", "message": str(exc), "stdout": "", "stderr": ""}
    except docker.errors.APIError as exc:
        return {"ok": False, "error_type": "docker_api", "message": str(exc), "stdout": "", "stderr": ""}
    except Exception as exc:
        return {"ok": False, "error_type": "execution_error", "message": str(exc), "stdout": "", "stderr": ""}

    def _logs() -> str:
        try:
            raw = container.logs(stdout=True, stderr=True)
            return raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        except Exception:
            return ""

    def _is_timeout_exc(exc: Exception) -> bool:
        # A genuine wall-clock timeout: container.wait's HTTP read exceeded `wall` (the
        # sandboxed code ran too long). NOT a daemon connection loss — see below.
        if isinstance(exc, TimeoutError):
            return True
        if requests is not None and isinstance(exc, requests.exceptions.ReadTimeout):
            return True
        return "timed out" in str(exc).lower() or "read timed out" in str(exc).lower()

    def _is_connection_exc(exc: Exception) -> bool:
        # C4 — a Docker daemon/socket TRANSPORT fault, not the model's code timing out.
        # Misclassifying it as docker_timeout makes the worker "shrink/rewrite" perfectly
        # good code in response to an infrastructure hiccup, and needlessly kills containers.
        return requests is not None and isinstance(exc, requests.exceptions.ConnectionError)

    try:
        try:
            status = container.wait(timeout=wall)
        except Exception as exc:
            if _is_timeout_exc(exc):
                stdout = _logs()
                try:
                    container.kill()
                except Exception:
                    pass
                return {
                    "ok": False,
                    "error_type": "docker_timeout",
                    "message": f"sandbox_wall_timeout: exceeded {wall}s",
                    "stdout": stdout,
                    "stderr": "",
                }
            if _is_connection_exc(exc):
                return {
                    "ok": False,
                    "error_type": "docker_transport",
                    "message": f"docker daemon/socket transport fault (infrastructure, not model code): {exc}",
                    "stdout": _logs(),
                    "stderr": "",
                }
            return {"ok": False, "error_type": "execution_error", "message": str(exc), "stdout": _logs(), "stderr": ""}

        exit_code = int(status.get("StatusCode", 0)) if isinstance(status, dict) else 0
        stdout = _logs()
        if exit_code != 0:
            return {
                "ok": False,
                "error_type": "container_error",
                "message": stdout or f"sandbox exited with status {exit_code}",
                "stdout": stdout,
                "stderr": stdout,
                "exit_code": exit_code,
            }
        parsed = _parse_stdout_json(stdout)
        return {"ok": True, "stdout": stdout, "stderr": "", "parsed": parsed}
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass


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
