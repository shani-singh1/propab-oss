from __future__ import annotations

import base64
import json
import re
from typing import Any

import docker


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
    """
    client = docker.from_env()
    payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
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
    except Exception as exc:
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
