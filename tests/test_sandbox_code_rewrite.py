from __future__ import annotations

import asyncio

from services.worker.sandbox_code_rewrite import (
    extract_python_from_llm_response,
    looks_like_heavy_training_code,
    rewrite_sandbox_code_after_timeout,
)


def test_looks_like_heavy_training_code_short_false() -> None:
    stub = "import json, sys\nprint(json.dumps({'sandbox':'ok'}))\n"
    assert looks_like_heavy_training_code(stub) is False


def test_looks_like_heavy_training_code_torch_true() -> None:
    code = "# x\n" * 50 + "\nimport torch\nfor epoch in range(1000):\n  pass\n"
    assert looks_like_heavy_training_code(code) is True


def test_extract_python_from_fence() -> None:
    raw = """Here is the code:
```python
import json
print(json.dumps({"sandbox": "ok"}))
```
"""
    out = extract_python_from_llm_response(raw)
    assert out is not None
    assert "import json" in out
    assert "sandbox" in out


def test_rewrite_sandbox_code_uses_llm_response() -> None:
    class _FakeLLM:
        async def call(self, **kwargs):  # noqa: ANN003, ANN002
            return "```python\nimport json\nprint(json.dumps({'sandbox':'ok','note':'rewritten'}))\n```"

    code = "# block\n" * 40 + "\nimport torch\nfor epoch in range(999):\n  pass\n"

    async def _run() -> str | None:
        return await rewrite_sandbox_code_after_timeout(
            _FakeLLM(),
            session_id="s",
            hypothesis_id="h",
            code=code,
            sandbox_timeout_sec=120,
            domain="ml_research",
        )

    new_c = asyncio.run(_run())
    assert new_c is not None
    assert "rewritten" in new_c or "json.dumps" in new_c
