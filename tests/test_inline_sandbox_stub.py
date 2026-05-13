"""Trusted inline execution for agent sandbox stubs (no Docker)."""

from services.worker.sub_agent_loop import (
    _is_trusted_inline_sandbox_code,
    _run_inline_trusted_sandbox_code,
    _sandbox_diag_tails,
    _think_act_stub_code,
)


def test_think_act_stub_trusted_and_runs() -> None:
    code = _think_act_stub_code("smoke")
    assert _is_trusted_inline_sandbox_code(code)
    out = _run_inline_trusted_sandbox_code(code)
    assert out.get("ok") is True
    assert isinstance(out.get("parsed"), dict)
    assert out["parsed"].get("sandbox") == "ok"


def test_think_act_stub_multiline_description_is_safe() -> None:
    """Newlines in the description must not become executable (old ``# {desc}`` bug)."""
    evil = "x\nimport os\nos.system('rm -rf /')\n"
    code = _think_act_stub_code(evil)
    out = _run_inline_trusted_sandbox_code(code)
    assert out.get("ok") is True
    parsed = out.get("parsed")
    assert isinstance(parsed, dict)
    assert parsed.get("computation") == evil


def test_think_act_stub_open_in_description_not_trusted_by_heuristic_but_runs() -> None:
    """Substring heuristics see ``open(`` inside JSON text; inline exec is still safe."""
    desc = "read weights via open(path)"
    code = _think_act_stub_code(desc)
    assert _is_trusted_inline_sandbox_code(code) is False
    out = _run_inline_trusted_sandbox_code(code)
    assert out.get("ok") is True
    assert out["parsed"].get("sandbox") == "ok"


def test_heuristic_tail_trusted() -> None:
    code = 'import json,sys\nprint(json.dumps({"sandbox":"ok","hypothesis_rank":1}))\n'
    assert _is_trusted_inline_sandbox_code(code)
    out = _run_inline_trusted_sandbox_code(code)
    assert out.get("ok") is True


def test_untrusted_code_rejected() -> None:
    assert _is_trusted_inline_sandbox_code("import os; os.system('ls')") is False


def test_sandbox_diag_tails() -> None:
    d = _sandbox_diag_tails(
        {"stderr": "a" * 100, "message": "timeout", "error_type": "docker_timeout"},
        maxlen=20,
    )
    assert d["stderr_tail"] == "a" * 20
    assert d["message_tail"] == "timeout"
    assert d["sandbox_error_type"] == "docker_timeout"
