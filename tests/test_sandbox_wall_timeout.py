"""Sandbox wall-timeout classification (Docker SDK message shapes)."""

from services.worker.sub_agent_loop import _is_sandbox_wall_timeout


def test_timeout_in_message() -> None:
    assert _is_sandbox_wall_timeout({"ok": False, "message": "Read timeout after 480s"}) is True


def test_timed_out_without_substring_timeout() -> None:
    assert _is_sandbox_wall_timeout({"ok": False, "message": "Read timed out."}) is True


def test_error_type_read_timeout() -> None:
    assert _is_sandbox_wall_timeout({"ok": False, "error_type": "docker_read_timeout", "message": "x"}) is True


def test_stderr_deadline() -> None:
    assert (
        _is_sandbox_wall_timeout(
            {"ok": False, "message": "", "stderr": "deadline exceeded while waiting for container\n"}
        )
        is True
    )


def test_plain_validation_error_not_timeout() -> None:
    assert _is_sandbox_wall_timeout({"ok": False, "message": "SyntaxError: invalid syntax"}) is False


def test_docker_sdk_timeout_kwarg_mismatch_not_wall_timeout() -> None:
    assert (
        _is_sandbox_wall_timeout(
            {
                "ok": False,
                "error_type": "execution_error",
                "message": "run() got an unexpected keyword argument 'timeout'",
            }
        )
        is False
    )
