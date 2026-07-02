#!/usr/bin/env python3
"""Patch inspect_ai Google provider to preserve Gemini 3 thought_signature on tool calls."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOOGLE_PY = (
    ROOT
    / "asta-bench"
    / ".venv"
    / "Lib"
    / "site-packages"
    / "inspect_ai"
    / "model"
    / "_providers"
    / "google.py"
)

MARKER = "# propab: gemini3 thought_signature b64"

OLD_TOOL_PARSE = """                    tool_calls.append(
                        ToolCall(
                            id=part.function_call.name,
                            function=part.function_call.name,
                            arguments=part.function_call.args,
                        )
                    )"""

NEW_TOOL_PARSE = """                    _internal = None
                    if part.thought_signature is not None:
                        _sig = part.thought_signature
                        if isinstance(_sig, bytes):
                            _sig = base64.b64encode(_sig).decode("ascii")
                        _internal = {"thought_signature": _sig}
                    tool_calls.append(
                        ToolCall(
                            id=part.function_call.name,
                            function=part.function_call.name,
                            arguments=part.function_call.args,
                            internal=_internal,
                        )
                    )"""

OLD_ASSISTANT_FC = """            content_parts.extend(
                [
                    Part.from_function_call(
                        name=tool_call.function,
                        args=tool_call.arguments,
                    )
                    for tool_call in message.tool_calls
                ]
            )"""

NEW_ASSISTANT_FC = f"""            for tool_call in message.tool_calls:
                _sig = None
                if isinstance(tool_call.internal, dict):
                    _raw = tool_call.internal.get("thought_signature")
                    if isinstance(_raw, str):
                        _sig = base64.b64decode(_raw)
                    elif _raw is not None:
                        _sig = _raw
                _part = Part.from_function_call(
                    name=tool_call.function,
                    args=tool_call.arguments,
                )
                if _sig is not None:
                    _part = Part(
                        function_call=_part.function_call,
                        thought_signature=_sig,
                    )
                content_parts.append(_part)  {MARKER}"""

LEGACY_MARKER = "# propab: gemini3 thought_signature"


def _ensure_base64_import(text: str) -> str:
    if "import base64" in text:
        return text
    return text.replace("import functools\n", "import functools\nimport base64\n", 1)


def main() -> int:
    if not GOOGLE_PY.is_file():
        print(f"Missing {GOOGLE_PY}", file=sys.stderr)
        return 1
    text = GOOGLE_PY.read_text(encoding="utf-8")
    if MARKER in text:
        print("thought_signature patch: already applied.", flush=True)
        return 0
    text = _ensure_base64_import(text)
    if LEGACY_MARKER in text and MARKER not in text:
        # upgrade legacy patch in-place
        text = text.replace(LEGACY_MARKER, MARKER)
        text = text.replace(
            "_sig = tool_call.internal.get(\"thought_signature\")",
            '_raw = tool_call.internal.get("thought_signature")\n'
            "                    if isinstance(_raw, str):\n"
            "                        _sig = base64.b64decode(_raw)\n"
            "                    elif _raw is not None:\n"
            "                        _sig = _raw",
        )
        text = text.replace(
            '_internal = {"thought_signature": part.thought_signature}',
            '_sig = part.thought_signature\n'
            "                        if isinstance(_sig, bytes):\n"
            "                            _sig = base64.b64encode(_sig).decode(\"ascii\")\n"
            '                        _internal = {"thought_signature": _sig}',
        )
        GOOGLE_PY.write_text(text, encoding="utf-8")
        print(f"Upgraded legacy patch in {GOOGLE_PY}", flush=True)
        return 0
    if OLD_TOOL_PARSE not in text or OLD_ASSISTANT_FC not in text:
        print("google.py layout changed — manual thought_signature patch required.", file=sys.stderr)
        return 1
    text = text.replace(OLD_TOOL_PARSE, NEW_TOOL_PARSE)
    text = text.replace(OLD_ASSISTANT_FC, NEW_ASSISTANT_FC)
    GOOGLE_PY.write_text(text, encoding="utf-8")
    print(f"Patched {GOOGLE_PY}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
