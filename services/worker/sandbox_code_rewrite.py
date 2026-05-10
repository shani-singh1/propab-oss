"""LLM-assisted simplification of sandbox Python after a wall-clock timeout."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from propab.llm import LLMClient

logger = logging.getLogger(__name__)

_HEAVY_HINTS = (
    "torch",
    "tensorflow",
    "keras",
    "sklearn",
    "fit(",
    "epoch",
    "n_epoch",
    "training_loop",
    "for epoch",
    "dataloader",
    "mnist",
    "DataLoader",
)


def looks_like_heavy_training_code(code: str) -> bool:
    """True when code is plausibly a real training script (not the tiny heuristic stub)."""
    s = (code or "").strip()
    if len(s) < 220:
        return False
    low = s.lower()
    return any(h in low for h in _HEAVY_HINTS)


def extract_python_from_llm_response(raw: str) -> str | None:
    """Pull executable Python from an LLM reply (markdown fences or whole body)."""
    text = (raw or "").strip()
    if not text:
        return None
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            chunk = p.strip()
            if chunk.lower().startswith("python"):
                chunk = chunk[6:].lstrip("\n")
            if len(chunk) >= 12 and ("\n" in chunk or "import " in chunk or "print(" in chunk):
                return chunk
    if len(text) >= 12 and ("\n" in text or "import " in text):
        return text
    return None


_REWRITE_PROMPT = """The following Python hit the Docker sandbox wall-clock limit ({timeout_sec}s, domain={domain}).

Your task: rewrite it to finish reliably inside that budget.
Rules:
- Reduce training: lower n_steps / epochs / iterations by 4–10× (or more if still heavy).
- Prefer larger batch_size when it shortens the loop.
- The sandbox prepends SANDBOX_WALL_SEC and SANDBOX_REMAINING_SEC() — you may call SANDBOX_REMAINING_SEC() in loops and break early if low.
- Keep the same scientific intent (same dataset/model family if possible) but underpowered.
- Still print one JSON object to stdout as the last line (same schema as before if obvious; else {{"sandbox":"ok","note":"simplified_after_timeout",...}}).

Return ONLY the Python source code (no markdown fences, no prose).

--- original code ---
{code}
--- end ---
"""


async def rewrite_sandbox_code_after_timeout(
    llm: LLMClient,
    *,
    session_id: str,
    hypothesis_id: str,
    code: str,
    sandbox_timeout_sec: int,
    domain: str,
) -> str | None:
    if not looks_like_heavy_training_code(code):
        return None
    prompt = _REWRITE_PROMPT.format(
        timeout_sec=int(sandbox_timeout_sec),
        domain=domain or "unknown",
        code=code[:12000],
    )
    try:
        raw = await llm.call(
            prompt=prompt,
            purpose="agent.sandbox_code_timeout_rewrite",
            session_id=session_id,
            hypothesis_id=hypothesis_id,
        )
    except Exception as exc:
        logger.warning("sandbox rewrite LLM call failed: %s", exc)
        return None
    new_code = extract_python_from_llm_response(raw)
    if not new_code or new_code.strip() == code.strip():
        return None
    if len(new_code) > 200_000:
        return None
    return new_code
