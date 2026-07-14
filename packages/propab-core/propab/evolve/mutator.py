"""Evolve — the mutation operator (WS-1): the LLM seam.

The LLM is a COMMODITY here. It does exactly one job: read the best programs found so far, with the
scores the VERIFIER gave them, and write a new program. It is not the source of the result — the
verifier is. That is why FunSearch beat SOTA on cap sets with a non-frontier model, why the client is
injected (a cheap model is the default; a frontier model is an optional ceiling-raiser, never a
prerequisite), and why credit for a result accrues to the engine rather than to whoever's model we
rent this month.

Two rules are absolute.

**mutate() must never raise.** A completion is adversarial by accident: the model will return prose,
half a function, a markdown essay, a syntax error, an empty string, or nothing at all, and the client
itself will time out and 500. Every one of those is a NORMAL event in a ten-thousand-step run, not an
error path to bail on. On any failure we return the no-op program: it emits no candidates, the
verifier scores it -inf, and it is evicted. One bad completion must never kill a run.

**The verifier is the sole authority on score.** Nothing a program says about itself is read — not
its stdout, not a "score" it returns, not a comment claiming optimality. The prompt therefore never
asks a program to report its own score, and this module never parses one out of a completion. The
only numbers shown to the model are the ones `Problem.verify` computed. A program that could talk its
way to a high score would make the whole ledger worthless.
"""
from __future__ import annotations

import ast
import logging
import re
import textwrap
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .island import FAMILY_KEY, UNKNOWN_FAMILY
from .problem import Problem
from .program import ENTRYPOINT, PROGRAM_CONTRACT, Program

logger = logging.getLogger(__name__)

#: Emits no candidates => `Engine.evaluate` scores it INVALID (-inf) => it is evicted. Harmless by
#: design, and identical every time, so the island dedupes the whole failure mode into one member.
NOOP_CODE = f"def {ENTRYPOINT}():\n    return []\n"

#: Keep the prompt bounded: one runaway parent must not blow the context window.
MAX_PARENT_CHARS = 12_000   # a family-sweep parent runs ~5k chars; 4k silently decapitated build()
#: A "program" larger than this is not a program, it is a data dump.
MAX_CODE_CHARS = 200_000
#: How many parents to show. Few-shot on winners; more than a handful just dilutes the signal.
DEFAULT_MAX_PARENTS = 3

_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FAMILY_RE = re.compile(r"^[#\s]*family[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_FAMILY_SLUG_RE = re.compile(r"[^a-z0-9 /+-]+")
MAX_FAMILY_CHARS = 40


@runtime_checkable
class LLMClient(Protocol):
    """The model seam: prompt in, completion out. WS-6 supplies the real router.

    Deliberately this small so tests can pass a lambda and production can pass a retrying,
    rate-limited, multi-provider client without this module knowing the difference.
    """

    def complete(self, prompt: str) -> str: ...


Completer = Callable[[str], str]


def _as_completer(llm: LLMClient | Completer) -> Completer:
    """Accept either a bare callable or anything with `.complete(prompt)`."""
    complete = getattr(llm, "complete", None)
    if callable(complete):
        return complete
    if callable(llm):
        return llm
    raise TypeError(
        f"llm must be callable or expose .complete(prompt) -> str, got {type(llm).__name__}"
    )


def normalize_family(raw: object) -> str:
    """Squash a model-supplied family tag into a stable, comparable slug.

    Grouping is by mathematical IDEA, so the tag is free text ("algebraic / cyclic", "concatenation",
    "shortening a known code") rather than a fixed enum — the engine cannot know a domain's idea space
    in advance, and a fixed enum would just push everything into "other". Normalizing is what stops
    "Concatenation", "concatenation.", and "CONCATENATION " from counting as three separate ideas.
    """
    if not isinstance(raw, str):
        return UNKNOWN_FAMILY
    family = _FAMILY_SLUG_RE.sub("", raw.strip().lower().rstrip(".")).strip()
    family = re.sub(r"\s+", " ", family)
    if not family:
        return UNKNOWN_FAMILY
    return family[:MAX_FAMILY_CHARS]


def extract_family(completion: str, code: str) -> str:
    """Pull the approach-family tag out of the completion (or the code's `# family:` header)."""
    for text in (code, completion):
        match = _FAMILY_RE.search(text or "")
        if match:
            family = normalize_family(match.group(1))
            if family != UNKNOWN_FAMILY:
                return family
    return UNKNOWN_FAMILY


def _is_runnable_program(code: str) -> bool:
    """Does this source define a top-level `build()`? Top-level because that is what the runner calls;
    a `build` nested inside a class or another function is not an entry point."""
    if not code.strip() or len(code) > MAX_CODE_CHARS:
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        # ValueError covers source with null bytes; the rest are what a pathological completion does.
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == ENTRYPOINT
        for node in tree.body
    )


def extract_program(completion: object) -> str | None:
    """Pull runnable `build()` source out of a raw completion. None when there isn't any.

    Tries each fenced block in order, then the bare completion (models often answer with code and no
    fence). Anything that does not parse, or does not define `build()`, is not a program.
    """
    if not isinstance(completion, str) or not completion.strip():
        return None
    blocks = [m.group(1) for m in _FENCE_RE.finditer(completion)]
    blocks.append(completion)  # last resort: the model answered with bare code
    for block in blocks:
        code = textwrap.dedent(block).strip("\n")
        if _is_runnable_program(code):
            return code
    return None


class LLMMutator:
    """Implements the `Mutator` protocol (engine.py): parents + problem -> a new Program.

    `set_exploration_hint` is an optional extra capability, not part of the protocol: the engine calls
    it (guarded by hasattr) to inject "family X has taken over the search, go somewhere else" before a
    mutation. The frozen `mutate(parents, problem)` signature has nowhere to put that, and threading a
    registry through it would have meant changing the contract.
    """

    def __init__(
        self,
        llm: LLMClient | Completer,
        *,
        max_parents: int = DEFAULT_MAX_PARENTS,
        max_parent_chars: int = MAX_PARENT_CHARS,
    ) -> None:
        self._complete = _as_completer(llm)
        self.max_parents = max(1, max_parents)
        self.max_parent_chars = max_parent_chars
        self._hint: str | None = None
        #: Bad completions are expected; a SPIKE in them is a bug (wrong model, broken key, …).
        self.failures = 0
        self.calls = 0

    # ---------------------------------------------------------------- prompt

    def set_exploration_hint(self, hint: str | None) -> None:
        """Steer the next mutation away from a crowded or circular approach family."""
        self._hint = hint

    def build_prompt(self, parents: list[Program], problem: Problem) -> str:
        shown = self._parents_to_show(parents)
        parts = [
            problem.describe().strip(),
            PROGRAM_CONTRACT.strip(),
            (
                "Tag your program with the mathematical idea it uses, as the FIRST line:\n"
                "    # family: <short name of the approach, e.g. 'concatenation'>\n"
                "Programs are grouped by that idea so the search can tell whether it is genuinely "
                "exploring or just re-wording one approach. Tag by IDEA, not by wording."
            ),
            (
                "Do NOT print, return, or claim a score. An independent verifier computes the score "
                "from what build() returns; anything the program says about itself is ignored."
            ),
        ]

        if shown:
            parts.append(
                "Programs found so far, worst to best. Every score below was computed by the "
                "verifier. Work out what makes the better ones better."
            )
            for i, program in enumerate(shown):
                parts.append(
                    f"--- program v{i} "
                    f"(verifier score: {_fmt_score(program.score)}, "
                    f"family: {program.detail.get(FAMILY_KEY, UNKNOWN_FAMILY)}) ---\n"
                    f"{self._clip(program.code)}"
                )
            parts.append(
                f"Write program v{len(shown)}: a new `{ENTRYPOINT}()` that the verifier will score "
                f"HIGHER than every program above. Change the ALGORITHM — do not just tweak "
                f"constants. Return one Python code block and nothing else."
            )
        else:
            parts.append(
                f"No working programs yet. Write the first `{ENTRYPOINT}()`. Return one Python code "
                f"block and nothing else."
            )

        if self._hint:
            parts.append(self._hint)
        return "\n\n".join(parts)

    def _parents_to_show(self, parents: list[Program]) -> list[Program]:
        """The best parents, worst-first so the model sees an improving trajectory to continue.

        Parents that never produced a valid candidate are dropped: showing the model broken code as
        an exemplar teaches it to write broken code. If that leaves nothing, we ask for a program
        from scratch rather than few-shotting on junk.
        """
        viable = [p for p in parents if isinstance(p.score, (int, float)) and p.score > float("-inf")]
        viable.sort(key=lambda p: p.score)
        return viable[-self.max_parents :]

    def _clip(self, code: str) -> str:
        """Clip a parent to fit the prompt — but NEVER at the cost of `build()`.

        Clipping from the front silently decapitates the program: our targets put a long prelude of
        shared helpers first and `build()` last, so a head-clip removes the one thing the model is
        supposed to imitate. That is not hypothetical — it happened. The family-sweep seed is ~5.2k
        chars against a 4k cap, so the model was shown a pile of helpers with the sweep cut off, and
        duly wrote short point-constructions (measured: 4 candidates/program from a 400-candidate
        parent). In program evolution the parent IS the instruction; truncating it truncates the
        instruction.

        So: keep the TAIL (which contains `build()`), and drop from the middle instead.
        """
        if len(code) <= self.max_parent_chars:
            return code

        marker = f"def {ENTRYPOINT}("
        cut = code.rfind(marker)
        if cut == -1:
            # No entrypoint to protect — fall back to a head clip.
            return code[: self.max_parent_chars] + "\n# ... truncated ...\n"

        tail = code[cut:]
        if len(tail) >= self.max_parent_chars:
            # build() alone overflows: keep its head, and say so.
            return tail[: self.max_parent_chars] + "\n# ... truncated ...\n"

        head_budget = self.max_parent_chars - len(tail) - 40
        head = code[:head_budget] if head_budget > 0 else ""
        return f"{head}\n# ... helper definitions elided ...\n\n{tail}"

    # ---------------------------------------------------------------- mutation

    def mutate(self, parents: list[Program], problem: Problem) -> Program:
        """Never raises. On any failure — client, prompt, or completion — returns the no-op program."""
        parents = list(parents or [])
        self.calls += 1
        try:
            prompt = self.build_prompt(parents, problem)
            completion = self._complete(prompt)
        except Exception as exc:  # noqa: BLE001 — a dead client must not kill the run
            logger.debug("evolve.mutator: LLM call failed: %s: %s", type(exc).__name__, exc)
            return self._noop(parents, f"llm_error: {type(exc).__name__}: {exc}")

        code = extract_program(completion)
        if code is None:
            logger.debug("evolve.mutator: no runnable build() in completion")
            return self._noop(parents, "no_program_in_completion")

        family = extract_family(completion if isinstance(completion, str) else "", code)
        return self._child(code, parents, {FAMILY_KEY: family})

    def _child(self, code: str, parents: list[Program], detail: dict[str, object]) -> Program:
        return Program(
            code=code,
            generation=1 + max((p.generation for p in parents), default=0),
            island=parents[0].island if parents else 0,
            parents=[p.id for p in parents],
            detail=detail,
        )

    def _noop(self, parents: list[Program], reason: str) -> Program:
        self.failures += 1
        return self._child(
            NOOP_CODE, parents, {FAMILY_KEY: UNKNOWN_FAMILY, "mutator_failure": reason}
        )


def _fmt_score(score: float) -> str:
    try:
        return f"{float(score):.6g}"
    except (TypeError, ValueError):  # pragma: no cover — defensive
        return "unscored"
