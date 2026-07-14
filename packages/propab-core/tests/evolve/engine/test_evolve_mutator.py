"""Mutator: prompt construction, code extraction, and the never-raise guarantee."""
from __future__ import annotations

import pytest
from conftest import CrashingLLM, HillClimbLLM, ScriptedLLM, SumProblem

from propab.evolve.island import FAMILY_KEY, UNKNOWN_FAMILY
from propab.evolve.mutator import (
    NOOP_CODE,
    LLMMutator,
    extract_family,
    extract_program,
    normalize_family,
)
from propab.evolve.program import ENTRYPOINT, Program


def parent(code: str, score: float, *, family: str = "algebraic") -> Program:
    return Program(code=code, score=score, valid=True, detail={FAMILY_KEY: family})


# --------------------------------------------------------------------------- prompt


def test_prompt_carries_the_spec_the_contract_and_the_winners():
    llm = ScriptedLLM("")
    mutator = LLMMutator(llm)
    problem = SumProblem()
    parents = [parent("def build():\n    return [(1, 1, 1, 1)]\n", 4.0)]

    prompt = mutator.build_prompt(parents, problem)

    assert problem.describe() in prompt          # what are we optimizing
    assert f"def {ENTRYPOINT}():" in prompt      # PROGRAM_CONTRACT
    assert "return [(1, 1, 1, 1)]" in prompt     # few-shot on the winner
    assert "4" in prompt                         # ...with its verifier score


def test_parents_are_shown_worst_to_best():
    """The model should see an improving trajectory to continue, not a random pile."""
    mutator = LLMMutator(ScriptedLLM(""))
    prompt = mutator.build_prompt(
        [parent("# lo\ndef build(): pass\n", 1.0), parent("# hi\ndef build(): pass\n", 9.0)],
        SumProblem(),
    )
    assert prompt.index("# lo") < prompt.index("# hi")


def test_prompt_forbids_self_reported_scores():
    """The verifier is the sole authority. A program that could talk its way to a score would make
    the ledger worthless, so the prompt must never invite one."""
    mutator = LLMMutator(ScriptedLLM(""))
    prompt = mutator.build_prompt([], SumProblem())
    assert "Do NOT print, return, or claim a score" in prompt
    assert "verifier computes the score" in prompt


def test_prompt_asks_for_a_family_tag():
    mutator = LLMMutator(ScriptedLLM(""))
    assert "# family:" in mutator.build_prompt([], SumProblem())


def test_invalid_parents_are_not_used_as_exemplars():
    """Few-shotting on broken code teaches the model to write broken code."""
    mutator = LLMMutator(ScriptedLLM(""))
    junk = Program(code="# junk\ndef build():\n    return []\n", score=float("-inf"))
    prompt = mutator.build_prompt([junk], SumProblem())
    assert "# junk" not in prompt
    assert "No working programs yet" in prompt


def test_only_the_best_parents_are_shown():
    mutator = LLMMutator(ScriptedLLM(""), max_parents=2)
    parents = [parent(f"# p{i}\ndef build(): pass\n", float(i)) for i in range(5)]
    prompt = mutator.build_prompt(parents, SumProblem())
    assert "# p4" in prompt and "# p3" in prompt
    assert "# p0" not in prompt


def test_long_parents_are_clipped():
    mutator = LLMMutator(ScriptedLLM(""), max_parent_chars=50)
    prompt = mutator.build_prompt([parent("# " + "x" * 5000 + "\ndef build(): pass\n", 1.0)], SumProblem())
    assert "truncated" in prompt
    assert len(prompt) < 2000


def test_exploration_hint_is_injected():
    mutator = LLMMutator(ScriptedLLM(""))
    mutator.set_exploration_hint("DIVERSITY WARNING: 'algebraic' has taken over.")
    assert "DIVERSITY WARNING" in mutator.build_prompt([], SumProblem())


# --------------------------------------------------------------------------- extraction


def test_extracts_fenced_python():
    code = extract_program("Sure!\n\n```python\ndef build():\n    return [1]\n```\nHope that helps.")
    assert code == "def build():\n    return [1]"


def test_extracts_bare_code_without_a_fence():
    assert extract_program("def build():\n    return [1]\n") is not None


def test_skips_a_fenced_block_that_is_not_a_program():
    completion = "```\nnot python at all $$$\n```\n```python\ndef build():\n    return [2]\n```"
    assert extract_program(completion) == "def build():\n    return [2]"


def test_dedents_an_indented_block():
    assert extract_program("```python\n    def build():\n        return [1]\n```") is not None


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "   ",
        None,
        42,
        "I'm sorry, I can't help with that.",
        "```python\ndef build(:\n```",                      # syntax error
        "```python\ndef helper():\n    return 1\n```",      # no entry point
        "```python\nclass C:\n    def build(self):\n        return [1]\n```",  # not top-level
    ],
)
def test_junk_completions_yield_no_program(completion):
    assert extract_program(completion) is None


def test_absurdly_large_completions_are_rejected():
    assert extract_program("def build():\n    return [" + "1," * 200_000 + "]") is None


# --------------------------------------------------------------------------- family tags


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Concatenation", "concatenation"),
        ("  ALGEBRAIC / cyclic  ", "algebraic / cyclic"),
        ("shortening a known code.", "shortening a known code"),
        ("", UNKNOWN_FAMILY),
        (None, UNKNOWN_FAMILY),
        ("!!!", UNKNOWN_FAMILY),
        ("x" * 100, "x" * 40),
    ],
)
def test_normalize_family(raw, expected):
    assert normalize_family(raw) == expected


def test_family_is_read_from_the_code_header():
    code = "# family: concatenation\ndef build():\n    return [1]\n"
    assert extract_family("", code) == "concatenation"


def test_family_falls_back_to_the_completion_prose():
    assert extract_family("family: algebraic\n```python\ndef build(): pass\n```", "def build(): pass") == "algebraic"


def test_untagged_program_is_unknown():
    assert extract_family("here", "def build():\n    return [1]\n") == UNKNOWN_FAMILY


def test_mutate_tags_the_child_with_its_family():
    mutator = LLMMutator(HillClimbLLM())
    child = mutator.mutate([], SumProblem())
    assert child.detail[FAMILY_KEY] == "hill-climb"


# --------------------------------------------------------------------------- never raise


@pytest.mark.parametrize(
    "llm",
    [
        CrashingLLM(),                                    # client is down
        ScriptedLLM("no code here, just vibes"),          # model refused
        ScriptedLLM(""),                                  # empty completion
        ScriptedLLM(None),                                # client returned garbage
        ScriptedLLM("```python\ndef build(:\n```"),       # syntax error
    ],
)
def test_mutate_never_raises_and_degrades_to_a_noop(llm):
    mutator = LLMMutator(llm)
    child = mutator.mutate([], SumProblem())

    assert isinstance(child, Program)
    assert child.code == NOOP_CODE   # emits nothing => verifier scores it -inf => it is evicted
    assert mutator.failures == 1


def test_mutate_survives_a_problem_whose_describe_explodes():
    class BrokenProblem(SumProblem):
        def describe(self) -> str:
            raise RuntimeError("spec unavailable")

    mutator = LLMMutator(HillClimbLLM())
    child = mutator.mutate([], BrokenProblem())
    assert child.code == NOOP_CODE


def test_every_failure_produces_the_same_noop_so_the_island_dedupes_it():
    mutator = LLMMutator(CrashingLLM())
    a = mutator.mutate([], SumProblem())
    b = mutator.mutate([], SumProblem())
    assert a.id == b.id


def test_child_records_provenance():
    mutator = LLMMutator(HillClimbLLM())
    parents = [parent("# p\ndef build():\n    return [(1, 1, 1, 1)]\n", 4.0)]
    parents[0].generation = 3
    parents[0].island = 2

    child = mutator.mutate(parents, SumProblem())

    assert child.parents == [parents[0].id]
    assert child.generation == 4
    assert child.island == 2


def test_accepts_a_client_object_with_complete():
    class ClientStyle:
        def complete(self, prompt: str) -> str:
            return "```python\ndef build():\n    return [(9, 9, 9, 9)]\n```"

    child = LLMMutator(ClientStyle()).mutate([], SumProblem())
    assert "(9, 9, 9, 9)" in child.code


def test_rejects_a_client_that_is_neither_callable_nor_a_completer():
    with pytest.raises(TypeError, match="callable"):
        LLMMutator(object())
