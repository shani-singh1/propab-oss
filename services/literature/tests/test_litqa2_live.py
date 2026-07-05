import random
from unittest.mock import patch

import httpx
import pytest

from services.literature.app.evaluator.litqa2_live import (
    UNSURE_ANSWER,
    build_choices,
    build_prompt,
    format_choices,
    gemini_generate,
    judge_evidence_sufficiency,
    parse_answer_letter,
    reformulate_query_for_search,
    score_results,
)


class TestBuildChoices:
    def test_includes_ideal_distractors_and_unsure(self):
        choices, target_idx, unsure_idx = build_choices("A", ["B", "C"], random.Random(0))
        assert set(choices) == {"A", "B", "C", UNSURE_ANSWER}
        assert choices[target_idx] == "A"
        assert choices[unsure_idx] == UNSURE_ANSWER

    def test_deterministic_with_same_seed(self):
        c1 = build_choices("A", ["B", "C", "D"], random.Random(42))
        c2 = build_choices("A", ["B", "C", "D"], random.Random(42))
        assert c1 == c2

    def test_different_seeds_can_differ(self):
        results = {tuple(build_choices("A", ["B", "C", "D", "E"], random.Random(s))[0]) for s in range(10)}
        assert len(results) > 1


class TestFormatChoices:
    def test_letters_in_order(self):
        out = format_choices(["first", "second", "third"])
        assert out == "A. first\nB. second\nC. third"


class TestBuildPrompt:
    def test_includes_question_choices_and_evidence(self):
        prompt = build_prompt("What is X?", ["Y", "Z"], ["Evidence one.", "Evidence two."])
        assert "What is X?" in prompt
        assert "A. Y" in prompt
        assert "B. Z" in prompt
        assert "[1] Evidence one." in prompt
        assert "[2] Evidence two." in prompt

    def test_handles_no_evidence(self):
        prompt = build_prompt("Q?", ["Y", "Z"], [])
        assert "no evidence retrieved" in prompt


class TestParseAnswerLetter:
    def test_json_format(self):
        assert parse_answer_letter('{"answer": "B"}', n_choices=4) == 1

    def test_json_format_with_extra_text(self):
        assert parse_answer_letter('Sure, my answer is {"answer": "C"} based on the evidence.', n_choices=4) == 2

    def test_labbench_tag_format(self):
        assert parse_answer_letter("[ANSWER]D[/ANSWER]", n_choices=4) == 3

    def test_bare_letter(self):
        assert parse_answer_letter("A", n_choices=4) == 0
        assert parse_answer_letter("(A)", n_choices=4) == 0

    def test_out_of_range_letter_returns_none(self):
        assert parse_answer_letter('{"answer": "Z"}', n_choices=3) is None

    def test_unparseable_returns_none(self):
        assert parse_answer_letter("I cannot determine this from the evidence provided.", n_choices=4) is None


class TestScoreResults:
    def test_empty_results(self):
        assert score_results([]) == {"n": 0, "accuracy": None, "coverage": None, "precision": None}

    def test_all_correct_and_sure(self):
        results = [{"is_correct": True, "is_sure": True} for _ in range(5)]
        scores = score_results(results)
        assert scores == {"n": 5, "accuracy": 1.0, "coverage": 1.0, "precision": 1.0}

    def test_mixed_results_matches_labbench_semantics(self):
        results = [
            {"is_correct": True, "is_sure": True},   # correct, sure
            {"is_correct": False, "is_sure": True},  # wrong, sure
            {"is_correct": False, "is_sure": False},  # picked "insufficient info"
            {"is_correct": False, "is_sure": False},  # unparseable
        ]
        scores = score_results(results)
        assert scores["n"] == 4
        assert scores["accuracy"] == 0.25  # 1/4 correct overall
        assert scores["coverage"] == 0.5  # 2/4 gave a sure (non-abstaining) answer
        assert scores["precision"] == 0.5  # 1/2 correct among the sure answers

    def test_unparseable_answer_never_counts_as_correct_or_sure(self):
        # Mirrors asta-bench's score_litqa2: no choice marked -> is_correct=False, is_sure=False.
        results = [{"is_correct": False, "is_sure": False}]
        scores = score_results(results)
        assert scores["accuracy"] == 0.0
        assert scores["coverage"] == 0.0
        assert scores["precision"] == 0.0  # 0 correct out of max(1, 0) sure


class TestReformulateQueryForSearch:
    @pytest.mark.asyncio
    async def test_returns_targeted_queries_from_llm_json(self):
        async def fake_gemini_generate(prompt, **kwargs):
            return '{"queries": ["\\"MLH1dn\\" prime editing", "PE2 mismatch repair", "MLH1 dominant negative"]}'

        with patch(
            "services.literature.app.evaluator.litqa2_live.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            queries = await reformulate_query_for_search(
                "By what factor does MLH1dn expression increase editing efficiency of PE2?",
                api_key="k", model="m",
            )

        assert queries == ['"MLH1dn" prime editing', "PE2 mismatch repair", "MLH1 dominant negative"]

    @pytest.mark.asyncio
    async def test_includes_choices_in_prompt_when_given(self):
        captured_prompt = {}

        async def fake_gemini_generate(prompt, **kwargs):
            captured_prompt["value"] = prompt
            return '{"queries": ["q1"]}'

        with patch(
            "services.literature.app.evaluator.litqa2_live.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            await reformulate_query_for_search(
                "What factor increases efficiency?",
                api_key="k", model="m",
                choices=["2.7 fold", "1.5 fold", "Insufficient information to answer the question"],
            )

        assert "2.7 fold" in captured_prompt["value"]
        assert "1.5 fold" in captured_prompt["value"]

    @pytest.mark.asyncio
    async def test_no_choices_block_when_choices_omitted(self):
        captured_prompt = {}

        async def fake_gemini_generate(prompt, **kwargs):
            captured_prompt["value"] = prompt
            return '{"queries": ["q1"]}'

        with patch(
            "services.literature.app.evaluator.litqa2_live.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            await reformulate_query_for_search("What factor increases efficiency?", api_key="k", model="m")

        assert "Answer choices" not in captured_prompt["value"]

    @pytest.mark.asyncio
    async def test_falls_back_to_stopword_strip_on_llm_failure(self):
        async def failing_gemini_generate(prompt, **kwargs):
            raise RuntimeError("simulated failure")

        with patch(
            "services.literature.app.evaluator.litqa2_live.gemini_generate",
            side_effect=failing_gemini_generate,
        ):
            terms = await reformulate_query_for_search("What is the effect of MLH1dn on editing?", api_key="k", model="m")

        assert len(terms) == 1
        assert "MLH1dn" in terms[0]
        assert "what" not in terms[0].lower().split()

    @pytest.mark.asyncio
    async def test_falls_back_when_llm_output_unparseable(self):
        async def fake_gemini_generate(prompt, **kwargs):
            return "I don't understand the request."

        with patch(
            "services.literature.app.evaluator.litqa2_live.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            terms = await reformulate_query_for_search("Does gene X affect protein Y?", api_key="k", model="m")

        assert isinstance(terms, list)
        assert terms  # non-empty fallback


class TestJudgeEvidenceSufficiency:
    @pytest.mark.asyncio
    async def test_parses_sufficient_true(self):
        async def fake_gemini_generate(prompt, **kwargs):
            return '{"sufficient": true, "follow_up_query": ""}'

        with patch(
            "services.literature.app.evaluator.litqa2_live.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            result = await judge_evidence_sufficiency("Q?", ["A", "B"], ["some evidence"], api_key="k", model="m")

        assert result == {"sufficient": True, "follow_up_query": ""}

    @pytest.mark.asyncio
    async def test_parses_insufficient_with_follow_up(self):
        async def fake_gemini_generate(prompt, **kwargs):
            return '{"sufficient": false, "follow_up_query": "MLH1dn dominant negative mismatch repair"}'

        with patch(
            "services.literature.app.evaluator.litqa2_live.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            result = await judge_evidence_sufficiency("Q?", ["A", "B"], ["unrelated evidence"], api_key="k", model="m")

        assert result["sufficient"] is False
        assert result["follow_up_query"] == "MLH1dn dominant negative mismatch repair"

    @pytest.mark.asyncio
    async def test_defaults_to_sufficient_on_llm_failure(self):
        async def failing_gemini_generate(prompt, **kwargs):
            raise RuntimeError("simulated failure")

        with patch(
            "services.literature.app.evaluator.litqa2_live.gemini_generate",
            side_effect=failing_gemini_generate,
        ):
            result = await judge_evidence_sufficiency("Q?", ["A", "B"], [], api_key="k", model="m")

        # Failing safe means "stop looping and answer with what we have,"
        # not "sufficient=True is somehow correct" — defaulting True here
        # prevents an infinite/wasted extra round on a broken judgment call.
        assert result["sufficient"] is True

    @pytest.mark.asyncio
    async def test_unparseable_response_defaults_to_sufficient(self):
        async def fake_gemini_generate(prompt, **kwargs):
            return "I'm not sure how to answer that."

        with patch(
            "services.literature.app.evaluator.litqa2_live.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            result = await judge_evidence_sufficiency("Q?", ["A", "B"], ["x"], api_key="k", model="m")

        assert result["sufficient"] is True
        assert result["follow_up_query"] == ""

    @pytest.mark.asyncio
    async def test_handles_no_evidence_yet(self):
        async def fake_gemini_generate(prompt, **kwargs):
            assert "(none yet)" in prompt
            return '{"sufficient": false, "follow_up_query": "some query"}'

        with patch(
            "services.literature.app.evaluator.litqa2_live.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            result = await judge_evidence_sufficiency("Q?", ["A", "B"], [], api_key="k", model="m")

        assert result["follow_up_query"] == "some query"
