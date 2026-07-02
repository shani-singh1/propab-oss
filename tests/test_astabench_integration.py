"""Tests for AstaBench answer extraction."""
from integrations.astabench.answer_extract import extract_discoverybench_answer


def test_abstain_when_no_strong_belief():
    payload = {
        "campaign": {
            "status": "budget_exhausted",
            "stop_reason": "TIME_BUDGET_EXHAUSTED",
            "belief_state": {
                "active_beliefs": [
                    {"statement": "Maybe X causes Y", "confidence": "unclear", "supporting_nodes": []}
                ]
            },
            "hypothesis_tree": {"nodes": {}},
        },
        "summary": {"total_hypotheses": 10, "total_confirmed": 0},
    }
    answer, audit = extract_discoverybench_answer(payload)
    assert "Insufficient evidence" in answer["hypothesis"]
    assert audit["source"] == "abstain"


def test_abstain_when_strong_belief_but_no_confirmed_nodes():
    """fixes.md Step 2: do not submit ungrounded strong beliefs."""
    payload = {
        "campaign": {
            "belief_state": {
                "active_beliefs": [
                    {
                        "statement": "Higher BMI increases diabetes risk among adults.",
                        "confidence": "strong",
                        "supporting_nodes": [],
                    }
                ]
            },
            "hypothesis_tree": {"nodes": {}},
        },
        "summary": {"total_hypotheses": 20, "total_confirmed": 0},
    }
    answer, audit = extract_discoverybench_answer(payload)
    assert "Insufficient evidence" in answer["hypothesis"]
    assert audit["source"] == "abstain"
    assert audit["abstain_reason"] == "confirmed_nodes_zero_despite_strong_beliefs"


def test_confirmed_finding_preferred_over_strong_belief():
    payload = {
        "campaign": {
            "belief_state": {
                "active_beliefs": [
                    {
                        "statement": "Wrong narrative belief.",
                        "confidence": "strong",
                        "supporting_nodes": [],
                    }
                ]
            },
            "hypothesis_tree": {
                "nodes": {
                    "n1": {
                        "verdict": "confirmed",
                        "text": "Time preference associates with BMI in 1989 cohort.",
                        "key_finding": "Positive association between savings proxy and BMI.",
                        "confidence": 0.9,
                    }
                }
            },
        },
        "summary": {"total_confirmed": 1},
    }
    answer, audit = extract_discoverybench_answer(payload)
    assert "Positive association" in answer["hypothesis"]
    assert audit["source"] == "confirmed_finding"
