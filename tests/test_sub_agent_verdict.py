from services.worker.sub_agent_loop import _hypothesis_relevance_score


def test_hypothesis_relevance_score_high_overlap() -> None:
    hyp = "activation function affects transformer training stability"
    outs = [{"summary": "activation function improves transformer stability", "significant": True}]
    s = _hypothesis_relevance_score(hyp, outs)
    assert s >= 0.08


def test_hypothesis_relevance_score_low_overlap() -> None:
    hyp = "activation function affects transformer training stability"
    outs = [{"summary": "protein folding benchmark metadata", "count": 3}]
    s = _hypothesis_relevance_score(hyp, outs)
    assert s < 0.08
