from services.orchestrator.hypothesis_ranking import composite_score


def test_composite_score_weights() -> None:
    assert composite_score(1.0, 1.0, 1.0, 1.0) == 1.0
    s = composite_score(0.0, 0.0, 0.0, 0.0)
    assert s == 0.0
    mid = composite_score(0.5, 0.5, 0.5, 0.5)
    assert abs(mid - 0.5) < 1e-6
