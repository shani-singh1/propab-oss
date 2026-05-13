"""Guards against false breakthroughs from implausible sandbox accuracy."""

from propab.campaign import BreakthroughCriteria, ResearchCampaign


def test_breakthrough_rejects_implausible_val_accuracy() -> None:
    bc = BreakthroughCriteria.default_accuracy()
    bc.baseline_value = 0.63
    finding = {
        "confidence": 0.9,
        "replication_count": 3,
        "metric_value": 0.9981,
    }
    assert bc.is_breakthrough(finding) is False


def test_breakthrough_accepts_accuracy_just_below_plausibility_cap() -> None:
    bc = BreakthroughCriteria.default_accuracy()
    bc.baseline_value = 0.5
    finding = {
        "confidence": 0.9,
        "replication_count": 3,
        "metric_value": 0.974,
    }
    assert bc.is_breakthrough(finding) is True


def test_breakthrough_rejects_accuracy_at_plausibility_cap() -> None:
    bc = BreakthroughCriteria.default_accuracy()
    bc.baseline_value = 0.5
    finding = {
        "confidence": 0.9,
        "replication_count": 3,
        "metric_value": 0.975,
    }
    assert bc.is_breakthrough(finding) is False


def test_update_best_metric_ignores_implausible_accuracy() -> None:
    c = ResearchCampaign(id="00000000-0000-0000-0000-000000000099", question="q")
    c.breakthrough_criteria.baseline_value = 0.5
    assert c.update_best_metric({"metric_value": 0.9981}) is False
    assert c.best_metric == 0.0


def test_update_best_metric_rejects_at_plausibility_cap() -> None:
    c = ResearchCampaign(id="00000000-0000-0000-0000-000000000098", question="q")
    c.breakthrough_criteria.baseline_value = 0.5
    assert c.update_best_metric({"metric_value": 0.975}) is False
    assert c.best_metric == 0.0
