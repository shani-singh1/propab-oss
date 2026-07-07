"""Issue O1: breakthrough-metric extraction must be domain-general.

The orchestrator's ``_extract_primary_metric_from_worker_result`` should prefer the
campaign's DECLARED ``breakthrough_criteria.metric_name`` and pull THAT key from the
worker result — including common nesting — instead of hardcoding ML accuracy names.
It must fail closed (return ``None``) for a declared non-accuracy metric that is
absent, and must never substitute an accuracy value for a differently-named metric.
The existing ML ``val_accuracy`` path must keep working unchanged.
"""

from propab.campaign import BreakthroughCriteria
from services.orchestrator.campaign_loop import _extract_primary_metric_from_worker_result


# ── Non-ML declared metric: extracted from the declared key ───────────────────

def test_non_ml_metric_top_level_declared_key() -> None:
    result = {"mean_r2": 0.42, "confidence": 0.9}
    assert _extract_primary_metric_from_worker_result(result, "mean_r2") == 0.42


def test_non_ml_metric_generic_value_name_pair() -> None:
    # Worker emits the generic metric_value + metric_name carrier.
    result = {"metric_value": 137.0, "metric_name": "crossing_n"}
    assert _extract_primary_metric_from_worker_result(result, "crossing_n") == 137.0


def test_non_ml_metric_in_common_subdict() -> None:
    result = {"scores": {"mean_r2": 0.61}, "verdict": "confirmed"}
    assert _extract_primary_metric_from_worker_result(result, "mean_r2") == 0.61


def test_non_ml_metric_drives_is_breakthrough() -> None:
    # End-to-end: extract a non-ML metric and feed it through is_breakthrough.
    bc = BreakthroughCriteria(
        metric_name="mean_r2",
        baseline_value=0.40,
        improvement_threshold=0.05,
        direction="higher_is_better",
    )
    result = {"metric_value": 0.60, "metric_name": "mean_r2", "confidence": 0.9}
    extracted = _extract_primary_metric_from_worker_result(result, "mean_r2")
    assert extracted == 0.60
    finding = {
        "confidence": 0.9,
        "replication_count": 3,
        "metric_value": extracted,
    }
    assert bc.is_breakthrough(finding) is True


def test_non_ml_metric_below_threshold_no_breakthrough() -> None:
    bc = BreakthroughCriteria(
        metric_name="mean_r2",
        baseline_value=0.40,
        improvement_threshold=0.05,
        direction="higher_is_better",
    )
    result = {"mean_r2": 0.41, "confidence": 0.9}
    extracted = _extract_primary_metric_from_worker_result(result, "mean_r2")
    assert extracted == 0.41
    finding = {"confidence": 0.9, "replication_count": 3, "metric_value": extracted}
    assert bc.is_breakthrough(finding) is False


# ── Fail closed: declared metric absent → None, no bogus substitution ─────────

def test_missing_non_ml_metric_returns_none() -> None:
    # Declared metric is crossing_n but the result carries only an accuracy — the
    # extractor must NOT substitute the accuracy value for a non-accuracy metric.
    result = {"val_accuracy": 0.97, "accuracy": 0.97}
    assert _extract_primary_metric_from_worker_result(result, "crossing_n") is None


def test_metric_value_pair_wrong_name_not_substituted() -> None:
    # metric_value present but labelled as a DIFFERENT metric than the declared one.
    result = {"metric_value": 0.97, "metric_name": "val_accuracy"}
    assert _extract_primary_metric_from_worker_result(result, "mean_r2") is None


def test_empty_result_returns_none() -> None:
    assert _extract_primary_metric_from_worker_result({}, "mean_r2") is None


# ── Existing ML val_accuracy path: no regression ──────────────────────────────

def test_ml_val_accuracy_declared_key() -> None:
    result = {"val_accuracy": 0.83, "confidence": 0.9}
    assert _extract_primary_metric_from_worker_result(result, "val_accuracy") == 0.83


def test_ml_val_accuracy_percentage_normalized() -> None:
    # 0–100 percentage reporting is rescaled to a fraction for accuracy metrics.
    result = {"val_accuracy": 83.0}
    assert _extract_primary_metric_from_worker_result(result, "val_accuracy") == 0.83


def test_ml_accuracy_family_fallback_still_works() -> None:
    # Declared metric is accuracy-family; result only carries a bare `accuracy`.
    result = {"accuracy": 0.71}
    assert _extract_primary_metric_from_worker_result(result, "val_accuracy") == 0.71


def test_ml_deep_nested_accuracy_still_found() -> None:
    result = {"evidence": {"scores": {"val_accuracy": 0.66}}}
    assert _extract_primary_metric_from_worker_result(result, "val_accuracy") == 0.66


def test_ml_metric_value_carrier_for_accuracy() -> None:
    # Generic metric_value carrier with no metric_name, declared metric is accuracy.
    result = {"metric_value": 0.79}
    assert _extract_primary_metric_from_worker_result(result, "val_accuracy") == 0.79


def test_ml_val_accuracy_drives_is_breakthrough() -> None:
    bc = BreakthroughCriteria.default_accuracy()
    bc.baseline_value = 0.60
    result = {"val_accuracy": 0.80, "confidence": 0.9}
    extracted = _extract_primary_metric_from_worker_result(result, "val_accuracy")
    assert extracted == 0.80
    finding = {"confidence": 0.9, "replication_count": 3, "metric_value": extracted}
    assert bc.is_breakthrough(finding) is True
