from services.worker.domain_router import coerce_routed_domain


def test_coerce_primary_domains_passthrough() -> None:
    assert coerce_routed_domain("deep_learning") == "deep_learning"
    assert coerce_routed_domain("ML_RESEARCH") == "ml_research"


def test_coerce_legacy_maps() -> None:
    assert coerce_routed_domain("ml_modeling") == "deep_learning"
    assert coerce_routed_domain("chemistry") == "general_computation"


def test_coerce_unknown_defaults() -> None:
    assert coerce_routed_domain("not_a_real_domain_xyz") == "general_computation"
