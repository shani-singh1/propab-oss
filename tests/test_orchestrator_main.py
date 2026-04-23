def test_orchestrator_health_callable() -> None:
    from services.orchestrator.main import health

    assert health() == {"status": "ok", "service": "orchestrator"}


def test_orchestrator_app_metadata() -> None:
    from services.orchestrator.main import app

    assert "Orchestrator" in app.title
