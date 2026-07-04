"""Concurrent lifetime claim writes (T1-001 exit criteria)."""
from __future__ import annotations

import threading
import uuid

import pytest

from propab import config
from propab.knowledge_graph import Claim, KnowledgeGraph, new_id
from propab.lifetime_postgres import lifetime_postgres_enabled, upsert_claim


def _postgres_available() -> bool:
    if not lifetime_postgres_enabled():
        return False
    try:
        from propab.lifetime_postgres import get_engine

        with get_engine().connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _postgres_available(), reason="postgres lifetime store not configured")
def test_concurrent_add_claim_both_persist(monkeypatch):
    """Two threads upsert different claims simultaneously — no data loss."""
    monkeypatch.setattr(config.settings, "lifetime_store_backend", "postgres")
    campaign_id = str(uuid.uuid4())
    domain = "test_concurrent"
    barrier = threading.Barrier(2)
    errors: list[str] = []

    def worker(text: str) -> None:
        try:
            barrier.wait(timeout=5)
            claim = Claim(
                id=new_id("kg"),
                text=text,
                verdict="confirmed",
                theme="test",
                confidence=0.9,
                campaign_id=campaign_id,
            )
            upsert_claim(claim, domain=domain)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    t1 = threading.Thread(target=worker, args=("Concurrent claim alpha",))
    t2 = threading.Thread(target=worker, args=("Concurrent claim beta",))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert not errors, errors
    graph = KnowledgeGraph.load()
    texts = {c.text for c in graph.claims.values() if c.campaign_id == campaign_id}
    assert "Concurrent claim alpha" in texts
    assert "Concurrent claim beta" in texts


def test_save_skips_json_file_when_postgres_backend(monkeypatch, tmp_path):
    """T1-001: postgres mode must not rewrite JSON archives."""
    from propab.knowledge_graph import knowledge_store_path

    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))
    monkeypatch.setattr(config.settings, "lifetime_store_backend", "postgres")

    called = {"postgres": False}

    def _fake_save(graph: KnowledgeGraph) -> None:
        called["postgres"] = True

    monkeypatch.setattr("propab.lifetime_postgres.save_knowledge_graph", _fake_save)

    graph = KnowledgeGraph()
    graph.add_claim(
        Claim(
            id=new_id("kg"),
            text="No JSON write test",
            verdict="confirmed",
            theme="test",
            campaign_id=str(uuid.uuid4()),
        )
    )
    graph.save()
    assert called["postgres"]
    assert not knowledge_store_path().is_file()
