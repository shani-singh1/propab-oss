from unittest.mock import patch

import httpx
import pytest

from services.literature.app.indexer.embeddings import EmbeddingClient, _normalize_google_model


class TestProviderSelection:
    def test_google_selected_when_key_present(self):
        client = EmbeddingClient(provider="google", google_api_key="k", dim=8)
        assert client.provider == "google"

    def test_gemini_alias_selects_google(self):
        client = EmbeddingClient(provider="gemini", google_api_key="k", dim=8)
        assert client.provider == "google"

    def test_google_without_key_falls_back_to_offline(self):
        client = EmbeddingClient(provider="google", google_api_key="", dim=8)
        assert client.provider == "offline"

    def test_openai_without_key_falls_back_to_offline(self):
        client = EmbeddingClient(provider="openai", api_key="", dim=8)
        assert client.provider == "offline"

    def test_normalize_google_model_strips_models_prefix(self):
        assert _normalize_google_model("models/gemini-embedding-2") == "gemini-embedding-2"
        assert _normalize_google_model("gemini-embedding-2") == "gemini-embedding-2"
        assert _normalize_google_model("") == "gemini-embedding-2"


_RealAsyncClient = httpx.AsyncClient


def _mock_google_client(handler):
    """Return a stand-in for httpx.AsyncClient(...) that routes requests
    through ``handler`` instead of the network. Must use the real
    AsyncClient class captured above, not ``httpx.AsyncClient`` — that
    attribute is what gets patched, so calling through it here would recurse
    into the mock instead of constructing a real (mock-transport-backed) client."""

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _RealAsyncClient(*args, **kwargs)

    return factory


class TestGoogleEmbed:
    @pytest.mark.asyncio
    async def test_embed_calls_batch_endpoint_with_expected_shape(self):
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["body"] = request.content
            # One "embeddings" entry per request in the batch, in order.
            return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2, 0.3]}, {"values": [0.4, 0.5, 0.6]}]})

        client = EmbeddingClient(provider="google", google_api_key="test-key", model="gemini-embedding-2", dim=3)
        with patch(
            "services.literature.app.indexer.embeddings.httpx.AsyncClient",
            side_effect=_mock_google_client(handler),
        ):
            result = await client.embed(["hello", "world"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert captured["headers"]["x-goog-api-key"] == "test-key"
        assert "gemini-embedding-2:batchEmbedContents" in captured["url"]
        assert b'"requests"' in captured["body"]
        assert b'"models/gemini-embedding-2"' in captured["body"]

    @pytest.mark.asyncio
    async def test_embed_retries_on_429_then_succeeds(self):
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(429)
            return httpx.Response(200, json={"embeddings": [{"values": [1.0, 0.0]}]})

        client = EmbeddingClient(provider="google", google_api_key="test-key", dim=2)
        with patch(
            "services.literature.app.indexer.embeddings.httpx.AsyncClient",
            side_effect=_mock_google_client(handler),
        ), patch("services.literature.app.indexer.embeddings.asyncio.sleep", return_value=None):
            result = await client.embed(["retry me"])

        assert calls["n"] == 2
        assert result == [[1.0, 0.0]]

    @pytest.mark.asyncio
    async def test_embed_falls_back_to_offline_on_persistent_error(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        client = EmbeddingClient(provider="google", google_api_key="bad-key", dim=8)
        with patch(
            "services.literature.app.indexer.embeddings.httpx.AsyncClient",
            side_effect=_mock_google_client(handler),
        ):
            result = await client.embed(["hello world"])

        assert len(result) == 1
        assert len(result[0]) == 8  # offline fallback vector of the configured dim
