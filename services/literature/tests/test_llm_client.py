from unittest.mock import patch

import httpx
import pytest

from services.literature.app.llm_client import gemini_generate


class TestGeminiGenerate:
    @pytest.mark.asyncio
    async def test_extracts_text_from_candidates(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["key"] == "test-key"
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": '{"answer": "A"}'}]}}]},
            )

        real_client = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_client(*args, **kwargs)

        with patch("services.literature.app.llm_client.httpx.AsyncClient", side_effect=factory):
            result = await gemini_generate("prompt", api_key="test-key", model="gemini-3-flash-preview")

        assert result == '{"answer": "A"}'

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty_string(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"candidates": []})

        real_client = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_client(*args, **kwargs)

        with patch("services.literature.app.llm_client.httpx.AsyncClient", side_effect=factory):
            result = await gemini_generate("prompt", api_key="test-key", model="gemini-3-flash-preview")

        assert result == ""

    @pytest.mark.asyncio
    async def test_retries_on_503_then_succeeds(self):
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(503)
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

        real_client = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_client(*args, **kwargs)

        with patch("services.literature.app.llm_client.httpx.AsyncClient", side_effect=factory), patch(
            "services.literature.app.llm_client.asyncio.sleep", return_value=None
        ):
            result = await gemini_generate("prompt", api_key="test-key", model="gemini-3-flash-preview")

        assert calls["n"] == 2
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_timeout_then_raises_after_exhausting(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("simulated timeout", request=request)

        real_client = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_client(*args, **kwargs)

        with patch("services.literature.app.llm_client.httpx.AsyncClient", side_effect=factory), patch(
            "services.literature.app.llm_client.asyncio.sleep", return_value=None
        ):
            with pytest.raises(httpx.ReadTimeout):
                await gemini_generate("prompt", api_key="test-key", model="gemini-3-flash-preview", max_retries=2)
