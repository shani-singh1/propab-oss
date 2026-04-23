from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.events import EventEmitter
from propab.types import EventType


class LLMClient:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        emitter: EventEmitter,
        session_factory: async_sessionmaker,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.emitter = emitter
        self.session_factory = session_factory

    async def call(
        self,
        *,
        prompt: str,
        purpose: str,
        session_id: str,
        hypothesis_id: str | None = None,
    ) -> str:
        started = time.perf_counter()
        await self.emitter.emit(
            session_id=session_id,
            event_type=EventType.LLM_PROMPT,
            step=f"llm.{purpose}",
            payload={"prompt": prompt, "purpose": purpose, "model": self.model},
            hypothesis_id=hypothesis_id,
        )

        response_text = await self._call_provider(prompt)

        await self.emitter.emit(
            session_id=session_id,
            event_type=EventType.LLM_RESPONSE,
            step=f"llm.{purpose}",
            payload={"response": response_text, "purpose": purpose, "model": self.model},
            hypothesis_id=hypothesis_id,
        )

        duration_ms = int((time.perf_counter() - started) * 1000)
        await self._persist_call(
            session_id=session_id,
            hypothesis_id=hypothesis_id,
            purpose=purpose,
            prompt=prompt,
            response=response_text,
            duration_ms=duration_ms,
        )
        return response_text

    async def _call_provider(self, prompt: str) -> str:
        if self.provider != "openai" or not self.api_key:
            return json.dumps(
                [
                    {
                        "id": "h1",
                        "text": "A controlled intervention can improve measurable outcome quality.",
                        "test_methodology": "Run baseline versus intervention and compare primary metrics.",
                        "gap_reference": "No indexed literature yet",
                        "expected_result": "Intervention outperforms baseline by statistically significant margin.",
                    }
                ]
            )

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=40) as client:
            response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"]

    async def _persist_call(
        self,
        *,
        session_id: str,
        hypothesis_id: str | None,
        purpose: str,
        prompt: str,
        response: str,
        duration_ms: int,
    ) -> None:
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO llm_calls (
                        id, session_id, hypothesis_id, call_purpose, model, prompt_text, response_text, input_tokens, output_tokens, duration_ms
                    )
                    VALUES (
                        :id, :session_id, :hypothesis_id, :call_purpose, :model, :prompt_text, :response_text, :input_tokens, :output_tokens, :duration_ms
                    )
                    """
                ),
                {
                    "id": str(uuid4()),
                    "session_id": session_id,
                    "hypothesis_id": hypothesis_id,
                    "call_purpose": purpose,
                    "model": self.model,
                    "prompt_text": prompt,
                    "response_text": response,
                    "input_tokens": None,
                    "output_tokens": None,
                    "duration_ms": duration_ms,
                },
            )
            await session.commit()
