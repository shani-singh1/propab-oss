from __future__ import annotations

from typing import Any

import httpx


async def embed_texts(*, texts: list[str], api_key: str, model: str) -> list[list[float]]:
    if not api_key or not texts:
        return []
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {"model": model, "input": texts}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()["data"]
    return [row["embedding"] for row in sorted(data, key=lambda x: x["index"])]
