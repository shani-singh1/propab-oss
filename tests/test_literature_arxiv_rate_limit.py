from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from services.orchestrator import literature as lit


class ArxivRateLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_on_429_then_succeeds(self) -> None:
        ok = MagicMock()
        ok.status_code = 200
        ok.text = """<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'></feed>"""
        ok.raise_for_status = MagicMock()

        blocked = MagicMock()
        blocked.status_code = 429
        blocked.headers = {"Retry-After": "0.01"}

        client = AsyncMock()
        client.get = AsyncMock(side_effect=[blocked, ok])

        with patch.object(lit, "_arxiv_throttle", new_callable=AsyncMock):
            resp = await lit._arxiv_http_get(client, "http://example", params={"q": "x"})
        self.assertIsNotNone(resp)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(client.get.await_count, 2)


if __name__ == "__main__":
    unittest.main()
