from __future__ import annotations

import unittest

from services.orchestrator.prior_builder import _extract_json_object


class PriorBuilderJsonTests(unittest.TestCase):
    def test_extract_plain_json(self) -> None:
        self.assertEqual(_extract_json_object('{"a": 1}'), {"a": 1})

    def test_extract_with_markdown_fence_stripped_manually(self) -> None:
        raw = 'Here is JSON:\n{"x": true}\n'
        self.assertEqual(_extract_json_object(raw), {"x": True})


if __name__ == "__main__":
    unittest.main()
