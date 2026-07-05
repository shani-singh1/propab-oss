"""BaseExtractor — domain-agnostic extraction over a FullTextDocument.

Extractors know how to find claims, tables, open problems, contradictions,
and gaps in *any* scientific text. They never know what a Sidon set or a
GTEx tissue is — that knowledge lives in the calling pipeline's use of the
domain profile, never inside an extractor.
"""
from __future__ import annotations

from typing import Any

from services.literature.app.models import FullTextDocument


class BaseExtractor:
    name: str = "base"

    async def extract(self, doc: FullTextDocument) -> list[Any]:
        raise NotImplementedError
