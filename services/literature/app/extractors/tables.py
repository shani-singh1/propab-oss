"""
Tabulated-values extractor — the most precise known values in a paper live
in its tables, not its theorems. A theorem might say ``F(n) ~ sqrt(n)``;
Table A.1 in the appendix might have F(n) computed exactly for n up to
10,000. That table is what the novelty checker needs.

Covers both formal ``\\begin{tabular}`` environments (already isolated by the
arXiv source's LaTeX parser into ``doc.tables_raw``, including appendix and
supplementary tables) and enumerated lists of values in free text that were
never wrapped in a table environment at all — "For n = 1, 2, 4, ... the
maximum Sidon set sizes are 1, 2, 3, ..." is a tabulation whether or not
LaTeX calls it one.
"""
from __future__ import annotations

import re

from services.literature.app.extractors.base import BaseExtractor
from services.literature.app.models import FullTextDocument, TabulatedSequence

_CELL_CLEAN_RE = re.compile(r"\\(?:hline|toprule|midrule|bottomrule|cline\{[^}]*\})")
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})?")
_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:e-?\d+)?$")
_ENUM_LIST_RE = re.compile(
    r"for\s+([a-zA-Z])\s*=\s*([\d,\s]+?)\s*,?\s*(?:the\s+)?([A-Za-z()\s]{3,60}?)\s+"
    r"(?:is|are)\s*:?\s*([\d,\s]+?)(?:[.\n]|$)",
    re.IGNORECASE,
)


def _clean_cell(cell: str) -> str:
    cell = _CELL_CLEAN_RE.sub("", cell)
    cell = _LATEX_CMD_RE.sub("", cell)
    return cell.strip().strip("$").strip()


def _parse_tabular(raw: str) -> tuple[list[str], list[list[str]]]:
    # Strip the column-spec argument (e.g. {c|c|c}) that precedes the body.
    body = re.sub(r"^\{[^}]*\}", "", raw.strip(), count=1)
    rows_raw = [r for r in re.split(r"\\\\", body) if r.strip()]
    rows: list[list[str]] = []
    for r in rows_raw:
        cells = [_clean_cell(c) for c in r.split("&")]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)
    if not rows:
        return [], []
    header = rows[0]
    header_is_numeric = all(_NUMERIC_RE.match(c) for c in header if c)
    if header_is_numeric or len(rows) == 1:
        return [], rows
    return header, rows[1:]


def _to_number(s: str) -> float | int | str:
    s = s.strip()
    if _NUMERIC_RE.match(s):
        return float(s) if "." in s or "e" in s.lower() else int(s)
    return s


class TablesExtractor(BaseExtractor):
    name = "tables"

    async def extract(self, doc: FullTextDocument) -> list[TabulatedSequence]:
        out: list[TabulatedSequence] = []

        for table in doc.tables_raw:
            raw = table.get("raw", "")
            location = table.get("location", "table")
            header, data_rows = _parse_tabular(raw)
            if not data_rows or len(data_rows[0]) < 2:
                continue
            index_header = header[0] if header else "n"
            value_header = header[1] if len(header) > 1 else "value"
            values: dict[str, object] = {}
            for row in data_rows:
                if len(row) < 2:
                    continue
                idx_raw, val_raw = row[0], row[1]
                idx = _to_number(idx_raw)
                values[str(idx)] = _to_number(val_raw)
            if not values:
                continue
            numeric_indices = [float(k) for k in values if _NUMERIC_RE.match(k)]
            expected_rows = raw.count("\\\\")
            confidence = min(1.0, len(values) / expected_rows) if expected_rows else 1.0
            out.append(
                TabulatedSequence(
                    description=f"Table ({location}): {value_header} by {index_header}",
                    index_variable=index_header,
                    value_variable=value_header,
                    values=values,
                    max_index=max(numeric_indices) if numeric_indices else None,
                    min_index=min(numeric_indices) if numeric_indices else None,
                    source_doi=doc.doi,
                    source_title=doc.title,
                    source_year=doc.year,
                    location=location,
                    is_in_appendix="appendix" in location.lower(),
                    is_in_supplementary="supplementary" in location.lower(),
                    extraction_confidence=round(confidence, 3),
                )
            )

        # Enumerated lists in free text, not wrapped in any table environment.
        for m in _ENUM_LIST_RE.finditer(doc.body_text or ""):
            index_var, idx_list_raw, value_desc, val_list_raw = m.groups()
            idx_list = [t.strip() for t in idx_list_raw.split(",") if t.strip()]
            val_list = [t.strip() for t in val_list_raw.split(",") if t.strip()]
            if len(idx_list) < 3 or len(idx_list) != len(val_list):
                continue
            values = {i: _to_number(v) for i, v in zip(idx_list, val_list)}
            numeric_indices = [float(i) for i in idx_list if _NUMERIC_RE.match(i)]
            out.append(
                TabulatedSequence(
                    description=f"Enumerated in text: {value_desc.strip()}",
                    index_variable=index_var,
                    value_variable=value_desc.strip(),
                    values=values,
                    max_index=max(numeric_indices) if numeric_indices else None,
                    min_index=min(numeric_indices) if numeric_indices else None,
                    source_doi=doc.doi,
                    source_title=doc.title,
                    source_year=doc.year,
                    location="body (enumerated list)",
                    extraction_confidence=1.0,
                )
            )

        return out
