import io
import json
import tarfile

import httpx
import pytest

from services.literature.app.models import RawDocument
from services.literature.app.sources._latex import flatten_inputs, parse_latex_document
from services.literature.app.sources.arxiv import ArxivSource, normalize_arxiv_id
from services.literature.app.sources.oeis import OeisSource

ARXIV_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>A Paper About Sidon Sets</title>
    <summary>We study Sidon sets and prove new bounds.</summary>
    <published>2024-01-05T00:00:00Z</published>
    <author><name>Alice Author</name></author>
    <author><name>Bob Author</name></author>
    <arxiv:doi>10.1000/xyz123</arxiv:doi>
  </entry>
</feed>
"""

SAMPLE_TEX = r"""
\documentclass{article}
\begin{document}
\section{Introduction}
We study Sidon sets.\footnote{A Sidon set has all pairwise sums distinct.}
\begin{theorem}
For every prime $p$, $F(p) \geq \sqrt{p} - 1$.
\end{theorem}
\begin{proof}
This gives us that $F(p) > p^{1/2} - 2$ by direct construction.
\end{proof}
\appendix
\section{Extra results}
\begin{table}
\caption{Values of $F(n)$ for small $n$.}
\begin{tabular}{c|c}
n & F(n) \\
1 & 1 \\
2 & 2 \\
3 & 3 \\
\end{tabular}
\end{table}
\end{document}
"""


def make_source(cache_dir) -> ArxivSource:
    return ArxivSource(cache_dir=str(cache_dir), max_results=10, min_interval_sec=0.0)


class TestLatexParsing:
    def test_parses_theorem_and_proof_and_table(self):
        parsed = parse_latex_document(SAMPLE_TEX)
        envs = {e["env"] for e in parsed.latex_environments}
        assert "theorem" in envs
        assert "proof" in envs
        assert len(parsed.tables_raw) == 1
        assert "F(n)" in parsed.tables_raw[0]["raw"]
        assert len(parsed.footnotes) == 1
        assert "pairwise sums" in parsed.footnotes[0]
        assert len(parsed.captions) == 1

    def test_appendix_environments_are_labeled(self):
        parsed = parse_latex_document(SAMPLE_TEX)
        table_loc = parsed.tables_raw[0]["location"]
        assert "appendix" in table_loc.lower()

    def test_flatten_inputs_substitutes_referenced_file(self):
        main = r"\documentclass{article}\begin{document}\input{extra}\end{document}"
        files = {"extra.tex": r"\begin{lemma}Extra content.\end{lemma}"}
        flattened = flatten_inputs(main, files)
        assert "Extra content" in flattened

    def test_nested_environment_of_same_name_is_balanced(self):
        tex = r"\begin{theorem}Outer \begin{theorem}Inner\end{theorem} tail\end{theorem}"
        parsed = parse_latex_document(tex)
        assert len(parsed.latex_environments) == 1
        assert "Inner" in parsed.latex_environments[0]["content"]
        assert "tail" in parsed.latex_environments[0]["content"]


class TestNormalizeArxivId:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("http://arxiv.org/abs/2401.00001v1", "2401.00001"),
            ("2401.00001", "2401.00001"),
            ("math.CO/0601001", "math.CO/0601001"),
        ],
    )
    def test_various_formats(self, raw, expected):
        assert normalize_arxiv_id(raw) == expected


class TestArxivSource:
    @pytest.mark.asyncio
    async def test_search_parses_atom_feed(self, tmp_path):
        source = make_source(tmp_path)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=ARXIV_ATOM_FEED)

        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        results = await source.search("sidon sets", {"search_terms": [], "classification_codes": {}})
        assert len(results) == 1
        assert results[0].external_id == "2401.00001"
        assert results[0].authors == "Alice Author, Bob Author"
        assert results[0].year == 2024

    @pytest.mark.asyncio
    async def test_fetch_full_text_extracts_latex_and_caches(self, tmp_path):
        source = make_source(tmp_path)
        tar_bytes = io.BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
            data = SAMPLE_TEX.encode("utf-8")
            info = tarfile.TarInfo(name="main.tex")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_bytes.seek(0)
        eprint_body = tar_bytes.read()

        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            assert "e-print" in str(request.url)
            return httpx.Response(200, content=eprint_body)

        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        doc = RawDocument(source="arxiv", external_id="2401.00001", title="T", authors="A", year=2024)
        full = await source.fetch_full_text(doc)
        assert full.extraction_method == "latex"
        assert any(e["env"] == "theorem" for e in full.latex_environments)
        assert calls["n"] == 1

        # Second fetch must hit the cache, not the network.
        full2 = await source.fetch_full_text(doc)
        assert calls["n"] == 1
        assert full2.extraction_method == "latex"

    @pytest.mark.asyncio
    async def test_falls_back_to_abstract_when_no_eprint_available(self, tmp_path):
        source = make_source(tmp_path)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        doc = RawDocument(source="arxiv", external_id="9999.99999", title="T", authors="A", year=2024, abstract="An abstract.")
        full = await source.fetch_full_text(doc)
        assert full.extraction_method == "abstract_only"
        assert full.body_text == "An abstract."


class TestOeisSource:
    @pytest.mark.asyncio
    async def test_check_tabulated_exact_index_match(self):
        source = OeisSource(min_interval_sec=0.0)
        source._sequence_cache["A000001"] = {
            "name": "Test sequence", "data": "1,1,1,2,1,5,2,15", "offset": "1,1",
        }
        matches = await source.check_tabulated({"index": 4, "value": 2, "candidate_sequence_ids": ["A000001"]})
        assert len(matches) == 1
        assert matches[0].matched is True
        assert matches[0].matched_value == 2

    @pytest.mark.asyncio
    async def test_check_tabulated_value_not_present(self):
        source = OeisSource(min_interval_sec=0.0)
        source._sequence_cache["A000001"] = {"name": "x", "data": "1,1,1,2", "offset": "1,1"}
        matches = await source.check_tabulated({"index": 100, "value": 999, "candidate_sequence_ids": ["A000001"]})
        assert all(not m.matched for m in matches) or matches == []

    @pytest.mark.asyncio
    async def test_search_parses_bare_list_response(self):
        # OEIS returns a bare JSON list on success, not {"results": [...]}.
        source = OeisSource(min_interval_sec=0.0)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[{"number": 3023, "name": "Test seq", "data": "1,2,3"}])

        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        results = await source.search("test", {})
        assert len(results) == 1
        assert results[0].external_id == "A003023"

    @pytest.mark.asyncio
    async def test_search_handles_null_response_for_no_matches(self):
        # OEIS returns the literal JSON null when there are no matches.
        source = OeisSource(min_interval_sec=0.0)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=None)

        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        results = await source.search("nomatch", {})
        assert results == []

    @pytest.mark.asyncio
    async def test_warm_cache_parses_bare_list_response(self):
        source = OeisSource(min_interval_sec=0.0)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[{"number": 3023, "name": "Test seq", "data": "1,2,3", "offset": "1,1"}])

        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await source.warm_cache(["A003023"])
        assert "A003023" in source._sequence_cache
        assert source._sequence_cache["A003023"]["data"] == "1,2,3"
