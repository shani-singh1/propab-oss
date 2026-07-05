import httpx
import pytest

from services.literature.app.sources._pmc import fetch_pmc_fulltext

_XML_WITH_TABLE = """<?xml version="1.0"?>
<pmc-articleset><article><body>
<p>This is the main text with enough content to survive the minimum body length filter here easily, since that filter requires at least two hundred characters of real paragraph text before any table content is even considered for inclusion in the result.</p>
<table-wrap>
<caption><p>Table 1: Editing efficiency by construct</p></caption>
<table>
<tr><td>Construct</td><td>Efficiency</td></tr>
<tr><td>PE2</td><td>12%</td></tr>
<tr><td>PE2 + MLH1dn</td><td>32%</td></tr>
</table>
</table-wrap>
</body></article></pmc-articleset>"""


def _mock_client(xml_body: str, status: int = 200) -> httpx.AsyncClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=xml_body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestFetchPmcFulltext:
    @pytest.mark.asyncio
    async def test_extracts_table_into_body_text(self):
        client = _mock_client(_XML_WITH_TABLE)
        result = await fetch_pmc_fulltext(client, "PMC1")

        assert result is not None
        assert "Editing efficiency by construct" in result["body_text"]
        assert "PE2 | 12%" in result["body_text"]
        assert "PE2 + MLH1dn | 32%" in result["body_text"]
        assert len(result["tables"]) == 1

    @pytest.mark.asyncio
    async def test_no_table_wrap_still_works(self):
        xml = '<?xml version="1.0"?><pmc-articleset><article><body><p>' + ("x" * 250) + "</p></body></article></pmc-articleset>"
        client = _mock_client(xml)
        result = await fetch_pmc_fulltext(client, "PMC1")
        assert result is not None
        assert result["tables"] == []

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        client = _mock_client("", status=500)
        result = await fetch_pmc_fulltext(client, "PMC1")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_publisher_restricts_fulltext(self):
        xml = (
            '<?xml version="1.0"?><pmc-articleset><article>'
            "<!--The publisher of this article does not allow downloading of the full text in XML form.-->"
            "</article></pmc-articleset>"
        )
        client = _mock_client(xml)
        result = await fetch_pmc_fulltext(client, "PMC1")
        assert result is None
