"""oeis_lookup — the best-known reference. Network is mocked so tests are deterministic;
the honesty invariant is that a network failure degrades to 'unavailable' (never a
fabricated match) and a real response is parsed faithfully.
"""
from __future__ import annotations

import propab.tools.mathematics.oeis_lookup as mod
from propab.tools.mathematics.oeis_lookup import oeis_lookup

# The real OEIS fmt=json API returns a JSON LIST of sequence dicts directly (verified
# live against oeis.org) — NOT a {"results": [...]} wrapper. The mock must match reality.
_CANNED = [
    {"number": 79, "name": "Powers of 2", "data": "1,2,4,8,16,32,64",
     "offset": "0,2", "keyword": "nonn,easy"},
    {"number": 3022, "name": "Length of shortest (or optimal) Golomb ruler with n marks.",
     "data": "0,1,3,6,11,17,25,34,44,55,72,85,106,127", "offset": "1,3", "keyword": "nonn,hard,more"},
]


def test_parses_real_response(monkeypatch):
    monkeypatch.setattr(mod, "_fetch_oeis", lambda q, t: _CANNED)
    r = oeis_lookup(terms=[1, 2, 4, 8, 16])
    assert r.success and r.output["status"] == "ok"
    top = r.output["results"][0]
    assert top["anum"] == "A000079" and top["terms"][:3] == [1, 2, 4]
    # the 'more'/'hard' Golomb entry is surfaced (that's where new terms would be novel)
    assert any(x["anum"] == "A003022" and "more" in (x["keyword"] or "") for x in r.output["results"])


def test_network_failure_degrades_not_fabricates(monkeypatch):
    def _boom(q, t):
        raise TimeoutError("oeis slow")
    monkeypatch.setattr(mod, "_fetch_oeis", _boom)
    r = oeis_lookup(terms=[1, 2, 4, 8])
    assert r.success and r.output["status"] == "unavailable" and r.output["results"] == []


def test_no_match_is_not_found(monkeypatch):
    monkeypatch.setattr(mod, "_fetch_oeis", lambda q, t: [])  # OEIS returns [] for no match
    r = oeis_lookup(terms=[7, 13, 999999])
    assert r.success and r.output["status"] == "not_found"


def test_requires_terms_or_anum():
    assert not oeis_lookup().success


def test_anum_query(monkeypatch):
    seen = {}
    monkeypatch.setattr(mod, "_fetch_oeis", lambda q, t: seen.update(q=q) or _CANNED)
    oeis_lookup(anum="A003022")
    assert seen["q"] == "A003022"
