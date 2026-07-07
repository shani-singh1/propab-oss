"""Domain-dependent research skills for the three biology domains (genomics,
enzyme_kinetics, mandrake). These assert the authored skills load under the
domain-general loader, carry the correct scope/phase, are grounded in each
domain's REAL verifier (LOFO + label-shuffle null), and never leak across
domains. Core-loader behaviour is covered by tests/test_skills.py; this file
only exercises the bio domain content."""
from __future__ import annotations

import pytest

from propab.skills import available_skill_index, load_skills, skills_prompt_block

BIO_DOMAINS = ("genomics", "enzyme_kinetics", "mandrake")

# Each domain's skills must speak to what its verifier actually computes.
_DOMAIN_GROUNDING = {
    "genomics": ("tissue", "leave-one-tissue-out", "tau", "shuffle"),
    "enzyme_kinetics": ("ec", "kcat", "leave-one-ec-class-out", "shuffle"),
    "mandrake": ("family", "rt", "leave-one-family-out", "shuffle"),
}


@pytest.mark.parametrize("domain", BIO_DOMAINS)
def test_domain_adds_at_least_two_skills(domain):
    generic = load_skills(None, "hypothesis") + load_skills(None, "experiment")
    dom = load_skills(domain, "hypothesis") + load_skills(domain, "experiment")
    dom_only = [s for s in dom if s.scope == domain]
    assert len(dom_only) >= 2, f"{domain} should author >=2 domain skills"
    assert len(dom) > len(generic)


@pytest.mark.parametrize("domain", BIO_DOMAINS)
def test_domain_skills_scoped_to_that_domain(domain):
    for phase in ("hypothesis", "experiment"):
        for s in load_skills(domain, phase):
            assert s.scope in ("core", domain), (domain, s.name, s.scope)


@pytest.mark.parametrize("domain", BIO_DOMAINS)
def test_domain_skills_do_not_leak_into_other_domains(domain):
    for other in BIO_DOMAINS:
        if other == domain:
            continue
        loaded = load_skills(other, "hypothesis") + load_skills(other, "experiment")
        assert not any(s.scope == domain for s in loaded), (domain, other)


@pytest.mark.parametrize("domain", BIO_DOMAINS)
def test_each_domain_has_a_hypothesis_and_an_experiment_skill(domain):
    hyp = [s for s in load_skills(domain, "hypothesis") if s.scope == domain]
    exp = [s for s in load_skills(domain, "experiment") if s.scope == domain]
    assert hyp, f"{domain} needs a hypothesis-phase skill"
    assert exp, f"{domain} needs an experiment-phase skill"


@pytest.mark.parametrize("domain", BIO_DOMAINS)
def test_domain_skills_are_grounded_in_the_real_verifier(domain):
    """A domain's skills, taken together, must reference the LOFO/null vocabulary
    of that domain's real verifier — not a generic, ungrounded methodology."""
    blob = " ".join(
        (s.body + " " + s.description).lower()
        for s in load_skills(domain, "hypothesis") + load_skills(domain, "experiment")
        if s.scope == domain
    )
    for token in _DOMAIN_GROUNDING[domain]:
        assert token in blob, f"{domain} skills missing grounding token {token!r}"


@pytest.mark.parametrize("domain", BIO_DOMAINS)
def test_domain_skills_push_falsifiable_novelty(domain):
    """Every domain skill must push toward a falsifiable, anti-rediscovery test —
    it should mention a null/hold-out AND guard against rediscovery/artifact."""
    for s in load_skills(domain, "hypothesis") + load_skills(domain, "experiment"):
        if s.scope != domain:
            continue
        body = s.body.lower()
        assert any(w in body for w in ("null", "shuffle", "hold-out", "holdout", "lofo", "leave-one")), (
            f"{s.name} names no real test/null"
        )
        assert any(w in body for w in ("rediscover", "artifact", "leakage", "surrogate", "tautolog")), (
            f"{s.name} has no anti-rediscovery / anti-artifact guard"
        )


@pytest.mark.parametrize("domain", BIO_DOMAINS)
def test_domain_block_renders_with_core_first(domain):
    block = skills_prompt_block(domain, "hypothesis")
    assert "RESEARCH METHODOLOGY" in block
    assert f"domain:{domain}" in block
    # core methodology frames the domain technique: the core tag appears first.
    assert block.index("[core]") < block.index(f"domain:{domain}")


def test_all_three_bio_domains_indexed():
    idx = available_skill_index()
    for domain in BIO_DOMAINS:
        assert idx.get(domain), f"{domain} skills should be indexed"
        assert len(idx[domain]) >= 2
