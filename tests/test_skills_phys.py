"""Domain-dependent research skills for the physical/graph domains
(graph_invariants, network_diffusion, materials).

These tests assert that each domain's authored skills load under the right scope
and phase, stay isolated from other domains, and encode the guardrails the real
verifiers depend on (no tautological invariant pairs / rediscovery for graphs,
cross-topology holdout + shuffle null for diffusion, crystal-system LOFO + leakage
guard for materials). Content lives in
``propab/skills/domains/<domain>/*.skill.md``; only skill files were added, so the
domain-general loader and its tests are untouched.
"""
from __future__ import annotations

import pytest

from propab.skills import Skill, load_skills

PHYS_DOMAINS = ("graph_invariants", "network_diffusion", "materials")


def _domain_skills(domain: str, phase: str) -> list[Skill]:
    return [s for s in load_skills(domain, phase) if s.scope == domain]


@pytest.mark.parametrize("domain", PHYS_DOMAINS)
def test_domain_has_two_or_three_skills(domain: str) -> None:
    names: set[str] = set()
    for phase in ("hypothesis", "experiment"):
        for s in _domain_skills(domain, phase):
            assert isinstance(s, Skill)
            assert s.scope == domain
            assert s.description, f"{s.name} missing a description"
            assert s.body.strip(), f"{s.name} has an empty body"
            names.add(s.name)
    assert 2 <= len(names) <= 3, f"{domain} should author 2-3 skills, got {sorted(names)}"


@pytest.mark.parametrize("domain", PHYS_DOMAINS)
def test_domain_covers_hypothesis_and_experiment(domain: str) -> None:
    # Task: phase mostly hypothesis, some experiment. Each domain must have both.
    assert _domain_skills(domain, "hypothesis"), f"{domain} has no hypothesis skill"
    assert _domain_skills(domain, "experiment"), f"{domain} has no experiment skill"


@pytest.mark.parametrize("domain", PHYS_DOMAINS)
def test_domain_skills_do_not_leak_into_other_domains(domain: str) -> None:
    others = [d for d in PHYS_DOMAINS if d != domain] + ["math_combinatorics", "genomics"]
    for other in others:
        loaded = load_skills(other, "hypothesis") + load_skills(other, "experiment")
        assert not any(s.scope == domain for s in loaded), (
            f"{domain} skills leaked into {other}"
        )


def _blob(domain: str) -> str:
    parts: list[str] = []
    for phase in ("hypothesis", "experiment"):
        for s in _domain_skills(domain, phase):
            parts.append((s.body + " " + s.description).lower())
    return "\n".join(parts)


def test_graph_invariants_guards_tautology_and_rediscovery() -> None:
    blob = _blob("graph_invariants")
    # DOM2b independence must be respected: modularity/clustering not a tautology.
    assert "tautolog" in blob
    assert "modularity" in blob and "clustering" in blob
    # Spectral vocabulary the verifier actually exposes.
    assert "algebraic connectivity" in blob or "algebraic_connectivity" in blob
    assert "spectral gap" in blob or "spectral_gap" in blob
    assert "newman" in blob  # anti-rediscovery of the textbook correlation
    # Cross-network-family generalization is the verifier's decision procedure.
    assert "held-out" in blob or "holdout" in blob or "hold out" in blob


def test_network_diffusion_guards_threshold_and_simulator() -> None:
    blob = _blob("network_diffusion")
    # Epidemic-threshold moment the adapter exposes as k2_over_k1.
    assert "k^2" in blob or "k2_over_k1" in blob or "⟨k²⟩" in blob
    # IC vs SIR robustness, and the outcome-shuffle null.
    assert "cascade" in blob and "sir" in blob
    assert "shuffle" in blob
    assert "rediscover" in blob or "rediscovery" in blob or "threshold" in blob
    # Cross-topology-family replication requirement.
    assert "topology" in blob and ("held-out" in blob or "hold out" in blob or "holdout" in blob)


def test_materials_guards_lofo_leakage_and_baseline() -> None:
    blob = _blob("materials")
    # Leave-one-crystal-system-out is the verifier's holdout.
    assert "lofo" in blob or "leave-one-crystal-system-out" in blob
    assert "crystal system" in blob or "crystal-system" in blob
    # The leakage guard: space_group_number proxies crystal-system identity.
    assert "space_group_number" in blob or "space group" in blob
    assert "leakage" in blob
    # Anti-rediscovery: beat the Matbench baseline, not reproduce it.
    assert "matbench" in blob and ("baseline" in blob or "rediscover" in blob)


@pytest.mark.parametrize("domain", PHYS_DOMAINS)
def test_domain_skills_ordered_after_core(domain: str) -> None:
    # The loader orders core methodology before domain technique; a domain skill
    # must never sort ahead of the core skills for the same phase.
    for phase in ("hypothesis", "experiment"):
        skills = load_skills(domain, phase)
        scopes = [s.scope for s in skills]
        if domain in scopes and "core" in scopes:
            assert scopes.index("core") < scopes.index(domain)
