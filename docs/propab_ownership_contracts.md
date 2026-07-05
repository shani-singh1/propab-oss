# Propab Ownership Contracts

This document catalogs the input/output contract and ownership boundary of
each major Propab component: what a component owns, what it explicitly does
not own, and the health metric that determines whether it can be trusted.
The purpose is to keep domain knowledge, campaign state, and verification
logic from leaking across component boundaries — each piece of Propab
should be replaceable without the others needing to change.

Sections are added by whichever agent/owner builds the corresponding
component. This file currently documents the Literature Intelligence
Service in full; other components (orchestrator, worker, domain plugins,
API) should have their contracts added here by their owners as they land.

---

## Literature Intelligence Service

**Location:** `services/literature/`

**Owns:** structured knowledge about what is known, contested, and unknown
in any research domain, sourced from published papers and authoritative
sources (arXiv, OEIS, Semantic Scholar, MathOverflow/StackExchange, zbMATH,
PubMed, bioRxiv, Crossref).

**Never owns:** hypotheses, verification of claims, campaign state, or
domain knowledge about which specific papers, sequences, or classification
codes matter for a domain — that belongs entirely to the domain plugin's
`literature_profile()` (see `packages/propab-core/propab/domain_modules/base.py`).
The service reads whatever profile it is given and never hardcodes anything
domain-specific; the architectural test for this is: adding a new domain
requires only implementing `literature_profile()`, zero changes to this
service's code.

**Input contract:**
- A research question (string)
- A domain identifier (string)
- A literature profile from the domain plugin (`literature_profile()` output —
  seed papers, search terms, source priorities, classification codes,
  open-problem sources, tabulation sources, canonical surveys, novelty criteria)
- Optional: depth (`"standard"` | `"deep"` | `"exhaustive"`)

**Output contract** (`POST /prior`):
- `established_facts` — attributed claims with verbatim quotes and citations
- `open_gaps` — explicitly unresolved questions with best-known partial results
- `contradictions` — conflicting claims between sources with citations
- `dead_ends` — approaches tried and ruled out
- `tabulated_values` — structured tabulations of known computed values
- `novelty_bar` — what a finding would need to show to be non-trivial
- Every claim has at least one real, verifiable citation (title + authors +
  year + DOI/arXiv id/URL minimum), and the `verbatim` field is the exact
  source text — never paraphrased, never summarized. A citation whose
  verbatim doesn't actually appear in the source is a fabricated citation,
  not an approximate one.

**Other endpoints:** `POST /novelty` (known/novel/uncertain verdict for a
candidate finding), `POST /gaps` (ranked frontier of open, computationally-
approachable problems), `POST /ingest` (manual single-paper ingestion),
`GET /coverage`, `GET /health`.

**Health metric:** Citation verification rate — the fraction of claims in
`established_facts` whose cited source, when re-fetched, actually contains
the attributed verbatim quote. Target: ≥ 90%. Below 80%: the service is
fabricating citations and its output must not be used for campaign launch
or novelty checking. Measured offline by `evaluator/metrics.py` (spot-check
re-fetch, not computed on every `/health` call — re-fetching sources is slow
and would make health checks flaky under source rate limits); the cached
result is what `/health`'s `citation_verification_rate` field reports.

First live measurement (2026-07-05, math_combinatorics domain, n=30 sampled
claims from a real `/prior` run against arXiv/OEIS/Semantic Scholar/
MathOverflow/zbMATH): **100%** verified. See
`services/literature/CHANGELOG.md` and `artifacts/literature_coverage.json`
for the full record; this single run is a first data point, not a
statistically robust estimate — treat it as "the citation-fidelity pipeline
works end-to-end against real sources," not as a permanent guarantee.

**Integration with the rest of Propab:** optional and additive. A campaign
without `literature_service_url` configured (`packages/propab-core/propab/config.py`)
falls back to the domain plugin's own `literature_prior()`/`literature_profile()
== {}` keyword-search path — nothing breaks if this service is down or unset.
Wiring the actual call sites in `services/orchestrator/campaign_loop.py` and
the verdict pipeline is owned by the orchestrator, not by this service (this
service's ownership boundary stops at its own HTTP contract); the contract
above is what that integration should call.
