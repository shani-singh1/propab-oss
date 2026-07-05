# Propab Literature Intelligence Service

Standalone microservice. Owns: structured knowledge about what is known,
contested, and unknown in a research domain, sourced from published papers
and authoritative sources. Never owns: hypotheses, campaign state, or
domain-specific knowledge about which papers/sequences matter — that lives
in each domain plugin's `literature_profile()`.

Full contract: [`docs/propab_ownership_contracts.md`](../../docs/propab_ownership_contracts.md).
Build spec this service was built against: `agent3.md` (repo root).

## Running locally

```bash
cd services/literature
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8020
```

No external services are required to start: storage backends default to
`postgres_backend=memory` / `qdrant_backend=memory` (in-process, non-persistent).
Embeddings default to `embed_provider=google` / `embed_model=gemini-embedding-2`
— matching the Gemini embeddings already used elsewhere in Propab
(`packages/propab-core/propab/embeddings.py`) — and read the same
**unprefixed** `GOOGLE_API_KEY`/`GEMINI_API_KEY` env var the rest of the repo
uses (not `LITERATURE_GOOGLE_API_KEY`), so a root `.env` with a Gemini key
already configured "just works" with no extra setup. If no Google/OpenAI key
is available, embeddings silently fall back to a deterministic offline
hashing embedding (no network, weaker semantic signal) so the service still
runs and its retrieval/novelty logic is still exercisable with zero
infrastructure at all. Real Postgres/Qdrant are opt-in via env vars for
production; embedding provider is switchable the same way:

```bash
export LITERATURE_POSTGRES_BACKEND=postgres
export LITERATURE_DATABASE_URL=postgresql://propab:propab@localhost:5432/propab_literature
export LITERATURE_QDRANT_BACKEND=qdrant
export LITERATURE_QDRANT_URL=http://localhost:6333

# Embeddings — google/gemini is the default; openai and offline are opt-in:
export LITERATURE_EMBED_PROVIDER=openai
export LITERATURE_OPENAI_API_KEY=sk-...
```

Settings are prefixed `LITERATURE_` **except** `GOOGLE_API_KEY`/`GEMINI_API_KEY`,
which intentionally aren't, so this service shares one key with the rest of
Propab instead of needing its own copy (see `app/config.py` for the full list).

### Docker

```bash
docker build -f services/literature/Dockerfile -t propab-literature .
docker run -p 8020:8020 propab-literature
```

To run alongside the rest of Propab, add a `literature` service block to the
root `docker-compose.yml` (not done by this service, since editing shared
compose files is outside this service's ownership boundary):

```yaml
literature:
  build:
    context: .
    dockerfile: services/literature/Dockerfile
  ports:
    - "8020:8020"
  environment:
    - LITERATURE_POSTGRES_BACKEND=postgres
    - LITERATURE_DATABASE_URL=postgresql://propab:propab@postgres:5432/propab_literature
    - LITERATURE_QDRANT_BACKEND=qdrant
    - LITERATURE_QDRANT_URL=http://qdrant:6333
    - LITERATURE_OPENAI_API_KEY=${OPENAI_API_KEY:-}
```

And set `literature_service_url=http://literature:8020` on the API/orchestrator
services (`LITERATURE_SERVICE_URL` env var maps to `propab.config.Settings.literature_service_url`).

## API

| Route | Method | Purpose |
|---|---|---|
| `/prior` | POST | Established facts, open gaps, contradictions, dead ends, tabulated values, novelty bar for a question + domain |
| `/novelty` | POST | Known / novel / uncertain verdict for a candidate finding |
| `/gaps` | POST | Ranked frontier of open, computationally-approachable problems for a domain |
| `/ingest` | POST | Manually trigger ingestion of one arXiv id or DOI |
| `/coverage` | GET | Papers/claims indexed per domain |
| `/health` | GET | Source reachability + cached citation verification rate |

Request/response schemas: `app/models.py`. See `agent3.md` for the full
contract narrative.

## Architecture

```
app/
├── main.py          FastAPI routes (thin — no business logic)
├── pipeline.py       constructs sources/stores, wires the retriever layer
├── context.py         PipelineContext — the dependency container passed to retriever/*
├── sources/            one file per external source (arXiv, OEIS, Semantic Scholar,
│                       MathOverflow/StackExchange, zbMATH, PubMed, bioRxiv, Crossref)
├── extractors/         claims, tables, open_problems, contradictions, gaps — all
│                       domain-agnostic, operate on FullTextDocument / claim lists
├── indexer/            embeddings (OpenAI + offline fallback), qdrant_store,
│                       postgres_store — each with an in-memory fallback backend
├── retriever/           query.py (the /prior pipeline), novelty_check.py (the
│                       six-step novelty algorithm), gap_mapper.py (/gaps ranking)
└── evaluator/           metrics.py (citation verification rate), domain_eval.py
                        (per-domain novelty accuracy), astabench.py (LitQA2-shaped proxy eval)
```

Every domain-specific input arrives via `literature_profile` on the request —
the service itself never hardcodes a sequence id, MSC code, or paper title.
Adding a new domain requires zero changes here, only a `literature_profile()`
override on the domain plugin (`packages/propab-core/propab/domain_modules/base.py`).

## Why some things are simpler than the spec's letter

- **PDF fallback uses `pypdf`, not `nougat`.** Nougat needs a multi-GB
  vision-transformer checkpoint and a GPU-friendly torch install — a
  disproportionate dependency for a service whose job is text/citation
  extraction, not layout recovery from scanned pages. `fetch_full_text`
  records `extraction_method` so a real nougat backend can be swapped in
  behind the same `BaseSource` interface later without touching callers.
- **AstaBench wiring is two things now, not one, and neither is the official
  `inspect_ai` harness.** `evaluator/astabench.run_litqa2_proxy` scores
  hand-built cases against this service's own indexed claims via embedding
  similarity (max-pool option-vs-retrieved-claim — see the module
  docstring); it's fast and offline but doesn't test real questions or use
  an LLM to answer. `evaluator/litqa2_live.py` is the honest upgrade: real
  questions from the public `futurehouse/lab-bench` LitQA2 dataset, live
  retrieval, an LLM answering from retrieved evidence — the same shape as
  every real leaderboard entry. Neither is the official `agenteval`/
  `inspect_ai` harness (`asta-bench/astabench/evals/labbench/litqa2/task.py`),
  which needs the gated `allenai/asta-bench` dataset for the official
  dev/test split partition and a compliant results bundle for submission.
  **Real measured numbers, against a real pulled leaderboard baseline
  (ReAct + Claude Opus 4.7, from `allenai/asta-bench-results`):**

  | | Accuracy | Coverage | Precision |
  |---|---|---|---|
  | This service (`litqa2_live`, n=25, live) | 0.44 | 0.64–0.68 | 0.65–0.69 |
  | Official leaderboard baseline | 0.84 | 0.89 | 0.94 |

  See `CHANGELOG.md` 0.3.0 for the full story: four real bugs found only by
  running against real questions (a full-sentence query returns zero PubMed
  hits; the claims extractor is the wrong evidence source for abstract-only
  sources; an over-cautious prompt caused 21/21 abstentions; retry tuning
  that made timeouts *worse* before it made them better), why the remaining
  gap is structural (proprietary retrieval infra + multi-round agentic
  search vs. this service's single blind search, plus real instability in
  the "-preview" model tier), and what's still open to try.
- **`open_problem_sources` scraping is best-effort, not a per-site scraper.**
  `gap_mapper._scrape_open_problem_source` reuses the same "Problem:"/
  "Open problem:" marker regex the extractor uses on papers. If a domain's
  list page doesn't use those markers, it silently contributes nothing —
  honest empty result beats a fragile bespoke parser per site.

## Evaluation loop

- `evaluator/metrics.citation_verification_rate` — re-fetches the source for
  a sample of indexed claims and confirms the verbatim quote is actually
  present. This is the primary health metric (target ≥ 90%); run it as an
  offline job (not on every `/health` call — re-fetching sources is slow and
  would make health checks flaky under source rate limits) and feed the
  result into `artifacts/literature_coverage.json`.
- `evaluator/domain_eval.run_domain_eval` — scores `/novelty` verdicts
  against a domain's hand-built test set (established / open / tabulated /
  not-tabulated cases). Target ≥ 90% before a domain integrates with
  campaign launch.
- `evaluator/astabench.run_litqa2_proxy` and `evaluator/litqa2_live.py` — see above.
- After any campaign where a "novel" finding turns out to be known, log the
  miss to `artifacts/literature_misses.json` (schema in `agent3.md`) — this
  is the highest-value quality signal for improving the service.

## Submitting to the real AstaBench leaderboard

This has been **researched but not done** — it's an externally-visible,
public action tied to a specific identity, not something to do unilaterally.
What it actually requires (from the leaderboard Space's own `submission.py`
and dataset schema):

1. A HuggingFace account (the Space uses `hf_oauth`) to sign in and submit.
2. A results bundle produced by the **official** `agenteval`/`inspect_ai`
   harness — not this repo's scripts — which means wiring this service (or a
   solver that calls it) into `asta-bench/astabench/evals/labbench/litqa2/`
   as an Inspect `Solver`, requesting access to the gated `allenai/asta-bench`
   dataset (auto-approval gate, not fully public) for the official test
   split, and running the real eval end-to-end via their CLI.
3. Filling out the submission form: agent name, description, a public
   `agent_url` (this repo isn't currently public), an openness declaration,
   and uploading the resulting `.tar.gz` — capped at one submission per
   split per 24h.
4. The bundle uploads to the public `allenai/asta-bench-submissions` dataset
   and, once processed, appears permanently in `allenai/asta-bench-results`
   and on the public leaderboard, attributed to that account.

None of this is a small last step on top of what's here — it's a separate
integration project (wiring an Inspect `Solver`, getting dataset access,
running the official harness) plus a decision only the user should make
(what identity/account to submit under, whether this system is ready to be
publicly attributed). Flagged for a decision, not executed.

## Tests

```bash
pip install pytest pytest-asyncio
pytest services/literature/tests -q
```

All tests run offline: HTTP calls are mocked with `httpx.MockTransport`,
storage/embeddings use their in-memory/offline backends. No network access
or running Postgres/Qdrant instance is required.
