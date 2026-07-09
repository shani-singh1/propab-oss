<div align="center">

# Propab

**An autonomous, literature-grounded research campaign engine.**

Ask a scientific question. Propab reads the literature, forms rival hypotheses,
tests them in parallel, refuses to confirm what doesn't generalize, and writes a
paper grounded in the actual experiment trace — while reporting a health metric
for every part of itself so you always know *which* component is working.

[![CI](https://github.com/shani-singh1/propab-oss/actions/workflows/ci.yml/badge.svg)](https://github.com/shani-singh1/propab-oss/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-530%20passing-brightgreen)](./tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](./pyproject.toml)

[Quickstart](#quickstart) · [How it works](#how-it-works) · [Current results](#current-results) · [Observability](#observability-the-differentiator) · [Adding a domain](#adding-a-domain) · [Architecture](./ARCHITECTURE.md)

</div>

---

## Why Propab is different

Most "AI scientist" systems fail silently: you run them for hours, get zero
confirmed findings, and have no idea whether the model was wrong, the data was
underpowered, or the pipeline broke. Propab is built around two constraints that
make it debuggable and trustworthy:

- **Every component has one health metric.** When something goes wrong, the
  first question is "which component's number is out of range" — literature
  citation verification rate, hypothesis duplicate rate, evidence-binding
  rejection rate, worker utilization, artifact-gate precision — not "it failed."
  See [`propab_ownership_contracts.md`](./propab_ownership_contracts.md).
- **A confirmed finding has to survive.** Every "confirmed" verdict passes an
  **artifact gate** (label-shuffle / permutation null tests) and an
  **out-of-distribution scope gate**. Citations are bound to beliefs *at write
  time*, so a fabricated or irrelevant citation is rejected before it is ever
  persisted.

**Evidence binding fix:** An early audit found that **94.5% of belief citations
were irrelevant** to the claim they supposedly supported — convenient nodes
attached after the fact. Propab now runs `evidence_binding` at belief write time;
citations that do not bear on the claim are dropped before persistence. Target:
≥95% citation integrity (see ownership contracts).

And it fails fast: a **domain preflight gate** refuses to launch a campaign on
data that is underpowered, instead of burning hours to discover it.

**Current state (honest):** Propab is research-grade infrastructure, not a
turnkey product. Math combinatorics and genomics domains are the most exercised;
materials and network diffusion have run real campaigns; enzyme kinetics and
graph invariants are stubs. The frontend is evolving separately (Agent 3).

---

## Features

| | |
|---|---|
| **Literature-grounded** | Builds a structured prior (established facts, open gaps, contradictions, dead ends) before generating hypotheses. |
| **Rival-belief campaigns** | Maintains ≤3 competing beliefs, runs the most discriminating experiment next, and detects when a direction is exhausted. |
| **Honest verification** | Composed verdict pipeline: classify → artifact gate → OOD gate → scope integrity. Confirmed means it generalized. |
| **Evidence binding at write time** | Citations that don't bear on a claim are removed before persistence — no convenient-but-fabricated support. |
| **Domain plugins** | Core is domain-agnostic; each science implements one `DomainPlugin`. Active: `math_combinatorics`, `genomics`, `materials`, `network_diffusion`, `mandrake`. Stubs: `enzyme_kinetics`, `graph_invariants`. |
| **Preflight power gate** | `DOMAIN_PREFLIGHT_FAILED` in seconds beats "8 hours, 0 confirmed, underpowered all along." |
| **Full observability** | Eight health metrics logged to Postgres; every campaign ends with a non-null enum stop reason. |
| **Resumable & resilient** | State checkpointed to Postgres each round; API restarts don't kill campaigns; crashed workers requeue their hypothesis. |
| **Transparent by design** | Every LLM prompt, tool call, and verdict is a structured event, streamed live over SSE and stored for inspection. |

---

## Quickstart

**Requirements:** Docker + Docker Compose, and an LLM API key (OpenAI or Google
Gemini). No GPU required. New here? See the step-by-step
[beta setup & troubleshooting guide](./docs/beta-setup.md).

```bash
git clone https://github.com/shani-singh1/propab-oss.git
cd propab-oss                 # keep this dir name — the sandbox image default depends on it
cp .env.example .env          # then set your LLM provider + key (see below)
docker compose up
```

In `.env`, set **one** provider:

- **OpenAI** (matches defaults): `LLM_PROVIDER=openai`, `OPENAI_API_KEY=sk-...`
- **Gemini**: `LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-2.0-flash`,
  `GOOGLE_API_KEY=AIza...`, and `EMBED_PROVIDER=google` (so embeddings reuse the
  same key). Everything else has a working default.

This starts the full stack — `api`, `orchestrator`, `worker`, `literature`,
`frontend`, plus Postgres, Redis, Qdrant, MinIO — and runs database migrations
automatically (the one-shot `migrate` service applies `alembic upgrade head`).

- Dashboard → http://localhost:3000
- API health → http://localhost:8000/health · API docs → http://localhost:8000/docs
- Literature service → http://localhost:8020/health

Launch a campaign from the dashboard, via the API ([docs](./docs/api_reference.md)), or
with curl. `math_combinatorics` is the safest first run — pure computation, no
dataset needed:

```bash
curl -X POST http://localhost:8000/campaigns \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which additive combinatorics constructions maximize Sidon density under greedy search? [domain_profile:math_combinatorics]", "compute_budget_hours": 1.0, "max_hypotheses": 40}'
```

Production-style deploy (built images, no source mounts):

```bash
docker compose -f docker-compose.prod.yml up -d
```

Then watch it live at `http://localhost:8000/stream/{campaign_id}` (SSE) or in
the dashboard.

---

## How it works

```
             question + domain
                    │
                    ▼
   ┌───────────────────────────────┐
   │  preflight gate                │  refuse launch if the domain is underpowered
   └───────────────┬───────────────┘
                   ▼
   ┌───────────────────────────────┐
   │  literature prior              │  facts · gaps · contradictions · dead ends
   └───────────────┬───────────────┘
                   ▼
   ┌───────────────────────────────┐
   │  hypotheses (scoped)           │  population · claimed generalization · OOD test
   └───────────────┬───────────────┘
        ┌──────────┴───────────────────────────┐
        │        parallel Celery workers        │  one hypothesis each, in a sandbox
        └──────────┬───────────────────────────┘
                   ▼
   ┌───────────────────────────────┐
   │  verification pipeline         │  classify → artifact gate → OOD → scope
   └───────────────┬───────────────┘
                   ▼
   ┌───────────────────────────────┐
   │  synthesis                     │  ≤3 rival beliefs · evidence bound at write time
   └───────────────┬───────────────┘   · refill frontier with the next best experiment
                   ▼
   ┌───────────────────────────────┐
   │  finalize + paper              │  non-null stop reason · paper from the real trace
   └───────────────────────────────┘
```

The campaign loop runs in the **orchestrator** service and checkpoints to
Postgres after every round, so it survives restarts. Full design in
[ARCHITECTURE.md](./ARCHITECTURE.md).

---

## Current results

Propab has produced **reproducible, verifier-checked findings** in math
combinatorics — notably greedy Sidon set density trends and cap-set ratio bands
at large `n`. These are **computational confirmations under stated scopes**, not
claims of new theorems; replay through fixed verifiers is the acceptance bar
(false confirm rate target < 10%).

The **genomics** plugin validates cross-tissue expression claims with
leave-one-tissue-out ridge regression and tissue-label shuffle nulls on a GTEx
subset (synthetic fallback when no local data is mounted).

Compare campaigns quantitatively:

```bash
python scripts/compare_campaigns.py --campaign-a <id> --campaign-b <id>
```

---

## Observability (the differentiator)

Propab logs one health metric per component to Postgres. A campaign dashboard can
flag any metric that is out of its target range, so debugging starts with a
number instead of a guess.

| Metric | Target | Table |
|---|---|---|
| Literature citation verification rate | ≥ 90% | `campaign_literature_priors` |
| Hypothesis duplicate rate (per round) | < 10% | `campaign_synthesis_events` |
| Belief citation integrity | ≥ 95% | `campaign_synthesis_events` |
| Evidence-binding rejection rate | < 5% | `campaign_synthesis_events` |
| Worker experiment success rate | ≥ 70% | `research_campaigns` |
| Worker utilization | ≥ 80% | `research_campaigns` |
| Artifact-gate precision | ≥ 90% | `campaign_audit_results` |
| Stop-reason accuracy | non-null | `research_campaigns.stop_reason` |

Full contracts (what each component owns, its inputs/outputs, and what a drop in
its metric means): [`propab_ownership_contracts.md`](./propab_ownership_contracts.md).

---

## Adding a domain

Core never imports a dataset, feature, or threshold directly — it asks a
`DomainPlugin`. **Full walkthrough:** [`docs/adding_a_domain.md`](./docs/adding_a_domain.md)
(genomics as the worked example).

Minimal sketch:

```python
from propab.domain_modules.base import DomainPlugin, PreflightResult

class MyDomainPlugin(DomainPlugin):
    domain_id = "mydomain"
    display_name = "My domain"

    def matches(self, *, question="", payload=None) -> bool:
        return "mydomain" in (question or "").lower()

    def available_features(self) -> list[str]:
        return ["feature_a", "feature_b"]

    def preflight(self) -> PreflightResult:
        n = load_my_dataset_size()
        if n < 500:
            return PreflightResult(False, f"underpowered: {n} rows", {"n": n})
        return PreflightResult(True, "ok", {"n": n})

    def run_verification(self, hypothesis, evidence=None, features=None) -> dict:
        ...  # return evidence dict with verified_true_steps, metric_name, ...

    def confirmation_criteria(self) -> dict:
        ...
```

Register in `propab/domain_modules/registry.py`, add a routing corpus (≥20 entries),
and run the [validation checklist](./docs/adding_a_domain.md#step-5--validation-checklist).

Adding a **tool** is lighter — a plain function plus a `TOOL_SPEC` dict; the
registry discovers it. See [TOOLS.md](./TOOLS.md).

---

## Configuration

All configuration is via environment variables (see [`.env.example`](./.env.example)).
Defaults work for a local run.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | — | LLM + embedding calls (one required) |
| `LLM_PROVIDER` | `openai` | `openai` · `gemini` · `ollama` |
| `LLM_MODEL` | `gpt-4o` | Model for orchestrator + agents |
| `DATABASE_URL` | local Postgres | Async connection string |
| `REDIS_URL` / `CELERY_BROKER_URL` | local Redis | Pub/sub + Celery broker |
| `ORCHESTRATOR_URL` | — | If set, API delegates campaigns to the orchestrator; else in-process |
| `ORCHESTRATOR_INTERNAL_TOKEN` | — | Shared secret for the internal campaign endpoint |
| `QDRANT_URL` / `MINIO_ENDPOINT` | — | Vector search / object storage |
| `SANDBOX_TIMEOUT_SEC` | `120` | Max CPU time for sandboxed code (profile-tuned) |
| `LIFETIME_STORE_BACKEND` | `json` | `json` (tests) or `postgres` (production upserts) |
| `PROPAB_DATA_DIR` | `./data` | Lifetime store, genomics cache, snapshots |

---

## Development

```bash
pip install -e ".[dev]"          # editable install + Alembic, psycopg, pytest

python -m pytest tests -q        # project suite (530+ passing)
alembic upgrade head             # apply migrations locally (uses DATABASE_URL_SYNC)
```

> **Note:** the vendored `asta-bench/` directory has its own dependencies and is
> not part of the project test suite. Always scope pytest to `tests/`.

**Fast iteration in Docker** (auto-reload for Python edits, no image rebuild):

```bash
docker compose -f docker-compose.yml -f docker-compose.mount-dev.yml up -d
# after worker code changes:
docker compose -f docker-compose.yml -f docker-compose.mount-dev.yml restart worker
```

**In-process debugging** (reproduce a single sub-agent without the Celery queue):

```bash
python -m propab health
python -m propab agent --profile campaign --cleanup
```

---

## Repository layout

```
packages/propab-core/propab/   domain-agnostic core (campaign, verification,
                               synthesis, evidence binding, domain plugins,
                               health metrics)
services/api/                  FastAPI HTTP entrypoint + routes + SSE
services/orchestrator/         run_campaign_loop, literature, events, paper
services/worker/               Celery worker + think-act sub-agent loop
alembic/                       migrations (single schema source of truth)
frontend/                      Vite/React campaign dashboard
docs/                          adding_a_domain, api_reference, operator_runbook
tests/                         pytest suite (530+)
```

---

## Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) — services, campaign path, verification, domains, persistence
- [docs/beta-setup.md](./docs/beta-setup.md) — clone-and-run beta setup + troubleshooting
- [docs/adding_a_domain.md](./docs/adding_a_domain.md) — add a domain (genomics worked example)
- [docs/api_reference.md](./docs/api_reference.md) — launch, monitor, resume campaigns
- [docs/operator_runbook.md](./docs/operator_runbook.md) — production troubleshooting
- [propab_ownership_contracts.md](./propab_ownership_contracts.md) — component contracts + health metrics
- [docs/component_map.md](./docs/component_map.md) — per-symbol wiring map
- [TOOLS.md](./TOOLS.md) — the sub-agent tool surface
- Interactive API — http://localhost:8000/docs (Swagger UI when API is running)

---

## Contributing

Contributions are welcome. The one rule that keeps the system debuggable: when
you add a component, **write its ownership contract (what it owns, never owns,
its input/output, and its health metric) before you write its code.** If you
can't write the contract, the component isn't well-defined yet.

---

## License

[MIT](./LICENSE)
