# Propab

**Autonomous AI research system. Ask a research question. Get a paper.**

Propab ingests scientific literature, builds a structured understanding of a field, generates and tests hypotheses in parallel using domain-specific computational tools, and writes an arxiv-formatted paper grounded in the actual experiment trace — all from a single research question.

```bash
git clone https://github.com/shani-singh1/propab-oss.git
cd propab-oss
cp .env.example .env        # add OPENAI_API_KEY
docker compose up
# → http://localhost:3000
# → http://localhost:8010/health  (orchestrator stub, Phase 1 service map)
```

### Fast debugging (no multi-hour campaign)

Use two profiles only: **`campaign`** (real runs; default in `docker-compose.yml`) and **`dev`**
(short sandbox, heuristic sub-agent, `fast_tool` baseline — see `packages/propab-core/propab/config.py`).

After `pip install -e .` from the repo root, run infrastructure checks and a **single sub-agent in-process**
(no Celery queue) to reproduce tool / sandbox / LLM issues in minutes:

```bash
set PROPAB_PROFILE=dev
propab health
propab agent --cleanup
# Campaign-shaped harness (caps + baseline mode) without exporting env first:
propab agent --profile campaign --cleanup
```

Without installing the `propab` script on your PATH, use: `python -m propab health` and `python -m propab agent --cleanup` (add `--profile campaign` to match production campaign settings).

Optional: `propab health --with-train-smoke` runs a tiny `train_model` (needs a working sandbox).
`propab health --with-celery` pings workers. **`propab bank --cleanup`** runs four fixed harness cases (tool/stat/grad paths) under a few minutes.
**`propab replay --snapshot <path> --phase synthesis|abstract|paper`** rebuilds synthesis / stub abstract / full paper from a snapshot written under `PROPAB_DATA_DIR/campaign_snapshots/<campaign_id>/` at baseline, first confirmed, and pre-paper.
**`propab trace-replay --hypothesis-id <uuid>`** replays stored `tool_calls` against the current registry.

**Fast code iteration in Docker (no image rebuild for Python edits):**

```bash
docker compose -f docker-compose.yml -f docker-compose.mount-dev.yml up -d
# API/orchestrator auto-reload; after worker code changes:
docker compose -f docker-compose.yml -f docker-compose.mount-dev.yml restart worker
```

**Cold start (order of magnitude, MNIST campaign v1)** — `hypotheses_tested` stays 0 until the first worker returns a tree update:

| Phase | `PROPAB_PROFILE=campaign` (typical) | `PROPAB_PROFILE=dev` |
|--------|-------------------------------------|-------------------------|
| Prior | up to ~180 s (`campaign_prior_timeout_sec`) | up to ~45 s |
| Baseline | **`fast_tool`** train_model slice (often **~0.5–3 min**) | **`fast_tool`** train_model slice often **~0.5–3 min** |
| First batch tick | first parallel completion often **~10–25+ min** after baseline | much smaller agents + wall caps (**minutes**) |

For full campaign monitoring once the API is up: `python scripts/start_campaign_v1.py` then `python scripts/monitor_campaign.py`.

### Roadmap phases (`ARCHITECTURE.md` §16)

| Phase | Scope | Repo status |
|------|--------|---------------|
| **1** — Foundation | Compose, Postgres **+ Alembic**, events, SSE, arXiv/PDF, chunk/Qdrant, BM25, citations, TTL | **Closed for this slice**: **Alembic is the single schema source of truth** — Docker Compose runs a one-shot `migrate` service (`alembic upgrade head`) before app services start, so the schema is always complete and revision-tracked. The `migrations/*.sql` files are DDL bodies consumed by the Alembic revisions (not applied directly). Orchestrator **stub** exposes `/health` on port **8010** (full loop still runs in the API until a later split). |
| **2** — Retrieval + prior | Query expansion, hybrid + RRF, cross-encoder, prior, short-circuit, literature tests | **Mostly done**: optional **cross-encoder** via `pip install -e ".[rerank]"` and `RERANKER_ENABLED=true`; RRF unit test in `tests/test_retrieval_rrf.py`. |
| **3** — Agent core | Loop, hypotheses, Celery, tools, sandbox, 40 tools | **In progress (DL/ALGO/ML v1)**: hypothesis ranking, Celery workers, **per-domain sandbox timeouts**, domain routing biased to `deep_learning` / `algorithm_optimization` / `ml_research` (+ math/stats/data/general), expanding **TOOLS.md** tool surface with tests (full 40-tool matrix still open). |
| **4** — Output + frontend | Full paper sections + Jinja, inspector UX | **Progress**: deterministic **Methods** + **Results** + **References** + optional **Figures** (MinIO objects embedded into LaTeX build), LLM **abstract / intro / discussion / conclusion**, frontend **paper links** + **LLM call** JSON; run `alembic upgrade head` for schema updates such as `hypotheses.tool_trace_id`. |
| **5** | Ollama, datasets, grounding | Backlog. |

**Dev deps:** `pip install -e ".[dev]"` — Alembic, `psycopg`, pytest.

**Schema:** Alembic is the single source of truth. Under Docker Compose this runs automatically
via the one-shot `migrate` service. For local (non-Docker) runs, apply migrations with
`DATABASE_URL_SYNC` from `.env` (sync driver for Alembic CLI):

```bash
alembic upgrade head
```

**CI:** GitHub Actions runs `pytest` on push and pull requests (`.github/workflows/ci.yml`).

**Still open (see `ARCHITECTURE.md` §16 and `TOOLS.md`):** full ~40-tool matrix and remaining P2/P3 tools; LLM-based intake domain; richer multi-step plans (LLM-planned); orchestrator split from API; Docker Compose CI smoke; Ollama, dataset plugins, claim-grounding (§15). **Done recently:** two tool steps + sandbox; **`model_id` chain** after `build_mlp` / `build_transformer` into the next tool when applicable.

---

## What it does

You submit a research question. Propab:

1. Searches arxiv and builds a structured prior — established facts, open gaps, dead ends
2. If the answer already exists in literature, returns it immediately with citations
3. If not — generates N ranked hypotheses and tests all of them in parallel
4. Each hypothesis gets its own sub-agent with access to domain-specific STEM tools
5. Agents that need custom computation write sandboxed Python and execute it safely
6. The orchestrator collects results, identifies breakthroughs and dead ends
7. Writes a full arxiv-format paper compiled from the actual experiment trace

Every step is transparent. Every tool call, every LLM prompt, every intermediate result is streamed to the UI in real time and stored for inspection. Nothing happens silently.

---

## Demo

> *"Does transformer attention efficiency degrade non-linearly with sequence length?"*

```
● session started
● literature: fetching papers from arxiv...
  ↳ found: "Efficient Transformers: A Survey" (2009.06732)
  ↳ found: "FlashAttention: Fast Memory-Efficient Exact Attention" (2205.14135)
  ↳ found: 14 more papers — cached 3, fetched 11
● prior built: 4 established facts · 3 open gaps · 2 dead ends
● no direct answer found — generating hypotheses
● 5 hypotheses generated and ranked
  ↳ h1 (score 0.87): "Attention cost grows quadratically but memory access patterns..."
  ↳ h2 (score 0.81): "Sparse attention approximations break down at..."
  ...
● experiments running (5 parallel agents)
  ↳ h1 → tool: run_regression(sequence_lengths=[128..8192], metric="flops")
  ↳ h2 → tool: statistical_test(type="curve_fit", model="polynomial")
  ↳ h3 → sandbox: custom benchmarking code executing...
● h1 confirmed (confidence: 0.91) — breakthrough
● h2 refuted · h3 inconclusive · h4 confirmed · h5 refuted
● paper compiling...
● paper ready → download PDF · download .tex
```

---

## How it works

```
research question
       │
       ▼
┌──────────────────┐
│  literature      │  hybrid search: dense embeddings + BM25 + citation graph
│                  │  builds structured prior: facts / gaps / dead ends
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  hypothesize     │  generate N candidates · rank by novelty + testability
└────────┬─────────┘
         │
    ┌────┴─────────────────────────────────┐
    │         parallel sub-agents          │
    │  h1 agent   h2 agent   ...  hN agent │
    │  tools      tools           tools    │
    │  sandbox    sandbox         sandbox  │
    └────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  synthesize      │  result ledger · breakthroughs · dead ends
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  paper           │  methods section compiled deterministically from trace
└──────────────────┘  introduction / results / discussion by LLM
```

---

## Transparency

Propab is built around one constraint: nothing happens silently.

Every agent decision, tool call, LLM prompt, intermediate result, and failure is emitted as a named structured event, streamed to the UI in real time, and persisted to the database. Researchers can inspect the full trace of any session — every prompt that was sent, every tool that was called, every piece of code that ran in the sandbox.

The paper's methods section is compiled directly from the experiment trace, not written by an LLM. Every claim can be traced back to a specific experiment step.

---

## STEM Tools

Sub-agents select from domain-specific tool clusters. Context never gets the full list — only the relevant cluster for the declared domain.

| Domain | Example tools |
|---|---|
| Computational biology | `sequence_align` `protein_fold_predict` `gc_content` `crispr_score` `rna_fold` |
| Statistics | `run_regression` `anova_test` `bootstrap_ci` `bayesian_update` `power_analysis` |
| Mathematics | `solve_ode` `symbolic_differentiate` `numerical_integrate` `eigenvalue_decompose` |
| ML modeling | `train_classifier` `cross_validate` `feature_importance` `learning_curve` |
| Data analysis | `describe_stats` `plot_distribution` `outlier_detect` `group_aggregate` |

When no pre-built tool covers the task, the sub-agent writes Python and executes it in an isolated sandbox with no network access, memory caps, and timeout enforcement.

---

## Requirements

- Docker + Docker Compose
- An API key for OpenAI (or Anthropic — set `LLM_PROVIDER=anthropic`)
- 8GB RAM recommended for running 3+ parallel experiments

No GPU required. The reranker runs on CPU. Everything else is API calls or Python.

---

## Configuration

All configuration is via environment variables. Defaults work for a local run.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. LLM + embedding calls |
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `ollama` |
| `LLM_MODEL` | `gpt-4o` | Model for orchestrator and agents |
| `PROPAB_WORKERS` | `3` | Parallel experiment workers |
| `PROPAB_MAX_HYPOTHESES` | `5` | Hypotheses generated per question |
| `PAPER_TTL_DAYS` | `30` | Days before cached papers refresh |
| `SANDBOX_TIMEOUT_SEC` | `30` | Max CPU time for sandboxed code |
| `SANDBOX_MEMORY_MB` | `512` | Max sandbox memory |

---

## Adding a tool

Tools are plain Python functions with a spec dict. No base classes, no registration step.

```python
# packages/propab-core/propab/tools/statistics/my_tool.py

TOOL_SPEC = {
    "name":        "my_tool",
    "domain":      "statistics",
    "description": "What this tool does in one sentence",
    "params": {
        "data": {"type": "list[float]", "required": True, "description": "Input data"}
    },
    "output": {
        "result": "float — the computed result"
    }
}

def my_tool(data: list[float]) -> ToolResult:
    # implementation
    return ToolResult(success=True, output={"result": ...})
```

The registry scanner discovers it automatically. Write a unit test, open a PR.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full tool submission checklist.

---

## Offline operator credit (Layer 0.5+)

Zero-token infrastructure for operator evolution: DB-backed traces, counterfactual replay,
hierarchical credit, and OperatorBench. Requires Postgres with completed campaigns.

```bash
# 1. Start stack (Postgres must be up)
docker compose up -d postgres redis migrate

# 2. Extract entropy trajectories from existing campaigns (once)
python scripts/extract_entropy_trajectories.py

# 3. Harvest DB-backed operator traces
python scripts/harvest_db_traces.py

# 4. Full credit cycle (prefers Postgres; falls back to snapshots)
python scripts/run_operator_credit.py

# Offline-only (no DB):
python scripts/run_operator_credit.py --no-db
```

Outputs land in `artifacts/operator_credit_report.json` and `data/lifetime_knowledge/`.
Theory doc: [docs/search_dynamics.md](./docs/search_dynamics.md).

---

## Architecture

Full system design, agent contracts, event schema, data layer, and failure handling:

→ [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## Roadmap

- [x] Architecture and engineering design
- [ ] Phase 1: Event system, infrastructure, literature ingestion
- [ ] Phase 2: Hybrid retrieval, prior builder
- [ ] Phase 3: Orchestrator, sub-agents, tool registry, sandbox
- [ ] Phase 4: Paper writer, frontend, one-command setup
- [ ] Phase 5: Ollama / local model support, dataset connectors

---

## License

MIT
