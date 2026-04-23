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

### Roadmap phases (`ARCHITECTURE.md` §16)

| Phase | Scope | Repo status |
|------|--------|---------------|
| **1** — Foundation | Compose, Postgres **+ Alembic**, events, SSE, arXiv/PDF, chunk/Qdrant, BM25, citations, TTL | **Closed for this slice**: Alembic at repo root; Postgres still auto-applies `migrations/001_initial.sql` on first Docker boot; run `alembic upgrade head` with `DATABASE_URL_SYNC` for revision tracking. Orchestrator **stub** exposes `/health` on port **8010** (full loop still runs in the API until a later split). |
| **2** — Retrieval + prior | Query expansion, hybrid + RRF, cross-encoder, prior, short-circuit, literature tests | **Mostly done**: optional **cross-encoder** via `pip install -e ".[rerank]"` and `RERANKER_ENABLED=true`; RRF unit test in `tests/test_retrieval_rrf.py`. |
| **3** — Agent core | Loop, hypotheses, Celery, tools, sandbox, 40 tools | **Partial** (no 40-tool pack yet). |
| **4** — Output + frontend | Full paper sections + Jinja, inspector UX | **Partial** (methods trace + minimal LaTeX; frontend submit/SSE/session JSON). |
| **5** | Ollama, datasets, grounding | Backlog. |

**Dev deps:** `pip install -e ".[dev]"` — Alembic, `psycopg`, pytest.

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