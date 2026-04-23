# PROPAB — Open Source AI Research System
## System Architecture & Engineering Design

> **Transparency is a first-class constraint.** Every agent decision, every tool call, every LLM prompt, every result, every failure emits a named structured event. Nothing happens silently. Researchers can inspect exactly what Propab did and why at every step.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Design Principles](#2-design-principles)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Event System — The Transparency Backbone](#4-event-system--the-transparency-backbone)
5. [Literature Layer](#5-literature-layer)
6. [Hypothesis Layer](#6-hypothesis-layer)
7. [Experiment Layer — Sub-Agent Architecture](#7-experiment-layer--sub-agent-architecture)
8. [Tool System](#8-tool-system)
9. [Orchestrator — Custom Async Loop](#9-orchestrator--custom-async-loop)
10. [Paper Writing Layer](#10-paper-writing-layer)
11. [Data Layer](#11-data-layer)
12. [API & Streaming Layer](#12-api--streaming-layer)
13. [Infrastructure & Setup](#13-infrastructure--setup)
14. [Failure Handling](#14-failure-handling)
15. [Open Problems (v2 Backlog)](#15-open-problems-v2-backlog)
16. [Build Roadmap](#16-build-roadmap)
17. [Repository Structure](#17-repository-structure)

---

## 1. System Overview

Propab is an autonomous research system. A researcher submits a question. The system:

1. Determines whether the answer already exists clearly in literature
2. If not — generates N ranked hypotheses
3. Spawns one sub-agent per hypothesis, all running in parallel
4. Each sub-agent uses pre-built STEM tools or sandboxed generated code to test its hypothesis
5. The orchestrator collects crisp structured results, identifies dead ends and working directions
6. Writes an arxiv-formatted paper grounded in the actual experiment trace

**Target user:** Any researcher — PhD student, independent researcher, lab team — who can run `docker compose up`.

**Setup requirement:** One command. No manual configuration beyond providing an LLM API key.

---

## 2. Design Principles

### 2.1 Transparency First
Every step emits a structured event. The event stream is the system's real-time log, the frontend's data source, and the paper writer's audit trail. If something is not in the event stream, it did not happen.

### 2.2 No Black Boxes
- Orchestrator is a custom async Python loop — every state transition is explicit code
- Tools are plain functions with a JSON schema declaration — no magic base classes
- LLM prompts are stored verbatim alongside responses in `llm_calls` table
- The methods section of the paper is compiled deterministically from experiment steps, not written by an LLM

### 2.3 Fail Loudly
No silent failures. Every error is a structured object with: error type, message, step context, input that caused it, and recovery action taken. Errors are surfaced to the frontend in real time.

### 2.4 Lazy-First Data
No bulk pre-ingestion of arxiv. Papers are fetched on demand, processed, and cached with a TTL. First queries fetch fresh. Repeat queries hit cache. This makes local setup feasible.

### 2.5 Tool Discipline
Ship 40 high-quality, well-tested STEM tools across 5 domains. Expose a plugin interface for community additions. Tools are loaded by cluster (15–25 tools per domain) — never the full list — so context never explodes.

### 2.6 One Command Setup
`docker compose up` starts everything. No manual config of vector DBs, queues, or model endpoints. All configuration is via environment variables with documented defaults.

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            PROPAB SYSTEM                                │
│                                                                         │
│  ┌──────────────┐      ┌──────────────────────────────────────────────┐ │
│  │   Frontend   │─────▶│           API Gateway  (FastAPI)             │ │
│  │  React + SSE │◀─────│    POST /research  GET /stream/{session_id}  │ │
│  └──────────────┘      └──────────────────┬───────────────────────────┘ │
│                                           │                             │
│                    ┌──────────────────────▼──────────────────────────┐  │
│                    │           ORCHESTRATOR SERVICE                   │  │
│                    │                                                  │  │
│                    │  research_loop()                                 │  │
│                    │  ├── intake()           parse + classify         │  │
│                    │  ├── literature()        build prior             │  │
│                    │  ├── hypothesize()       generate + rank         │  │
│                    │  ├── experiment()        dispatch sub-agents     │  │
│                    │  ├── synthesize()        collect + evaluate      │  │
│                    │  └── write_paper()       compile + render        │  │
│                    │                                                  │  │
│                    │  emit(event) ──────────────────────────────────▶ │  │
│                    └──────────┬──────────────────────────────────────┘  │
│                               │                                         │
│         ┌─────────────────────┼──────────────────────┐                  │
│         │                     │                      │                  │
│  ┌──────▼──────────┐  ┌───────▼──────────┐  ┌───────▼────────────────┐ │
│  │  LITERATURE     │  │  HYPOTHESIS      │  │  EXPERIMENT RUNNER     │ │
│  │  SERVICE        │  │  SERVICE         │  │  (N Celery workers)    │ │
│  │                 │  │                  │  │  sub_agent_loop()      │ │
│  │  hybrid_search()│  │  generate()      │  │  ├── tool_dispatch()   │ │
│  │  build_prior()  │  │  rank()          │  │  ├── code_exec()       │ │
│  │  detect_gaps()  │  │                  │  │  └── emit(event)       │ │
│  └──────┬──────────┘  └───────────────── ┘  └───────────────────────┘ │
│         │                                                               │
│  ┌──────▼──────────────────────────────────────────────────────────┐   │
│  │                     SHARED INFRASTRUCTURE                        │   │
│  │                                                                  │   │
│  │  Postgres · Redis · Qdrant · MinIO · Docker sandbox             │   │
│  │  EVENT BUS (Redis pub/sub) ◀── all services emit here           │   │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Service Map

| Service | Responsibility | Stack |
|---|---|---|
| API Gateway | HTTP + SSE endpoints, session management, event fan-out | FastAPI, uvicorn |
| Orchestrator | Main research loop, state machine, synthesis | Python asyncio |
| Literature Service | Arxiv fetch, PDF parse, hybrid index, prior builder | Python, Qdrant, BM25 |
| Hypothesis Service | Generate + rank hypotheses from prior | LLM API |
| Experiment Runner | Sub-agents, tool dispatch, sandboxed code exec | Celery, Docker |
| Tool Registry | Load tool clusters by domain, plugin discovery | Python package |
| Paper Writer | Trace compiler + section generator + LaTeX render | Jinja2, pdflatex |
| Frontend | Submit jobs, live event stream, result explorer | React, TailwindCSS |
| Event Bus | Real-time event pub/sub across all services | Redis pub/sub |

---

## 4. Event System — The Transparency Backbone

The event system is the most important architectural component. Every service emits events. The frontend subscribes. The database persists. The paper writer reads. Nothing is inferred — everything is recorded.

### 4.1 Event Schema

Every event is a JSON object with this exact shape:

```python
@dataclass
class PropabEvent:
    event_id: str          # uuid4
    session_id: str        # research session this belongs to
    timestamp: str         # ISO 8601
    source: str            # which service emitted this
    event_type: EventType  # enum — see below
    step: str              # human-readable step name e.g. "literature.fetch"
    payload: dict          # event-specific data — typed per event_type
    parent_event_id: str | None  # links sub-agent events to orchestrator events
    hypothesis_id: str | None    # null for orchestrator-level events
```

### 4.2 Event Type Registry

```python
class EventType(str, Enum):

    # ── Session lifecycle ──────────────────────────────────────────────
    SESSION_STARTED          = "session.started"
    SESSION_COMPLETED        = "session.completed"
    SESSION_FAILED           = "session.failed"

    # ── Intake ────────────────────────────────────────────────────────
    INTAKE_PARSED            = "intake.parsed"        # question parsed, domain classified
    INTAKE_DECOMPOSED        = "intake.decomposed"    # sub-questions extracted

    # ── Literature ────────────────────────────────────────────────────
    LIT_FETCH_STARTED        = "literature.fetch_started"
    LIT_PAPER_FOUND          = "literature.paper_found"     # one per paper
    LIT_PAPER_CACHED         = "literature.paper_cached"    # hit cache, no re-fetch
    LIT_PAPER_PARSED         = "literature.paper_parsed"
    LIT_PAPER_INDEXED        = "literature.paper_indexed"
    LIT_RETRIEVAL_QUERY      = "literature.retrieval_query" # exact query + rephrasings
    LIT_RETRIEVAL_RESULTS    = "literature.retrieval_results"
    LIT_PRIOR_BUILT          = "literature.prior_built"     # full prior JSON in payload
    LIT_ANSWER_FOUND         = "literature.answer_found"    # short-circuit: answer exists

    # ── Hypothesis ────────────────────────────────────────────────────
    HYPO_GENERATED           = "hypothesis.generated"       # all N hypotheses
    HYPO_RANKED              = "hypothesis.ranked"          # with scores per dimension
    HYPO_DISPATCHED          = "hypothesis.dispatched"      # sent to sub-agent

    # ── Sub-agent lifecycle ───────────────────────────────────────────
    AGENT_STARTED            = "agent.started"
    AGENT_PLAN_CREATED       = "agent.plan_created"         # test plan for this hypothesis
    AGENT_STEP_STARTED       = "agent.step_started"
    AGENT_STEP_COMPLETED     = "agent.step_completed"
    AGENT_STEP_FAILED        = "agent.step_failed"
    AGENT_COMPLETED          = "agent.completed"
    AGENT_FAILED             = "agent.failed"

    # ── Tool calls ────────────────────────────────────────────────────
    TOOL_SELECTED            = "tool.selected"              # which tool, why
    TOOL_CALLED              = "tool.called"                # exact params
    TOOL_RESULT              = "tool.result"                # full output
    TOOL_ERROR               = "tool.error"                 # structured error

    # ── Code execution ────────────────────────────────────────────────
    CODE_GENERATED           = "code.generated"             # LLM-written code, verbatim
    CODE_SUBMITTED           = "code.submitted"             # sent to sandbox
    CODE_RESULT              = "code.result"                # stdout JSON
    CODE_ERROR               = "code.error"                 # structured error + retry #
    CODE_TIMEOUT             = "code.timeout"

    # ── LLM calls ─────────────────────────────────────────────────────
    LLM_PROMPT               = "llm.prompt"                 # full prompt verbatim
    LLM_RESPONSE             = "llm.response"               # full response verbatim
    LLM_PARSE_ERROR          = "llm.parse_error"            # failed to parse response

    # ── Synthesis ─────────────────────────────────────────────────────
    SYNTH_RESULT_RECEIVED    = "synthesis.result_received"  # crisp result from sub-agent
    SYNTH_LEDGER_UPDATED     = "synthesis.ledger_updated"   # orchestrator's current state
    SYNTH_BREAKTHROUGH       = "synthesis.breakthrough"     # high-confidence positive
    SYNTH_DEAD_END           = "synthesis.dead_end"

    # ── Paper ─────────────────────────────────────────────────────────
    PAPER_TRACE_COMPILED     = "paper.trace_compiled"
    PAPER_SECTION_STARTED    = "paper.section_started"
    PAPER_SECTION_COMPLETED  = "paper.section_completed"
    PAPER_LATEX_COMPILED     = "paper.latex_compiled"
    PAPER_READY              = "paper.ready"                # download URL in payload
```

### 4.3 Event Flow

```
All services → emit(event) → Redis pub/sub channel: propab:{session_id}
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                  │
            API Gateway          Postgres           MinIO
            SSE fan-out        persist all       store artifacts
            to frontend         events            referenced
                                                  in events
```

### 4.4 SSE Endpoint

```
GET /stream/{session_id}
Content-Type: text/event-stream

data: {"event_type": "literature.paper_found", "payload": {...}}
data: {"event_type": "tool.called", "payload": {"tool": "sequence_align", "params": {...}}}
data: {"event_type": "code.generated", "payload": {"code": "import numpy as np\n..."}}
...
```

The frontend subscribes to this endpoint for the duration of the research session. Every event is pushed as it happens. No polling.

---

## 5. Literature Layer

### 5.1 Ingestion Strategy — Lazy Cache

```
User query arrives
        │
        ▼
┌───────────────────────────────────┐
│  Cache check (Postgres)           │
│  Has this topic been indexed?     │
└──────────────┬────────────────────┘
               │
      ┌────────┴────────┐
    HIT                MISS
      │                  │
      ▼                  ▼
Check TTL           Fetch from arxiv API
(default 30d)       + Semantic Scholar API
      │                  │
   Fresh?           Parse PDFs
      │                  │
    YES → use       Chunk + Embed
      │                  │
     NO → re-fetch  BM25 index
                         │
                    Store in cache
                         │
                    emit(LIT_PAPER_INDEXED)
```

No bulk pre-ingestion. Papers older than `PAPER_TTL_DAYS` are re-fetched on next hit. This makes local setup feasible on a laptop.

### 5.2 PDF Processing Pipeline

```python
# Each step emits an event

def process_paper(arxiv_id: str, session_id: str) -> ProcessedPaper:

    # 1. Fetch
    emit(LIT_PAPER_FOUND, {"arxiv_id": arxiv_id, "source": "arxiv_api"})
    pdf_bytes = fetch_pdf(arxiv_id)

    # 2. Parse — section-aware
    emit(LIT_PAPER_PARSED, {"arxiv_id": arxiv_id, "section_count": N})
    sections = parse_pdf_sections(pdf_bytes)
    # Returns: {title, abstract, introduction, methods, results, discussion, references}

    # 3. Chunk — preserve section context in metadata
    chunks = chunk_sections(sections, size=512, overlap=64)
    # Each chunk carries: {text, section_name, paper_id, chunk_index}

    # 4. Embed
    embeddings = embed_batch(chunks)  # text-embedding-3-small or BGE-M3

    # 5. Store vectors
    qdrant.upsert(collection="papers", points=build_points(chunks, embeddings))

    # 6. BM25 index (per-session SQLite)
    bm25_index.add(paper_id=arxiv_id, text=full_text(sections))

    # 7. Citation graph
    refs = extract_references(sections["references"])
    postgres.insert_citations(source=arxiv_id, cited=refs)

    emit(LIT_PAPER_INDEXED, {"arxiv_id": arxiv_id})
    return ProcessedPaper(...)
```

### 5.3 Hybrid Retrieval Pipeline

Cosine similarity alone breaks when questions are rephrased. This pipeline does not.

```
Query: "does attention efficiency degrade non-linearly with sequence length?"
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  QUERY EXPANSION  (1 LLM call, emits LIT_RETRIEVAL_QUERY)  │
│                                                            │
│  Original + 3 rephrasings + key concept extraction:        │
│  → "transformer quadratic complexity"                      │
│  → "self-attention O(n²) scaling"                         │
│  → "long context computational cost"                       │
└──────────────────────────┬─────────────────────────────────┘
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
  ┌──────▼──────┐   ┌──────▼──────┐   ┌───────▼───────┐
  │  DENSE      │   │  SPARSE     │   │  CITATION     │
  │  Qdrant     │   │  BM25       │   │  GRAPH        │
  │  top-40     │   │  top-40     │   │  2-hop walk   │
  │  per query  │   │  per query  │   │  from seeds   │
  └──────┬──────┘   └──────┬──────┘   └───────┬───────┘
         │                 │                  │
  ┌──────▼─────────────────▼──────────────────▼────────┐
  │  RECIPROCAL RANK FUSION                            │
  │  score = Σ 1/(k + rank_i) for each retrieval arm  │
  │  Deduplicates. Produces unified ranked list.       │
  └──────────────────────────┬─────────────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────┐
  │  RERANKER  (cross-encoder, runs locally)            │
  │  cross_encoder.predict(query, chunk_text)           │
  │  Model: cross-encoder/ms-marco-MiniLM-L-6-v2        │
  │  Runs inline. Emits progress events.                │
  └──────────────────────────┬─────────────────────────┘
                             │
                        Top-K chunks (default K=20)
                        → Prior Builder
```

**Reranker choice:** `cross-encoder/ms-marco-MiniLM-L-6-v2` runs on CPU in ~2s for 40 candidates. No GPU required. No API call. Swappable via `RERANKER_MODEL` env var.

### 5.4 Prior Builder

The prior is not a list of papers. It is a structured field-state object that the orchestrator reasons over directly.

```python
# Schema — output of prior builder LLM call
class Prior(TypedDict):
    established_facts: list[Claim]     # appear in 3+ independent papers, consistent
    contested_claims:  list[Dispute]   # papers disagree — flagged
    open_gaps:         list[Gap]       # questions in future-work with no follow-up
    dead_ends:         list[DeadEnd]   # approaches explicitly reported as failing
    key_papers:        list[PaperRef]  # most relevant papers with one-line summary

class Claim(TypedDict):
    text:       str
    confidence: float       # 0–1
    paper_ids:  list[str]   # supporting papers

class Gap(TypedDict):
    text:        str
    source_paper: str       # paper that raised this gap
    gap_type:    str        # "unanswered_question" | "missing_data" | "untested_assumption"
```

The prior builder emits `LIT_PRIOR_BUILT` with the full prior JSON in the payload. Researchers can inspect it directly from the event stream.

**Short-circuit:** If the prior contains an established fact that directly answers the research question (cosine similarity > 0.92 between question embedding and claim embedding), emit `LIT_ANSWER_FOUND` and return immediately with the supporting papers. No hypothesis generation needed.

---

## 6. Hypothesis Layer

### 6.1 Generation

Input: research question + structured prior
Output: N ranked hypotheses

```python
# Prompt contract — stored verbatim in llm_calls table
HYPOTHESIS_PROMPT = """
You are a research hypothesis generator.

Research question: {question}

Prior — established facts:
{established_facts}

Prior — open gaps:
{open_gaps}

Prior — dead ends (do not repeat these):
{dead_ends}

Generate exactly {N} hypotheses. Each hypothesis must:
- Be specific and falsifiable
- Not duplicate any established fact
- Not repeat any dead end
- State its expected test methodology in one sentence
- Reference at least one open gap from the prior

Return JSON array only. Schema:
[{
  "id": "h1",
  "text": "...",
  "test_methodology": "...",
  "gap_reference": "gap text from prior",
  "expected_result": "..."
}]
"""
```

Emits: `LLM_PROMPT` (full prompt) → `LLM_RESPONSE` (full response) → `HYPO_GENERATED`

### 6.2 Ranking

Each hypothesis is scored across 4 dimensions. Scores are computed, not vibes.

| Dimension | Method | Weight |
|---|---|---|
| Novelty | Embedding distance from established_facts centroid | 30% |
| Testability | LLM score: can this be tested with available tools + data? | 30% |
| Potential impact | LLM score: significance if confirmed, relative to gap importance | 25% |
| Scope fit | LLM score: is this appropriately scoped to test in one session? | 15% |

Emits: `HYPO_RANKED` with scores per dimension per hypothesis — fully auditable.

```python
class RankedHypothesis(TypedDict):
    id:            str
    text:          str
    test_methodology: str
    scores: {
        novelty:      float   # 0–1
        testability:  float   # 0–1
        impact:       float   # 0–1
        scope_fit:    float   # 0–1
        composite:    float   # weighted sum
    }
    rank: int
```

---

## 7. Experiment Layer — Sub-Agent Architecture

### 7.1 Orchestrator ↔ Sub-Agent Contract

**What the orchestrator sends to each sub-agent:**

```python
class ExperimentTask(TypedDict):
    session_id:     str
    hypothesis_id:  str
    hypothesis:     RankedHypothesis
    prior:          Prior               # read-only reference
    available_tools: list[str]          # tool names for declared domain
    resource_limits: ResourceLimits     # sandbox caps
```

**What the sub-agent returns to the orchestrator:**

```python
class ExperimentResult(TypedDict):
    hypothesis_id:  str
    verdict:        Literal["confirmed", "refuted", "inconclusive"]
    confidence:     float           # 0–1
    evidence_summary: str           # 2–3 sentences max
    key_finding:    str | None      # one-line breakthrough if confirmed
    tool_trace_id:  str             # pointer to full trace in Postgres
    figures:        list[str]       # MinIO artifact IDs
    duration_sec:   float
    failure_reason: str | None      # if inconclusive due to error
```

The orchestrator reads only `ExperimentResult`. It never reads the full sub-agent trace during execution. Full traces live in Postgres, queryable for the paper writer and for researcher inspection.

### 7.2 Sub-Agent Loop

```python
async def sub_agent_loop(task: ExperimentTask) -> ExperimentResult:

    emit(AGENT_STARTED, {"hypothesis_id": task.hypothesis_id})

    # 1. Build test plan
    plan = await build_test_plan(task.hypothesis, task.prior)
    emit(AGENT_PLAN_CREATED, {"plan": plan})

    # 2. Execute steps
    results = []
    for step in plan.steps:
        emit(AGENT_STEP_STARTED, {"step": step})
        try:
            result = await execute_step(step, task)
            emit(AGENT_STEP_COMPLETED, {"step": step, "result": result})
            results.append(result)
        except StepError as e:
            emit(AGENT_STEP_FAILED, {"step": step, "error": e.to_dict()})
            if e.is_fatal:
                break

    # 3. Evaluate
    verdict = await evaluate_results(task.hypothesis, results)
    emit(AGENT_COMPLETED, {"verdict": verdict.verdict, "confidence": verdict.confidence})

    return build_experiment_result(task, verdict, results)
```

### 7.3 Parallelism Model

```python
# Orchestrator dispatches all hypotheses simultaneously
# Workers are Celery tasks, concurrency controlled by PROPAB_WORKERS

async def run_experiments(hypotheses: list[RankedHypothesis], ...) -> ResultLedger:
    tasks = [
        celery_app.send_task("run_sub_agent", args=[ExperimentTask(...)])
        for h in hypotheses
    ]

    ledger = ResultLedger()

    # Collect as results arrive — don't wait for all to finish
    for future in asyncio.as_completed(tasks):
        result = await future
        ledger.add(result)
        emit(SYNTH_RESULT_RECEIVED, {"result": result})
        emit(SYNTH_LEDGER_UPDATED, {"ledger": ledger.summary()})

        if result.verdict == "confirmed" and result.confidence > 0.85:
            emit(SYNTH_BREAKTHROUGH, {"finding": result.key_finding})

        if result.verdict == "refuted":
            emit(SYNTH_DEAD_END, {"hypothesis": result.hypothesis_id})

    return ledger
```

N defaults to 5. Capped at `PROPAB_WORKERS`. Hypotheses are dispatched simultaneously and results are collected as they arrive — the orchestrator updates its ledger incrementally and emits live state.

---

## 8. Tool System

### 8.1 Tool Interface Contract

A tool is a plain Python function with a `TOOL_SPEC` dict. No base classes. No magic. A contributor can write a new tool by reading one example.

```python
# tools/computational_biology/sequence_align.py

from propab.tools.types import ToolResult, ToolError

TOOL_SPEC = {
    "name":        "sequence_align",
    "domain":      "computational_biology",
    "description": "Align two or more DNA/RNA/protein sequences using Smith-Waterman or Needleman-Wunsch",
    "params": {
        "sequences":  {"type": "list[str]",  "required": True,  "description": "List of sequences to align"},
        "algorithm":  {"type": "str",         "required": False, "default": "smith_waterman",
                       "enum": ["smith_waterman", "needleman_wunsch"]},
        "match_score":{"type": "float",       "required": False, "default": 2.0},
        "gap_penalty":{"type": "float",       "required": False, "default": -1.0},
    },
    "output": {
        "alignment":        "list[str]   — aligned sequences with gap characters",
        "identity_pct":     "float       — percent identity across alignment",
        "score":            "float       — alignment score",
        "alignment_length": "int",
    },
    "example": {
        "params":  {"sequences": ["ATCG", "ATGG"], "algorithm": "smith_waterman"},
        "output":  {"alignment": ["ATCG", "ATGG"], "identity_pct": 75.0, "score": 5.0, "alignment_length": 4}
    }
}

def sequence_align(sequences: list[str], algorithm: str = "smith_waterman",
                   match_score: float = 2.0, gap_penalty: float = -1.0) -> ToolResult:
    try:
        # ... implementation using Biopython
        return ToolResult(success=True, output={...})
    except Exception as e:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(e)))
```

### 8.2 Tool Registry

```python
class ToolRegistry:
    """
    Discovers tools by scanning for TOOL_SPEC in tool modules.
    Loads clusters on demand — never the full list.
    """

    def __init__(self, tools_dir: Path):
        self._registry: dict[str, ToolEntry] = {}
        self._scan(tools_dir)

    def _scan(self, tools_dir: Path):
        for path in tools_dir.rglob("*.py"):
            module = import_module(path)
            if hasattr(module, "TOOL_SPEC") and hasattr(module, module.TOOL_SPEC["name"]):
                spec = module.TOOL_SPEC
                self._registry[spec["name"]] = ToolEntry(
                    spec=spec,
                    fn=getattr(module, spec["name"]),
                    domain=spec["domain"],
                )

    def get_cluster(self, domain: str) -> list[ToolSpec]:
        """Returns specs only — not functions. Sub-agent uses specs for LLM context."""
        return [e.spec for e in self._registry.values() if e.domain == domain]

    def call(self, tool_name: str, params: dict, emit_fn) -> ToolResult:
        entry = self._registry[tool_name]
        emit_fn(TOOL_CALLED, {"tool": tool_name, "params": params})
        result = entry.fn(**params)
        if result.success:
            emit_fn(TOOL_RESULT, {"tool": tool_name, "output": result.output})
        else:
            emit_fn(TOOL_ERROR, {"tool": tool_name, "error": result.error.to_dict()})
        return result
```

### 8.3 Domain Routing

```python
DOMAIN_ROUTING_PROMPT = """
Given this hypothesis, return the single most relevant domain.

Hypothesis: {hypothesis_text}

Domains:
- computational_biology: DNA/RNA/protein, genomics, CRISPR, cell biology
- chemistry: molecular simulation, reaction prediction, spectroscopy
- physics: mechanics, electromagnetism, quantum, thermodynamics
- mathematics: algebra, calculus, linear algebra, number theory, topology
- statistics: hypothesis testing, regression, Bayesian analysis, experimental design
- ml_modeling: training models, evaluation, feature engineering, architecture search
- data_analysis: EDA, visualization, aggregation, cleaning
- general_computation: anything else — code execution, data fetching, parsing

Return JSON only: {"domain": "domain_name", "reason": "one sentence"}
"""
```

One cheap LLM call. Emits `TOOL_SELECTED` with domain and reason. Sub-agent then receives only that domain's tool cluster in context.

### 8.4 Sandboxed Code Execution

For tasks no pre-built tool covers, the sub-agent writes Python and submits it to the sandbox.

```
Sub-agent → generate code → emit(CODE_GENERATED, {code: "..."})
                                       │
                                       ▼
                             ┌─────────────────────┐
                             │  SANDBOX CONTAINER  │
                             │                     │
                             │  network: none      │
                             │  memory: 512MB      │
                             │  cpu timeout: 30s   │
                             │  fs: read-only      │
                             │     except /tmp     │
                             └──────────┬──────────┘
                                        │
                           ┌────────────┴────────────┐
                         SUCCESS                   FAILURE
                           │                          │
                  emit(CODE_RESULT)          emit(CODE_ERROR, {
                  {stdout_json: {...}}         error_type: "...",
                                               message: "...",
                                               retry: N
                                             })
```

**Output contract:** Code must write results as structured JSON to stdout.

```python
# All sandbox code must end with:
import json, sys
print(json.dumps({"result": ..., "figures": [...]}))
```

**Retry policy:**
- Retry up to 3 times
- On each retry: previous error + previous code included in prompt
- After 3 failures: hypothesis marked `inconclusive`, error trace attached to result
- Emit `CODE_ERROR` on every failure with retry number

**Resource guard:**
- CPU timeout: `SANDBOX_TIMEOUT_SEC` (default 30)
- Memory: `SANDBOX_MEMORY_MB` (default 512)
- Both exceeded → structured error return, not container crash

---

## 9. Orchestrator — Custom Async Loop

No LangGraph. Every state transition is explicit Python. Every decision is a logged event. Contributors can read the control flow without framework knowledge.

```python
async def research_loop(session_id: str, question: str) -> ResearchResult:

    ctx = ResearchContext(session_id=session_id, question=question)
    emit = partial(emit_event, session_id=session_id)

    emit(SESSION_STARTED, {"question": question})

    try:
        # ── Stage 1: Intake ──────────────────────────────────────────
        ctx.parsed_question = await intake(question, emit)
        emit(INTAKE_PARSED, {"domain": ctx.parsed_question.domain,
                              "sub_questions": ctx.parsed_question.sub_questions})

        # ── Stage 2: Literature ──────────────────────────────────────
        ctx.prior = await build_prior(ctx.parsed_question, emit)
        emit(LIT_PRIOR_BUILT, {"prior": ctx.prior.to_dict()})

        # ── Stage 3: Short-circuit check ────────────────────────────
        if answer := find_existing_answer(ctx.prior, ctx.parsed_question):
            emit(LIT_ANSWER_FOUND, {"answer": answer})
            return ResearchResult(type="existing_answer", answer=answer)

        # ── Stage 4: Hypotheses ──────────────────────────────────────
        ctx.hypotheses = await generate_hypotheses(ctx.prior, ctx.parsed_question, emit)
        emit(HYPO_RANKED, {"hypotheses": [h.to_dict() for h in ctx.hypotheses]})

        # ── Stage 5: Experiments ─────────────────────────────────────
        ctx.ledger = await run_experiments(ctx.hypotheses, ctx.prior, emit)
        emit(SYNTH_LEDGER_UPDATED, {"final_ledger": ctx.ledger.to_dict()})

        # ── Stage 6: Paper ───────────────────────────────────────────
        paper_url = await write_paper(ctx, emit)
        emit(PAPER_READY, {"url": paper_url})

        emit(SESSION_COMPLETED, {"paper_url": paper_url,
                                  "breakthroughs": ctx.ledger.breakthroughs,
                                  "dead_ends": ctx.ledger.dead_ends})

        return ResearchResult(type="paper", url=paper_url, ledger=ctx.ledger)

    except Exception as e:
        emit(SESSION_FAILED, {"error": str(e), "stage": ctx.current_stage})
        raise
```

The state is explicit. Every stage transition emits an event. Every failure emits an event with the stage it failed at. No hidden state inside a framework.

---

## 10. Paper Writing Layer

### 10.1 Core Constraint

The methods section is **not written by an LLM**. It is compiled deterministically from `experiment_steps` rows in order. Every claim in the paper can be traced to a specific database row. Researchers can verify what ran.

### 10.2 Pipeline

```
INPUTS (all from Postgres + MinIO):
  ├── research_sessions row        (question, prior JSON)
  ├── hypotheses rows              (all N with verdicts + confidence)
  ├── experiment_steps rows        (full ordered trace per hypothesis)
  ├── tool_calls rows              (params + results)
  ├── llm_calls rows               (prompts + responses)
  └── MinIO artifacts              (figures, output files)

PIPELINE:
  1. trace_compiler()
     Load + group experiment_steps by hypothesis
     Tag each step: tool_call | code_exec | llm_reasoning
     Group hypotheses: confirmed / refuted / inconclusive
     → TraceReport object

  2. section_generator()   (one LLM call per section, prompt below)
     abstract      ← question + breakthrough summary from ledger
     introduction  ← prior.established_facts + prior.open_gaps
     methods       ← DETERMINISTIC: compiled from trace_report (no LLM)
     results       ← tool_outputs + figure refs + verdicts per hypothesis
     discussion    ← synthesis across hypotheses, dead ends, new directions
     conclusion    ← confirmed hypotheses + implications for field
     references    ← papers from prior + papers fetched during experiments

  3. figure_embedding()
     MinIO artifact IDs → presigned URLs → LaTeX \includegraphics refs

  4. latex_render()
     Jinja2 template (arxiv format) → .tex file
     pdflatex → PDF
     Both stored in MinIO

  5. emit(PAPER_READY, {"tex_url": ..., "pdf_url": ...})
```

### 10.3 Methods Section Compilation (Deterministic)

```python
def compile_methods_section(hypothesis_id: str, db: Database) -> str:
    """
    Pure function. No LLM. Reads experiment_steps rows in order.
    Outputs structured text that is inserted directly into the LaTeX template.
    """
    steps = db.query(
        "SELECT * FROM experiment_steps WHERE hypothesis_id = ? ORDER BY created_at",
        hypothesis_id
    )

    lines = []
    for step in steps:
        if step.step_type == "tool_call":
            lines.append(f"Tool \\texttt{{{step.tool_name}}} was called with "
                         f"parameters {format_params(step.input_json)}. "
                         f"Execution completed in {step.duration_ms}ms.")
        elif step.step_type == "code_exec":
            lines.append(f"Custom computation was executed in an isolated sandbox "
                         f"(memory limit: {step.memory_mb}MB, timeout: {step.timeout_sec}s). "
                         f"Code is available in Appendix~\\ref{{appendix:{step.id}}}.")
        elif step.step_type == "llm_reasoning":
            lines.append(f"The agent evaluated intermediate results and "
                         f"\\textit{{{step.summary}}}.")  # summary field, not full LLM output

    return "\n".join(lines)
```

---

## 11. Data Layer

### 11.1 Storage Components

| Store | Purpose | What lives here |
|---|---|---|
| **Postgres** | Source of truth for all structured state | Sessions, hypotheses, experiment steps, tool calls, LLM calls, paper cache index, citation graph |
| **Qdrant** | Vector search for literature retrieval | Paper chunk embeddings with section + paper metadata |
| **Redis** | Task queue + event bus + short-lived cache | Celery queue, SSE event pub/sub, query result cache (TTL: 1hr) |
| **MinIO** | Binary artifact storage | PDFs, generated figures, compiled LaTeX, experiment output files |
| **SQLite** | BM25 index for keyword retrieval | Per-session inverted index over paper text |

### 11.2 Postgres Schema

```sql
-- Research sessions
CREATE TABLE research_sessions (
    id              UUID PRIMARY KEY,
    question        TEXT NOT NULL,
    status          TEXT NOT NULL,  -- pending | running | completed | failed
    stage           TEXT,           -- current stage for failure reporting
    prior_json      JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

-- Paper cache
CREATE TABLE papers (
    id              TEXT PRIMARY KEY,   -- arxiv_id
    title           TEXT,
    authors         JSONB,
    abstract        TEXT,
    pdf_url         TEXT,
    sections_json   JSONB,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    status          TEXT                -- processing | indexed | failed
);

CREATE TABLE paper_citations (
    source_paper_id TEXT REFERENCES papers(id),
    cited_paper_id  TEXT,
    PRIMARY KEY (source_paper_id, cited_paper_id)
);

-- Hypotheses
CREATE TABLE hypotheses (
    id              UUID PRIMARY KEY,
    session_id      UUID REFERENCES research_sessions(id),
    text            TEXT NOT NULL,
    test_methodology TEXT,
    scores_json     JSONB,          -- novelty, testability, impact, scope_fit, composite
    rank            INT,
    status          TEXT,           -- pending | running | completed | failed
    verdict         TEXT,           -- confirmed | refuted | inconclusive
    confidence      FLOAT,
    evidence_summary TEXT,
    key_finding     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Full experiment trace — every step of every sub-agent
CREATE TABLE experiment_steps (
    id              UUID PRIMARY KEY,
    hypothesis_id   UUID REFERENCES hypotheses(id),
    step_type       TEXT NOT NULL,  -- tool_call | code_exec | llm_reasoning
    step_index      INT,
    input_json      JSONB,
    output_json     JSONB,
    error_json      JSONB,
    duration_ms     INT,
    memory_mb       INT,            -- for code_exec steps
    timeout_sec     INT,            -- for code_exec steps
    summary         TEXT,           -- for llm_reasoning steps
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Tool calls (subset of experiment_steps, denormalized for query convenience)
CREATE TABLE tool_calls (
    id              UUID PRIMARY KEY,
    step_id         UUID REFERENCES experiment_steps(id),
    hypothesis_id   UUID REFERENCES hypotheses(id),
    tool_name       TEXT NOT NULL,
    domain          TEXT NOT NULL,
    params_json     JSONB,
    result_json     JSONB,
    success         BOOLEAN,
    duration_ms     INT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Every LLM call stored verbatim — full prompt + full response
CREATE TABLE llm_calls (
    id              UUID PRIMARY KEY,
    session_id      UUID REFERENCES research_sessions(id),
    hypothesis_id   UUID,
    call_purpose    TEXT,           -- "hypothesis_generation" | "tool_routing" | etc.
    model           TEXT,
    prompt_text     TEXT NOT NULL,  -- verbatim
    response_text   TEXT NOT NULL,  -- verbatim
    input_tokens    INT,
    output_tokens   INT,
    duration_ms     INT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Event log — every emitted event persisted
CREATE TABLE events (
    id              UUID PRIMARY KEY,
    session_id      UUID REFERENCES research_sessions(id),
    event_type      TEXT NOT NULL,
    source          TEXT,
    step            TEXT,
    hypothesis_id   UUID,
    parent_event_id UUID,
    payload_json    JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX events_session_idx ON events(session_id, created_at);
```

---

## 12. API & Streaming Layer

### 12.1 Endpoints

```
POST   /research                    Submit a research question → returns session_id
GET    /stream/{session_id}         SSE stream of all events for this session
GET    /sessions/{session_id}       Full session state (synchronous)
GET    /sessions/{session_id}/prior Prior JSON for a completed session
GET    /sessions/{session_id}/hypotheses  All hypotheses with scores and verdicts
GET    /sessions/{session_id}/trace Full experiment trace (all steps, all tools)
GET    /sessions/{session_id}/paper Paper PDF + .tex download URLs
GET    /sessions/{session_id}/llm-calls  All LLM prompts + responses for this session
GET    /tools                       List all available tools with specs
GET    /tools/{domain}              Tools for a specific domain
GET    /health                      Liveness check
```

### 12.2 POST /research

```json
// Request
{
  "question": "Does transformer attention efficiency degrade non-linearly with sequence length?",
  "config": {
    "max_hypotheses": 5,       // optional — default from env
    "paper_ttl_days": 30,      // optional — default from env
    "llm_model": "gpt-4o"      // optional — default from env
  }
}

// Response
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "stream_url": "/stream/550e8400-e29b-41d4-a716-446655440000",
  "status": "started"
}
```

### 12.3 SSE Stream Format

```
GET /stream/550e8400-e29b-41d4-a716-446655440000

data: {"event_id":"...","event_type":"session.started","step":"intake","payload":{"question":"..."}}

data: {"event_id":"...","event_type":"literature.paper_found","step":"literature.fetch","payload":{"arxiv_id":"2305.12345","title":"Efficient Transformers: A Survey"}}

data: {"event_id":"...","event_type":"llm.prompt","step":"literature.prior_build","payload":{"prompt":"You are a research prior builder..."}}

data: {"event_id":"...","event_type":"tool.called","step":"experiment.h1.step_3","hypothesis_id":"h1","payload":{"tool":"sequence_align","params":{...}}}

data: {"event_id":"...","event_type":"code.generated","step":"experiment.h2.step_1","hypothesis_id":"h2","payload":{"code":"import numpy as np\n..."}}

data: {"event_id":"...","event_type":"paper.ready","step":"paper","payload":{"pdf_url":"...","tex_url":"..."}}
```

---

## 13. Infrastructure & Setup

### 13.1 docker-compose.yml

```yaml
version: "3.9"

services:

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      - VITE_API_URL=http://localhost:8000

  api:
    build: ./services/api
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql://propab:propab@postgres:5432/propab
      - REDIS_URL=redis://redis:6379
    depends_on: [postgres, redis, qdrant, minio]

  orchestrator:
    build: ./services/orchestrator
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
      - LLM_MODEL=${LLM_MODEL:-gpt-4o}
      - EMBED_MODEL=${EMBED_MODEL:-text-embedding-3-small}
      - DATABASE_URL=postgresql://propab:propab@postgres:5432/propab
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
      - MINIO_URL=http://minio:9000
    depends_on: [postgres, redis, qdrant, minio]

  worker:
    build: ./services/worker
    command: celery -A propab.worker worker --loglevel=info --concurrency=${PROPAB_WORKERS:-3}
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://propab:propab@postgres:5432/propab
      - REDIS_URL=redis://redis:6379
      - MINIO_URL=http://minio:9000
      - SANDBOX_TIMEOUT_SEC=${SANDBOX_TIMEOUT_SEC:-30}
      - SANDBOX_MEMORY_MB=${SANDBOX_MEMORY_MB:-512}
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock   # for sandbox spawning
    depends_on: [postgres, redis]

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: propab
      POSTGRES_USER: propab
      POSTGRES_PASSWORD: propab
    volumes: [postgres_data:/var/lib/postgresql/data]

  redis:
    image: redis:7-alpine
    volumes: [redis_data:/data]

  qdrant:
    image: qdrant/qdrant:v1.7.0
    volumes: [qdrant_data:/qdrant/storage]

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: propab
      MINIO_ROOT_PASSWORD: propab_secret
    volumes: [minio_data:/data]

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
  minio_data:
```

### 13.2 Environment Variables

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | — | **Yes** | LLM + embedding calls |
| `LLM_PROVIDER` | `openai` | No | `openai` \| `anthropic` \| `ollama` |
| `LLM_MODEL` | `gpt-4o` | No | Model for orchestrator + agents |
| `EMBED_MODEL` | `text-embedding-3-small` | No | Embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | No | Local cross-encoder |
| `PROPAB_WORKERS` | `3` | No | Parallel experiment workers |
| `PROPAB_MAX_HYPOTHESES` | `5` | No | Hypotheses per question |
| `PAPER_TTL_DAYS` | `30` | No | Days before cached papers refresh |
| `SANDBOX_TIMEOUT_SEC` | `30` | No | Max CPU time for sandboxed code |
| `SANDBOX_MEMORY_MB` | `512` | No | Max memory for sandboxed code |

### 13.3 Setup

```bash
git clone https://github.com/propab/propab
cd propab
cp .env.example .env          # add your OPENAI_API_KEY
docker compose up
# → frontend: http://localhost:3000
# → api:      http://localhost:8000
# → api docs: http://localhost:8000/docs
```

---

## 14. Failure Handling

### 14.1 Failure Map

| Failure | Detection | Recovery | Event emitted |
|---|---|---|---|
| Literature fetch returns 0 papers | Empty results from arxiv + Semantic Scholar | Broaden query via LLM, retry once. If still 0: halt with `SESSION_FAILED`, explain in message | `SESSION_FAILED` |
| Prior builder returns malformed JSON | JSON parse fails | Retry with stricter prompt (schema injected). Max 2 retries. | `LLM_PARSE_ERROR` |
| Sub-agent tool call fails | `ToolResult.success == False` | Log error, continue to next step in plan if non-fatal | `TOOL_ERROR` |
| Sandbox code times out | `SANDBOX_TIMEOUT_SEC` exceeded | Emit `CODE_TIMEOUT`, retry with simplified code prompt (max 3x) | `CODE_TIMEOUT` |
| Sandbox code returns non-JSON | stdout parse fails | Emit `CODE_ERROR`, retry with explicit output contract reminder | `CODE_ERROR` |
| 3 sandbox retries exhausted | retry counter == 3 | Mark step inconclusive, attach error trace, continue | `CODE_ERROR {retry: 3}` |
| Sub-agent crashes | Celery task exception | Hypothesis marked `inconclusive`, full traceback stored, orchestrator continues with remaining hypotheses | `AGENT_FAILED` |
| All sub-agents return inconclusive | Ledger has 0 confirmed/refuted | Write paper with "inconclusive" framing — report what was tried and what failed | `SESSION_COMPLETED` (special framing) |
| LLM rate limit hit | HTTP 429 | Exponential backoff: 2s, 4s, 8s, 16s, give up at 5 retries | `LLM_PROMPT` retry count |
| LaTeX compilation fails | pdflatex exit code != 0 | Return .tex source with compile error attached. Frontend offers raw download. | `PAPER_LATEX_COMPILED {success: false}` |

### 14.2 Failure Principle

> A research session never silently fails. If something goes wrong, the system continues as far as it can, documents every failure in the event stream, and produces the best output it can with what succeeded. A partial result with honest failure documentation is more valuable than an aborted run.

---

## 15. Open Problems (v2 Backlog)

These are known hard problems deliberately deferred from v1.

| # | Problem | Why deferred | Proposed v2 direction |
|---|---|---|---|
| 1 | **Cross-hypothesis memory sharing** | Sub-agents share no state during execution. Interim findings from one agent could help others. | Result ledger broadcast: orchestrator pushes interim verdicts to running sub-agents at checkpoints. Needs bias-cascade analysis first. |
| 2 | **Dataset access** | Many STEM hypotheses require actual datasets. Where do they come from? | Phase 1: curated public datasets as optional add-ons (UCI ML, NCBI, TCGA, NASA). Phase 2: data source plugin interface. |
| 3 | **Claim grounding / hallucination guard** | Paper claims could be confabulated rather than grounded in trace. | Automated grounding pass: each paper claim matched to a specific `experiment_steps` row. Ungrounded claims flagged for human review before paper is finalized. |
| 4 | **Multi-session knowledge accumulation** | Each session is isolated. Prior work from previous sessions is not reused. | Shared knowledge base with provenance tracking across sessions. |
| 5 | **Tool quality gate** | Community-contributed tools may be low quality or unsafe. | Submission requires: typed `TOOL_SPEC`, unit tests, sandbox execution proof, code review before registry merge. |
| 6 | **Local model support** | v1 requires LLM API key. | Ollama integration via `LLM_PROVIDER=ollama`. The LLM client is already abstracted — only the client adapter needs writing. |

---

## 16. Build Roadmap

### Phase 1 — Foundation (Weeks 1–4)
- [ ] Monorepo structure, Docker Compose skeleton, all services stubbed with health checks
- [ ] Postgres schema + migrations (Alembic)
- [ ] Event system: `emit()` → Redis pub/sub → Postgres persist
- [ ] SSE endpoint: Redis subscribe → fan-out to client
- [ ] arxiv API client + PDF parser (PyMuPDF)
- [ ] Paper chunker + Qdrant upsert
- [ ] BM25 index (Rank-BM25 + SQLite)
- [ ] Citation graph extraction + Postgres edges
- [ ] Lazy cache logic with TTL check

### Phase 2 — Retrieval + Prior (Weeks 5–6)
- [ ] Query expansion (LLM call)
- [ ] Hybrid retrieval: dense + sparse + citation graph
- [ ] Reciprocal Rank Fusion
- [ ] Cross-encoder reranker (local, CPU)
- [ ] Prior builder: LLM synthesis → structured `Prior` JSON
- [ ] Short-circuit answer detection
- [ ] All literature events wired and tested

### Phase 3 — Agent Core (Weeks 7–10)
- [ ] Orchestrator research loop (explicit async Python)
- [ ] Hypothesis generator + ranker
- [ ] Sub-agent harness + Celery task
- [ ] Domain routing (one LLM call)
- [ ] Tool registry: scanner + cluster loader + `call()`
- [ ] Sandbox executor: Docker-in-Docker + resource caps + retry logic
- [ ] Result ledger + synthesis logic
- [ ] First 40 STEM tools across 5 domains (with `TOOL_SPEC` + unit tests)

### Phase 4 — Output + Frontend (Weeks 11–14)
- [ ] Trace compiler (deterministic methods section)
- [ ] Paper section generator (one LLM call per section)
- [ ] Jinja2 arxiv LaTeX template
- [ ] pdflatex compilation + MinIO storage
- [ ] React frontend: submit form + SSE event viewer + result explorer
- [ ] Paper download (PDF + .tex)
- [ ] LLM call inspector (view all prompts + responses for a session)
- [ ] One-command setup validation + README

### Phase 5 — Community Hardening (Post-launch)
- [ ] Ollama / local model support
- [ ] Tool plugin submission process + review checklist
- [ ] Dataset connectors (UCI ML, NCBI)
- [ ] Claim grounding / hallucination guard (v2 open problem #3)

---

## 17. Repository Structure

```
propab/
├── docker-compose.yml
├── .env.example
├── README.md
├── ARCHITECTURE.md               ← this document
│
├── services/
│   ├── api/                      # FastAPI gateway
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── research.py
│   │   │   ├── stream.py
│   │   │   └── sessions.py
│   │   └── Dockerfile
│   │
│   ├── orchestrator/             # Main research loop
│   │   ├── research_loop.py      # the explicit async state machine
│   │   ├── intake.py
│   │   ├── literature.py
│   │   ├── hypotheses.py
│   │   ├── experiments.py
│   │   ├── synthesis.py
│   │   ├── paper.py
│   │   └── Dockerfile
│   │
│   └── worker/                   # Celery sub-agent workers
│       ├── sub_agent_loop.py
│       ├── sandbox.py
│       └── Dockerfile
│
├── packages/
│   └── propab-core/              # shared Python package
│       ├── events.py             # EventType enum + emit() + PropabEvent
│       ├── types.py              # all shared TypedDicts
│       ├── llm.py                # LLM client abstraction (openai/anthropic/ollama)
│       ├── db.py                 # Postgres + Qdrant + Redis clients
│       └── tools/
│           ├── registry.py       # ToolRegistry scanner + cluster loader
│           ├── types.py          # ToolResult, ToolError, TOOL_SPEC shape
│           ├── computational_biology/
│           │   ├── sequence_align.py        # fn + TOOL_SPEC
│           │   ├── protein_fold_predict.py
│           │   ├── gc_content.py
│           │   └── ...
│           ├── mathematics/
│           ├── statistics/
│           ├── ml_modeling/
│           ├── data_analysis/
│           └── general_computation/
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Submit.tsx
│   │   │   ├── Session.tsx       # live event stream view
│   │   │   ├── Results.tsx
│   │   │   └── Inspector.tsx     # LLM call + tool call inspector
│   │   └── components/
│   └── Dockerfile
│
└── migrations/
    └── versions/                 # Alembic migrations
```

---

## Appendix A — Tool Writing Guide

To add a new tool:

1. Create a file in `packages/propab-core/propab/tools/{domain}/your_tool.py`
2. Define `TOOL_SPEC` dict (see Section 8.1 for exact schema)
3. Define a function with the same name as `TOOL_SPEC["name"]`
4. Function must return `ToolResult`
5. Write a unit test in `tests/tools/test_your_tool.py`
6. Run `propab tools validate your_tool` to confirm registry picks it up

The registry scanner will discover it automatically. No registration step needed.

---

## Appendix B — LLM Client Abstraction

```python
# packages/propab-core/propab/llm.py

class LLMClient:
    """
    Single interface regardless of provider.
    Automatically stores every call in llm_calls table.
    Automatically emits LLM_PROMPT and LLM_RESPONSE events.
    """
    def __init__(self, provider: str, model: str, db: Database, emit_fn):
        self.provider = provider
        self.model = model
        self.db = db
        self.emit = emit_fn

    async def call(self, prompt: str, purpose: str,
                   session_id: str, hypothesis_id: str | None = None,
                   response_schema: type | None = None) -> str:

        self.emit(LLM_PROMPT, {"prompt": prompt, "purpose": purpose, "model": self.model})

        response = await self._call_provider(prompt)

        self.emit(LLM_RESPONSE, {"response": response, "purpose": purpose})

        self.db.insert("llm_calls", {
            "session_id": session_id,
            "hypothesis_id": hypothesis_id,
            "call_purpose": purpose,
            "model": self.model,
            "prompt_text": prompt,
            "response_text": response,
            ...
        })

        return response

    async def _call_provider(self, prompt: str) -> str:
        if self.provider == "openai":
            # openai client call
        elif self.provider == "anthropic":
            # anthropic client call
        elif self.provider == "ollama":
            # ollama client call
```

Every LLM call in the system goes through this client. No direct API calls anywhere else. Every prompt is stored. Every response is stored. Every call emits two events.

---

*This document is the authoritative architecture reference for Propab. Implementation decisions that diverge from this spec must be recorded here with rationale before the divergence is merged.*