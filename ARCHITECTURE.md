# PROPAB — Open Source AI Research System
## System Architecture & Engineering Design

> **Transparency is a first-class constraint.** Every agent decision, every tool call, every LLM prompt, every result, every failure emits a named structured event. Nothing happens silently. Researchers can inspect exactly what Propab did and why at every step.

> **Long-running agent vision.** Propab is designed to work for hours or a full day on a hard research question. Given sufficient compute and time, it should make progressive, verifiable progress — forming and testing thousands of hypotheses, accumulating evidence across rounds, and producing a paper with real findings. This document specifies all architecture needed to reach that goal.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Design Principles](#2-design-principles)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Event System — The Transparency Backbone](#4-event-system--the-transparency-backbone)
5. [Literature Layer](#5-literature-layer)
6. [Hypothesis Layer](#6-hypothesis-layer)
7. [Experiment Layer — Think-Act Sub-Agent Architecture](#7-experiment-layer--think-act-sub-agent-architecture)
8. [Tool System](#8-tool-system)
9. [Orchestrator — Multi-Round Research Loop](#9-orchestrator--multi-round-research-loop)
10. [Research Memory System](#10-research-memory-system)
11. [Budget & Progress Management](#11-budget--progress-management)
12. [Evidence Accumulation & Significance Gate](#12-evidence-accumulation--significance-gate)
13. [Paper Writing Layer](#13-paper-writing-layer)
14. [Data Layer](#14-data-layer)
15. [API & Streaming Layer](#15-api--streaming-layer)
16. [Infrastructure & Setup](#16-infrastructure--setup)
17. [Failure Handling](#17-failure-handling)
18. [Build Roadmap](#18-build-roadmap)
19. [Repository Structure](#19-repository-structure)

---

## 1. System Overview

Propab is an autonomous long-running research system. A researcher submits a question. The system:

1. Determines whether the answer already exists clearly in literature
2. If not — generates N ranked hypotheses
3. Runs a **multi-round experiment loop**: spawns sub-agents, collects evidence, synthesizes findings, generates refined hypotheses, repeats
4. Each sub-agent uses a **think-act loop**: it reasons after every tool call and decides what to do next based on what it just learned
5. All agents share a **live result ledger**: partial findings from one agent are visible to others
6. The system continues until a **stopping criterion** is reached: time budget exhausted, sufficient confirmed findings, or diminishing returns detected
7. Writes an arxiv-formatted paper grounded in the accumulated multi-round experiment trace

**Target user:** Any researcher — PhD student, independent researcher, lab team — who can run `docker compose up`.

**Long-running mode:** A session may run for hours or a full day. It checkpoints progress to Postgres at every round. If interrupted, it can resume from the last checkpoint. Given compute and time, it makes progressive research progress.

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
- Every round of the research loop is a named, queryable entity in the database

### 2.3 Fail Loudly
No silent failures. Every error is a structured object with: error type, message, step context, input that caused it, and recovery action taken. Errors are surfaced to the frontend in real time. Sessions never abort silently — they checkpoint, emit a structured failure event, and the researcher can inspect exactly where and why.

### 2.4 Lazy-First Data
No bulk pre-ingestion of arxiv. Papers are fetched on demand, processed, and cached with a TTL. First queries fetch fresh. Repeat queries hit cache. This makes local setup feasible.

### 2.5 Tool Discipline
Ship high-quality, well-tested STEM tools across domains. Expose a plugin interface for community additions. Tools are loaded by cluster (15–25 tools per domain) — never the full list — so context never explodes. Tool selection accounts for both hypothesis text and the original research question to prevent drift.

### 2.6 One Command Setup
`docker compose up` starts everything. No manual config of vector DBs, queues, or model endpoints. All configuration is via environment variables with documented defaults.

### 2.7 Evidence Before Claims
No hypothesis can be confirmed or refuted without passing the **significance gate**: at least one tool call that produces a p-value, effect size, or confidence interval. Raw metric values alone are insufficient for confirmation. This is enforced in code, not just convention.

### 2.8 Progressive Research
The system does not stop after one pass. It accumulates evidence across rounds, learns from dead ends, refines hypotheses, and continues making progress. Diminishing-returns detection prevents infinite loops. Budget controls prevent runaway cost.

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PROPAB SYSTEM                                  │
│                                                                             │
│  ┌──────────────┐      ┌────────────────────────────────────────────────┐   │
│  │   Frontend   │─────▶│         API Gateway  (FastAPI)                  │   │
│  │  React + SSE │◀─────│  POST /research  GET /stream/{session_id}       │   │
│  └──────────────┘      └──────────────────┬─────────────────────────────┘   │
│                                           │                                 │
│                    ┌──────────────────────▼────────────────────────────┐    │
│                    │           ORCHESTRATOR SERVICE                     │    │
│                    │                                                    │    │
│                    │  research_loop()  — MULTI-ROUND                    │    │
│                    │  ├── intake()              parse + classify         │    │
│                    │  ├── literature()           build prior             │    │
│                    │  ├── [ROUND LOOP]                                   │    │
│                    │  │   ├── hypothesize()      generate + rank         │    │
│                    │  │   ├── experiment()        dispatch think-act     │    │
│                    │  │   │   agents (parallel)                          │    │
│                    │  │   ├── synthesize()        collect + evaluate     │    │
│                    │  │   ├── check_budget()      continue or stop?      │    │
│                    │  │   └── refine()            learn → new hypotheses │    │
│                    │  └── write_paper()           compile + render       │    │
│                    │                                                    │    │
│                    │  emit(event) ──────────────────────────────────▶   │    │
│                    └──────────┬─────────────────────────────────────────┘    │
│                               │                                             │
│         ┌─────────────────────┼──────────────────────┐                      │
│         │                     │                      │                      │
│  ┌──────▼──────────┐  ┌───────▼──────────┐  ┌────────▼───────────────────┐  │
│  │  LITERATURE     │  │  HYPOTHESIS      │  │  EXPERIMENT RUNNER         │  │
│  │  SERVICE        │  │  SERVICE         │  │  (N Celery workers)        │  │
│  │                 │  │                  │  │  think_act_loop()          │  │
│  │  hybrid_search()│  │  generate()      │  │  ├── decide_next_step()    │  │
│  │  build_prior()  │  │  rank()          │  │  ├── tool_dispatch()       │  │
│  │  detect_gaps()  │  │  refine()        │  │  ├── observe_result()      │  │
│  └──────┬──────────┘  │  deduplicate()   │  │  ├── significance_check()  │  │
│         │             └───────────────── ┘  │  └── emit(event)           │  │
│         │                                   └───────────────────────────┘  │
│         │                                                                   │
│  ┌──────▼──────────────────────────────────────────────────────────────┐    │
│  │                     RESEARCH MEMORY SYSTEM                           │    │
│  │                                                                      │    │
│  │  Session Memory     Cross-Session KB     Agent Working Memory        │    │
│  │  (this run)         (persists forever)   (per sub-agent context)     │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                     SHARED INFRASTRUCTURE                             │    │
│  │                                                                       │    │
│  │  Postgres · Redis · Qdrant · MinIO · Docker sandbox                  │    │
│  │  EVENT BUS (Redis pub/sub) ◀── all services emit here               │    │
│  └───────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Service Map

| Service | Responsibility | Stack |
|---|---|---|
| API Gateway | HTTP + SSE endpoints, session management, event fan-out | FastAPI, uvicorn |
| Orchestrator | Multi-round research loop, state machine, synthesis, budget control | Python asyncio |
| Literature Service | Arxiv fetch, PDF parse, hybrid index, prior builder | Python, Qdrant, BM25 |
| Hypothesis Service | Generate + rank + refine + deduplicate hypotheses across rounds | LLM API |
| Experiment Runner | Think-act sub-agents, significance gate, tool dispatch, sandboxed code | Celery, Docker |
| Tool Registry | Load tool clusters by domain + question context, plugin discovery | Python package |
| Research Memory | Session memory, cross-session KB, agent working memory | Postgres, Qdrant |
| Budget Manager | Time/compute/quality budgets, stopping criteria, progress tracking | Python |
| Paper Writer | Multi-round trace compiler + section generator + LaTeX render | Jinja2, pdflatex |
| Frontend | Submit jobs, live event stream, round progress, result explorer | React, TailwindCSS |
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
    round_id: str | None         # which research round this belongs to
```

### 4.2 Event Type Registry

```python
class EventType(str, Enum):

    # ── Session lifecycle ──────────────────────────────────────────────
    SESSION_STARTED          = "session.started"
    SESSION_COMPLETED        = "session.completed"
    SESSION_FAILED           = "session.failed"
    SESSION_RESUMED          = "session.resumed"       # long-running resume from checkpoint

    # ── Intake ────────────────────────────────────────────────────────
    INTAKE_PARSED            = "intake.parsed"
    INTAKE_DECOMPOSED        = "intake.decomposed"

    # ── Literature ────────────────────────────────────────────────────
    LIT_FETCH_STARTED        = "literature.fetch_started"
    LIT_PAPER_FOUND          = "literature.paper_found"
    LIT_PAPER_CACHED         = "literature.paper_cached"
    LIT_PAPER_PARSED         = "literature.paper_parsed"
    LIT_PAPER_INDEXED        = "literature.paper_indexed"
    LIT_RETRIEVAL_QUERY      = "literature.retrieval_query"
    LIT_RETRIEVAL_RESULTS    = "literature.retrieval_results"
    LIT_PRIOR_BUILT          = "literature.prior_built"
    LIT_ANSWER_FOUND         = "literature.answer_found"

    # ── Round lifecycle ────────────────────────────────────────────────
    ROUND_STARTED            = "round.started"         # new research round begins
    ROUND_COMPLETED          = "round.completed"       # round finished, ledger updated
    ROUND_SKIPPED            = "round.skipped"         # budget/diminishing-returns skip

    # ── Hypothesis ────────────────────────────────────────────────────
    HYPO_GENERATED           = "hypothesis.generated"
    HYPO_RANKED              = "hypothesis.ranked"
    HYPO_DISPATCHED          = "hypothesis.dispatched"
    HYPO_REFINED             = "hypothesis.refined"    # existing hypothesis narrowed/sharpened
    HYPO_DEDUPLICATED        = "hypothesis.deduplicated" # duplicate pruned before dispatch
    HYPO_PROMOTED            = "hypothesis.promoted"   # inconclusive hypo re-queued with refinement
    HYPO_RETIRED             = "hypothesis.retired"    # dead end, never to be retried

    # ── Sub-agent lifecycle ───────────────────────────────────────────
    AGENT_STARTED            = "agent.started"
    AGENT_PLAN_CREATED       = "agent.plan_created"
    AGENT_STEP_STARTED       = "agent.step_started"
    AGENT_STEP_COMPLETED     = "agent.step_completed"
    AGENT_STEP_FAILED        = "agent.step_failed"
    AGENT_DECIDED_NEXT_STEP  = "agent.decided_next_step"  # think-act decision event
    AGENT_DECIDED_STOP       = "agent.decided_stop"       # agent chose to stop early
    AGENT_COMPLETED          = "agent.completed"
    AGENT_FAILED             = "agent.failed"

    # ── Tool calls ────────────────────────────────────────────────────
    TOOL_SELECTED            = "tool.selected"
    TOOL_CALLED              = "tool.called"
    TOOL_RESULT              = "tool.result"
    TOOL_ERROR               = "tool.error"

    # ── Significance gate ─────────────────────────────────────────────
    SIG_GATE_PASSED          = "significance.gate_passed"   # p<0.05 / effect_size found
    SIG_GATE_FAILED          = "significance.gate_failed"   # not enough statistical evidence
    SIG_GATE_BYPASSED        = "significance.gate_bypassed" # refuted path, gate not required

    # ── Code execution ────────────────────────────────────────────────
    CODE_GENERATED           = "code.generated"
    CODE_SUBMITTED           = "code.submitted"
    CODE_RESULT              = "code.result"
    CODE_ERROR               = "code.error"
    CODE_TIMEOUT             = "code.timeout"

    # ── LLM calls ─────────────────────────────────────────────────────
    LLM_PROMPT               = "llm.prompt"
    LLM_RESPONSE             = "llm.response"
    LLM_PARSE_ERROR          = "llm.parse_error"

    # ── Cross-agent memory ────────────────────────────────────────────
    MEMORY_LEDGER_BROADCAST  = "memory.ledger_broadcast"  # partial results pushed to agents
    MEMORY_PEER_FINDING      = "memory.peer_finding"      # agent received peer's result
    MEMORY_KB_WRITTEN        = "memory.kb_written"        # cross-session KB updated

    # ── Budget & progress ─────────────────────────────────────────────
    BUDGET_CHECKPOINT        = "budget.checkpoint"        # budget state at each round
    BUDGET_EXHAUSTED         = "budget.exhausted"         # hard stop: budget hit
    PROGRESS_DIMINISHING     = "progress.diminishing"     # soft stop: returns declining
    PROGRESS_MILESTONE       = "progress.milestone"       # noteworthy finding confirmed

    # ── Synthesis ─────────────────────────────────────────────────────
    SYNTH_RESULT_RECEIVED    = "synthesis.result_received"
    SYNTH_LEDGER_UPDATED     = "synthesis.ledger_updated"
    SYNTH_BREAKTHROUGH       = "synthesis.breakthrough"
    SYNTH_DEAD_END           = "synthesis.dead_end"
    SYNTH_ALL_INCONCLUSIVE   = "synthesis.all_inconclusive" # warning when all results weak

    # ── Paper ─────────────────────────────────────────────────────────
    PAPER_TRACE_COMPILED     = "paper.trace_compiled"
    PAPER_SECTION_STARTED    = "paper.section_started"
    PAPER_SECTION_COMPLETED  = "paper.section_completed"
    PAPER_LATEX_COMPILED     = "paper.latex_compiled"
    PAPER_READY              = "paper.ready"
    PAPER_SKIPPED            = "paper.skipped"
```

---

## 5. Literature Layer

### 5.1 Ingestion Strategy — Lazy Cache

Papers are fetched on demand and cached with a configurable TTL. No bulk pre-ingestion. This makes local setup feasible on a laptop.

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

### 5.2 PDF Processing Pipeline

```python
def process_paper(arxiv_id: str, session_id: str) -> ProcessedPaper:
    emit(LIT_PAPER_FOUND, {"arxiv_id": arxiv_id, "source": "arxiv_api"})
    pdf_bytes = fetch_pdf(arxiv_id)

    emit(LIT_PAPER_PARSED, {"arxiv_id": arxiv_id, "section_count": N})
    sections = parse_pdf_sections(pdf_bytes)

    chunks = chunk_sections(sections, size=512, overlap=64)
    embeddings = embed_batch(chunks)
    qdrant.upsert(collection="papers", points=build_points(chunks, embeddings))

    bm25_index.add(paper_id=arxiv_id, text=full_text(sections))
    refs = extract_references(sections["references"])
    postgres.insert_citations(source=arxiv_id, cited=refs)

    emit(LIT_PAPER_INDEXED, {"arxiv_id": arxiv_id})
    return ProcessedPaper(...)
```

### 5.3 Hybrid Retrieval Pipeline

```
Query: "does attention efficiency degrade non-linearly with sequence length?"
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  QUERY EXPANSION  (1 LLM call, emits LIT_RETRIEVAL_QUERY)  │
│                                                            │
│  Original + 3 rephrasings + key concept extraction         │
└──────────────────────────┬─────────────────────────────────┘
                           │
         ┌─────────────────┼──────────────────┐
  ┌──────▼──────┐   ┌──────▼──────┐   ┌───────▼───────┐
  │  DENSE      │   │  SPARSE     │   │  CITATION     │
  │  Qdrant     │   │  BM25       │   │  GRAPH        │
  │  top-40     │   │  top-40     │   │  2-hop walk   │
  └──────┬──────┘   └──────┬──────┘   └───────┬───────┘
         │                 │                  │
  ┌──────▼─────────────────▼──────────────────▼────────┐
  │  RECIPROCAL RANK FUSION                            │
  └──────────────────────────┬─────────────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────┐
  │  RERANKER  (cross-encoder, runs locally)            │
  │  Model: cross-encoder/ms-marco-MiniLM-L-6-v2        │
  └──────────────────────────┬─────────────────────────┘
                             │
                        Top-K chunks (default K=20)
                        → Prior Builder
```

### 5.4 Prior Builder

```python
class Prior(TypedDict):
    established_facts: list[Claim]
    contested_claims:  list[Dispute]
    open_gaps:         list[Gap]
    dead_ends:         list[DeadEnd]
    key_papers:        list[PaperRef]

class Claim(TypedDict):
    text:       str
    confidence: float
    paper_ids:  list[str]

class Gap(TypedDict):
    text:        str
    source_paper: str
    gap_type:    str  # "unanswered_question" | "missing_data" | "untested_assumption"
```

**Short-circuit:** If cosine similarity > 0.92 between question embedding and an established claim embedding, emit `LIT_ANSWER_FOUND` and return without generating hypotheses.

---

## 6. Hypothesis Layer

### 6.1 Generation (per round)

Hypotheses are generated fresh each round, but each round receives the accumulated context from prior rounds: what was tried, what was learned, what dead ends were found.

```python
HYPOTHESIS_PROMPT = """
You are a research hypothesis generator working on round {round_number} of a multi-round study.

Research question: {question}

Prior — established facts:
{established_facts}

Prior — open gaps:
{open_gaps}

Prior — dead ends (do not repeat these):
{dead_ends}

Results from previous rounds (confirmed/refuted/inconclusive with reasons):
{prior_round_findings}

Generate exactly {N} hypotheses. Requirements:
- Specific and falsifiable
- Not duplicate any established fact or prior-round confirmed finding
- Not repeat any dead end or prior-round refuted hypothesis
- State its expected test methodology in one sentence, naming specific tools
- Reference at least one open gap from the prior or an unresolved prior-round finding
- For round > 1: hypotheses should be more targeted than round 1, informed by what was learned

Return JSON array only. Schema:
[{
  "id": "h1",
  "text": "...",
  "test_methodology": "...",
  "gap_reference": "gap text from prior or prior round",
  "expected_result": "...",
  "refinement_of": "prior hypothesis id if this refines a prior-round inconclusive, else null"
}]
"""
```

**Validation gate:** Reject any hypothesis whose text matches `"^Hypothesis \\d+:"` (generic fallback template). Retry generation up to 2 times with stricter prompt before accepting.

### 6.2 Ranking

| Dimension | Method | Weight |
|---|---|---|
| Novelty | Embedding distance from established_facts centroid + prior confirmed findings | 30% |
| Testability | LLM score: can this be tested with available tools? Does it name specific tools? | 30% |
| Potential impact | LLM score: significance if confirmed relative to gap importance | 25% |
| Scope fit | LLM score: appropriately scoped for one session? | 15% |

### 6.3 Hypothesis Lifecycle at Scale

In long-running mode the system may generate and manage hundreds or thousands of hypotheses across rounds. The lifecycle tracks each one:

```
GENERATED → RANKED → DISPATCHED → [RUNNING] → COMPLETED
                                                  │
                              ┌───────────────────┴──────────────┐
                           CONFIRMED                        INCONCLUSIVE/REFUTED
                              │                                   │
                         RETIRED                    ┌─────────────┴──────────┐
                         (write paper)         PROMOTED               RETIRED
                                               (refine +              (dead end)
                                                re-queue)
```

```python
class HypothesisStatus(str, Enum):
    PENDING       = "pending"
    RUNNING       = "running"
    COMPLETED     = "completed"
    FAILED        = "failed"
    PROMOTED      = "promoted"   # inconclusive → refined version queued for next round
    RETIRED       = "retired"    # dead end, not to be retried

class HypothesisVerdict(str, Enum):
    CONFIRMED     = "confirmed"
    REFUTED       = "refuted"
    INCONCLUSIVE  = "inconclusive"
```

**Promotion logic:** An inconclusive hypothesis is promoted (not just discarded) if its confidence was > 0.25 and its evidence showed partial signal. The promoted version carries a `refinement_of` pointer and a `learned_from` summary so the next-round agent starts with context.

**Retirement logic:** A hypothesis is retired (never retried) if: (a) it was refuted, (b) it was inconclusive for 2+ rounds with no evidence improvement, or (c) it duplicates a confirmed finding.

### 6.4 Deduplication

Before dispatching a new round of hypotheses, run embedding-based deduplication against all prior-round hypotheses (confirmed, refuted, and inconclusive). Cosine similarity > 0.88 → flag as duplicate → emit `HYPO_DEDUPLICATED` → replace with next-ranked hypothesis.

---

## 7. Experiment Layer — Think-Act Sub-Agent Architecture

### 7.1 Orchestrator ↔ Sub-Agent Contract

**What the orchestrator sends to each sub-agent:**

```python
class ExperimentTask(TypedDict):
    session_id:      str
    round_id:        str
    hypothesis_id:   str
    hypothesis:      RankedHypothesis
    prior:           Prior               # read-only reference
    available_tools: list[str]           # tool names for declared domain
    resource_limits: ResourceLimits      # sandbox caps
    peer_findings:   list[PeerFinding]   # partial results from other agents this round
    learned_from:    str | None          # prior-round learning if this is a promoted hypothesis
    budget:          AgentBudget         # time/steps budget for this agent
```

**What the sub-agent returns:**

```python
class ExperimentResult(TypedDict):
    hypothesis_id:   str
    round_id:        str
    verdict:         Literal["confirmed", "refuted", "inconclusive"]
    confidence:      float           # 0–1
    evidence_summary: str            # 2–3 sentences
    key_finding:     str | None      # one-line breakthrough if confirmed
    significance:    SignificanceResult | None   # p_value / effect_size / CI
    tool_trace_id:   str
    figures:         list[str]
    duration_sec:    float
    failure_reason:  str | None
    recommend_retry: bool            # agent's own recommendation: worth re-running?
    learned:         str | None      # what this agent learned that would help future agents
```

### 7.2 Think-Act Loop (Core Architecture Change)

The sub-agent does **not** execute a pre-built plan. It decides the next action after observing each result. This is the key difference between a recipe-executor and an actual reasoner.

```python
async def think_act_loop(task: ExperimentTask) -> ExperimentResult:
    """
    Agent observes results after each step and decides what to do next.
    No pre-built plan. The LLM chooses each action based on accumulated context.
    """
    emit(AGENT_STARTED, {"hypothesis_id": task.hypothesis_id})

    context = AgentContext(
        hypothesis=task.hypothesis,
        prior=task.prior,
        peer_findings=task.peer_findings,
        learned_from=task.learned_from,
        results_so_far=[],
        steps_taken=0,
    )

    # Initial plan: first 1–2 tool calls chosen upfront (domain-routed, question-aware)
    # This gives the agent a starting direction without over-committing
    initial_steps = await plan_initial_steps(task, context)

    while not _should_stop(context, task.budget):
        # THINK: what should I do next, given everything I know so far?
        next_action = await decide_next_action(task, context, llm)
        emit(AGENT_DECIDED_NEXT_STEP, {"action": next_action, "reasoning": next_action.reasoning})

        if next_action.action_type == "stop":
            emit(AGENT_DECIDED_STOP, {"reason": next_action.reasoning})
            break

        # ACT: execute the chosen action
        if next_action.action_type == "tool":
            result = await execute_tool(next_action.tool_name, next_action.params, task)
        elif next_action.action_type == "code":
            result = await execute_sandbox(next_action.code, task)

        context.results_so_far.append(result)
        context.steps_taken += 1

        # After each result: check if significance gate can be passed
        sig = check_significance(context.results_so_far)
        if sig.gate_passed:
            emit(SIG_GATE_PASSED, {"p_value": sig.p_value, "effect_size": sig.effect_size})
            # Significance found — still continue to corroborate, but verdict is now determinable
        elif sig.gate_definitively_failed:
            # Strong evidence of no effect — can emit refuted, continue to verify
            pass

    # Evaluate final verdict based on accumulated results
    verdict = evaluate_accumulated_evidence(context.results_so_far, task.hypothesis)
    emit(AGENT_COMPLETED, {"verdict": verdict.verdict, "confidence": verdict.confidence})

    return build_experiment_result(task, verdict, context)


async def decide_next_action(task: ExperimentTask, context: AgentContext, llm: LLMClient) -> AgentAction:
    """
    Core think step. LLM observes context and chooses next action.
    Prompt includes: hypothesis, results so far, available tools, peer findings, budget remaining.
    """
    prompt = build_think_prompt(
        hypothesis=task.hypothesis,
        results_so_far=context.results_so_far,
        available_tools=task.available_tools,
        peer_findings=task.peer_findings,
        steps_taken=context.steps_taken,
        max_steps=task.budget.max_steps,
        significance_status=check_significance(context.results_so_far),
        learned_from=task.learned_from,
    )
    response = await llm.call(prompt, purpose="agent.decide_next_step", ...)
    return parse_agent_action(response)


def _should_stop(context: AgentContext, budget: AgentBudget) -> bool:
    if context.steps_taken >= budget.max_steps:
        return True
    if time.monotonic() >= budget.deadline:
        return True
    sig = check_significance(context.results_so_far)
    if sig.gate_passed and context.steps_taken >= budget.min_steps_after_significance:
        return False  # significance found but still want corroboration
    return False
```

**Think prompt contract:**

```
THINK_PROMPT = """
You are a research sub-agent testing this hypothesis:
"{hypothesis_text}"

Test methodology: {test_methodology}
Learned from prior work: {learned_from or "none"}

Results so far ({steps_taken} of max {max_steps} steps):
{results_summary}

Current significance status:
  p_value: {p_value or "not yet measured"}
  effect_size: {effect_size or "not yet measured"}
  Has significance gate passed: {gate_passed}

Peer agents in this round found:
{peer_findings_summary or "none yet"}

Available tools: {available_tools_with_descriptions}

What should you do next? Choose ONE action:
1. Call a specific tool (name the tool and its parameters)
2. Write custom code (describe what the code should compute)
3. Stop (if you have sufficient evidence or additional steps won't help)

If stopping, explain why.
If continuing, explain what you expect to learn from the next action.

Return JSON only:
{
  "action_type": "tool" | "code" | "stop",
  "tool_name": "...",   // if action_type == "tool"
  "params": {...},      // if action_type == "tool"
  "code_description": "...",  // if action_type == "code"
  "reasoning": "one sentence explaining why this action next",
  "expected_outcome": "what result would confirm/refute/clarify"
}
"""
```

### 7.3 Significance Gate (Enforced in Code)

This is non-negotiable. Before any `confirmed` or `refuted` verdict is possible, the significance gate must be evaluated.

```python
@dataclass
class SignificanceResult:
    gate_passed: bool
    gate_definitively_failed: bool
    p_value: float | None
    effect_size: float | None
    confidence_interval: list[float] | None
    n_observations: int
    method: str | None   # "p_value" | "effect_size" | "confidence_interval"


def check_significance(results: list[StepResult]) -> SignificanceResult:
    """
    Scan accumulated results for any statistical evidence.
    Gate passes if ANY of: p < 0.05, |effect_size| > 0.2, non-overlapping 95% CI.
    Gate definitively fails if statistical tests ran and produced null/contrary results.
    Gate is pending if no statistical test has run yet.
    """
    p_value = _find_p_value(results)
    effect_size = _find_effect_size(results)
    ci = _find_confidence_interval(results)

    gate_passed = (
        (p_value is not None and p_value < 0.05)
        or (effect_size is not None and abs(effect_size) > 0.2)
        or (ci is not None and not _intervals_overlap(ci, [0.0, 0.0]))
    )

    gate_definitively_failed = (
        (p_value is not None and p_value >= 0.3)
        and (effect_size is None or abs(effect_size) < 0.05)
    )

    return SignificanceResult(
        gate_passed=gate_passed,
        gate_definitively_failed=gate_definitively_failed,
        p_value=p_value,
        effect_size=effect_size,
        confidence_interval=ci,
        n_observations=_count_observations(results),
        method=_identify_method(p_value, effect_size, ci),
    )


def compute_verdict(hypothesis: Hypothesis, sig: SignificanceResult, results: list) -> HypothesisVerdict:
    """
    Verdict logic. Significance gate is the entry condition for confirmed/refuted.
    Without significance evidence, verdict is always inconclusive.
    """
    if not sig.gate_passed and not sig.gate_definitively_failed:
        emit(SIG_GATE_FAILED, {"reason": "no significance evidence produced"})
        return HypothesisVerdict(verdict="inconclusive", confidence=_compute_confidence(results, sig))

    if sig.gate_definitively_failed:
        emit(SIG_GATE_BYPASSED, {"reason": "significance test found no effect"})
        return HypothesisVerdict(verdict="refuted", confidence=0.7)

    # Gate passed — check direction
    supports = _direction_supports_hypothesis(hypothesis, results)
    verdict = "confirmed" if supports else "refuted"
    emit(SIG_GATE_PASSED, {"verdict": verdict, "p_value": sig.p_value, "effect_size": sig.effect_size})
    return HypothesisVerdict(verdict=verdict, confidence=_compute_confidence(results, sig))
```

**Mandatory significance-capable tool per hypothesis:** The think-act loop's planning prompt explicitly instructs the agent to include at least one of: `statistical_significance`, `bootstrap_confidence`, `literature_baseline_compare`, or a custom statistical code block before attempting to stop. If the agent tries to stop without having run any significance-capable tool, it receives a correction prompt:

```
"You have not yet run any statistical significance test for this hypothesis.
Before stopping, you must either:
1. Call statistical_significance, bootstrap_confidence, or literature_baseline_compare
2. Write custom code that computes a p-value or effect size
Without this, the verdict will be inconclusive regardless of other evidence.
Choose your significance test now."
```

### 7.4 Domain Routing — Question-Aware

The domain router now receives both the hypothesis text **and the original research question** to prevent drift where optimizer-ranking questions end up routed to generic deep-learning tools.

```python
DOMAIN_ROUTING_PROMPT = """
Given this hypothesis and the original research question, return the single most relevant domain.

Original research question: {question}
Hypothesis: {hypothesis_text}

Domains:
- computational_biology: DNA/RNA/protein, genomics, CRISPR, cell biology
- chemistry: molecular simulation, reaction prediction, spectroscopy
- physics: mechanics, electromagnetism, quantum, thermodynamics
- mathematics: algebra, calculus, linear algebra, number theory, topology
- statistics: hypothesis testing, regression, Bayesian analysis, experimental design
- ml_research: statistical tests on ML results, significance testing, learning curves
- deep_learning: training models, architecture search, activation functions, optimizers
- algorithm_optimization: algorithm comparison, convergence, loss landscapes, benchmarking
- data_analysis: EDA, visualization, aggregation, cleaning
- general_computation: anything else

Important: if the research question involves COMPARING or RANKING approaches,
prefer statistics, ml_research, or algorithm_optimization over deep_learning,
even if the hypothesis mentions neural networks.

Return JSON only: {"domain": "domain_name", "reason": "one sentence"}
"""
```

### 7.5 Cross-Agent Memory During Parallel Execution

While agents run in parallel, the orchestrator broadcasts partial ledger updates to running agents. Agents receive peer findings as they arrive and can adjust their strategy.

```python
# Orchestrator: broadcast when any agent completes
async def _broadcast_peer_finding(result: ExperimentResult, running_agent_ids: list[str]):
    finding = PeerFinding(
        hypothesis_id=result.hypothesis_id,
        verdict=result.verdict,
        confidence=result.confidence,
        key_finding=result.key_finding,
        learned=result.learned,
    )
    for agent_id in running_agent_ids:
        await redis.publish(
            f"propab:agent_peer:{agent_id}",
            json.dumps(finding),
        )
    emit(MEMORY_LEDGER_BROADCAST, {"finding": finding, "broadcast_to": running_agent_ids})

# Sub-agent: consume peer broadcasts non-blocking
async def _poll_peer_findings(agent_id: str, redis) -> list[PeerFinding]:
    findings = []
    while True:
        msg = await redis.lpop(f"propab:agent_peer:{agent_id}")
        if msg is None:
            break
        findings.append(PeerFinding(**json.loads(msg)))
    return findings
```

The agent includes peer findings in its think prompt, so it can avoid duplicating work a peer already did and focus its remaining budget on unexplored angles.

### 7.6 Parallelism Model

```python
async def run_experiments_round(
    hypotheses: list[RankedHypothesis],
    round_id: str,
    budget: RoundBudget,
    ...
) -> RoundResult:

    tasks = [
        celery_app.send_task("run_sub_agent", args=[ExperimentTask(...)])
        for h in hypotheses
    ]

    ledger = RoundLedger(round_id=round_id)
    running_agent_ids = [t.id for t in tasks]
    deadline = time.monotonic() + budget.max_seconds_per_round

    for future in asyncio.as_completed(tasks):
        result = await future
        ledger.add(result)
        emit(SYNTH_RESULT_RECEIVED, {"result": result, "round_id": round_id})
        emit(SYNTH_LEDGER_UPDATED, {"ledger": ledger.summary(), "round_id": round_id})

        # Broadcast to remaining running agents
        still_running = [id for id in running_agent_ids if id not in ledger.completed_ids]
        await _broadcast_peer_finding(result, still_running)

        if time.monotonic() > deadline:
            emit(BUDGET_EXHAUSTED, {"reason": "round deadline"})
            break

    return ledger
```

---

## 8. Tool System

### 8.1 Tool Interface Contract

A tool is a plain Python function with a `TOOL_SPEC` dict.

```python
TOOL_SPEC = {
    "name":        "sequence_align",
    "domain":      "computational_biology",
    "description": "...",
    "params": {
        "sequences":  {"type": "list[str]", "required": True, "description": "..."},
        "algorithm":  {"type": "str", "required": False, "default": "smith_waterman"},
    },
    "output": {
        "alignment":        "list[str]",
        "identity_pct":     "float",
        "score":            "float",
    },
    "significance_capable": False,  # NEW: marks tools that produce p_value/effect_size
    "example": {...}
}
```

**`significance_capable` flag:** Tools that can produce p-values, effect sizes, or confidence intervals (e.g. `statistical_significance`, `bootstrap_confidence`, `literature_baseline_compare`) are flagged. The think-act loop enforcer checks whether at least one significance-capable tool was called before accepting a stop decision.

### 8.2 Tool Registry

```python
class ToolRegistry:
    def get_cluster(self, domain: str) -> list[ToolSpec]:
        """Returns specs only. Sub-agent uses specs for LLM context."""
        return [e.spec for e in self._registry.values() if e.domain == domain]

    def get_significance_tools(self) -> list[ToolSpec]:
        """Returns all significance-capable tools regardless of domain."""
        return [e.spec for e in self._registry.values() if e.spec.get("significance_capable")]

    def get_cluster_with_significance(self, domain: str) -> list[ToolSpec]:
        """Domain cluster + always includes significance tools."""
        cluster = {s["name"]: s for s in self.get_cluster(domain)}
        for sig_tool in self.get_significance_tools():
            cluster.setdefault(sig_tool["name"], sig_tool)
        return list(cluster.values())
```

### 8.3 Sandboxed Code Execution

```
Sub-agent → generate code → emit(CODE_GENERATED)
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │  SANDBOX CONTAINER  │
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
              emit(CODE_RESULT)          emit(CODE_ERROR, {retry: N})
```

Code must write results as JSON to stdout. Retry up to 3 times with error + previous code in context.

### 8.4 Per-Domain Sandbox Timeout Profiles

| Domain | Default timeout | Rationale |
|---|---|---|
| `general_computation` | 30s | Simple ops |
| `mathematics` | 60s | Symbolic computation |
| `statistics` | 60s | Bootstrap resampling |
| `data_analysis` | 60s | Large dataset processing |
| `ml_research` | 120s | Statistical tests, grid experiments |
| `algorithm_optimization` | 180s | Benchmarking across input sizes |
| `deep_learning` | 300s | Training loops |

---

## 9. Orchestrator — Multi-Round Research Loop

No LangGraph. Every state transition is explicit Python. Every decision is a logged event.

```python
async def research_loop(session_id: str, question: str) -> ResearchResult:

    ctx = ResearchContext(session_id=session_id, question=question)
    emit = partial(emit_event, session_id=session_id)

    emit(SESSION_STARTED, {"question": question})

    # ── Stage 1: Intake ──────────────────────────────────────────
    ctx.parsed_question = await intake(question, emit)

    # ── Stage 2: Literature ──────────────────────────────────────
    ctx.prior = await build_prior(ctx.parsed_question, emit)

    # ── Stage 3: Short-circuit check ────────────────────────────
    if answer := find_existing_answer(ctx.prior, ctx.parsed_question):
        emit(LIT_ANSWER_FOUND, {"answer": answer})
        return ResearchResult(type="existing_answer", answer=answer)

    # ── Stage 4: Multi-Round Research Loop ───────────────────────
    budget = ResearchBudget.from_config(settings)
    round_number = 0
    accumulated_ledger = AccumulatedLedger()

    while not budget.exhausted() and not _sufficient_evidence(accumulated_ledger, budget):

        round_id = str(uuid4())
        emit(ROUND_STARTED, {
            "round": round_number,
            "round_id": round_id,
            "budget_remaining": budget.summary(),
        })

        # Generate hypotheses for this round (informed by prior rounds)
        hypotheses = await generate_ranked_hypotheses(
            ctx.prior,
            ctx.parsed_question,
            prior_round_findings=accumulated_ledger.summary(),
            round_number=round_number,
            emit=emit,
        )
        # Deduplicate against prior rounds
        hypotheses = deduplicate_hypotheses(hypotheses, accumulated_ledger.all_hypothesis_embeddings)

        # Dispatch and run experiments
        round_result = await run_experiments_round(
            hypotheses=hypotheses,
            round_id=round_id,
            budget=budget.round_budget(round_number),
            prior=ctx.prior,
            emit=emit,
        )

        accumulated_ledger.merge(round_result)
        budget.record_round(round_result)

        emit(ROUND_COMPLETED, {
            "round": round_number,
            "round_id": round_id,
            "confirmed": len(round_result.confirmed),
            "refuted": len(round_result.refuted),
            "inconclusive": len(round_result.inconclusive),
            "budget_remaining": budget.summary(),
        })

        # Detect diminishing returns
        if budget.rounds_completed >= 2:
            returns = accumulated_ledger.marginal_return(round_number)
            if returns < budget.min_marginal_return:
                emit(PROGRESS_DIMINISHING, {
                    "round": round_number,
                    "marginal_return": returns,
                    "threshold": budget.min_marginal_return,
                })
                break

        # Warn if all hypotheses inconclusive again
        if round_result.all_inconclusive():
            emit(SYNTH_ALL_INCONCLUSIVE, {
                "round": round_number,
                "note": "All hypotheses inconclusive despite experiment execution. Check significance gate logs.",
            })

        # Checkpoint for resumability
        await checkpoint_session(ctx, accumulated_ledger, round_number, budget)

        round_number += 1

    # ── Stage 5: Paper ────────────────────────────────────────────
    paper_url = await write_paper(ctx, accumulated_ledger, emit)
    emit(PAPER_READY, {"url": paper_url})
    emit(SESSION_COMPLETED, {
        "paper_url": paper_url,
        "total_rounds": round_number,
        "confirmed": accumulated_ledger.total_confirmed,
        "refuted": accumulated_ledger.total_refuted,
        "inconclusive": accumulated_ledger.total_inconclusive,
    })

    return ResearchResult(type="paper", url=paper_url, ledger=accumulated_ledger)
```

### 9.1 Stopping Criteria

The loop terminates when **any** of these conditions is met:

| Criterion | Check | Configurable |
|---|---|---|
| Hard time budget | `time.monotonic() >= budget.deadline` | `RESEARCH_MAX_HOURS` (default: 1h) |
| Hard round budget | `round_number >= budget.max_rounds` | `RESEARCH_MAX_ROUNDS` (default: 5) |
| Confirmed findings threshold | `accumulated_ledger.total_confirmed >= budget.target_confirmed` | `RESEARCH_TARGET_CONFIRMED` (default: 3) |
| Diminishing returns | `marginal_return(round) < min_marginal_return` for 2 consecutive rounds | `RESEARCH_MIN_MARGINAL_RETURN` (default: 0.05) |
| Budget exhausted | Token/API cost budget hit | `RESEARCH_MAX_COST_USD` (default: none) |

### 9.2 Session Checkpointing (Resumability)

Long-running sessions checkpoint after every round. A session can be resumed if interrupted.

```python
async def checkpoint_session(
    ctx: ResearchContext,
    ledger: AccumulatedLedger,
    round_number: int,
    budget: ResearchBudget,
) -> None:
    """Writes a snapshot to Postgres that can be used to resume from this point."""
    async with session_factory() as session:
        await session.execute(
            text("""
                INSERT INTO session_checkpoints
                    (id, session_id, round_number, ledger_json, budget_json, created_at)
                VALUES
                    (:id, :session_id, :round_number, CAST(:ledger_json AS jsonb),
                     CAST(:budget_json AS jsonb), NOW())
                ON CONFLICT (session_id, round_number) DO UPDATE
                SET ledger_json = EXCLUDED.ledger_json, budget_json = EXCLUDED.budget_json
            """),
            {
                "id": str(uuid4()),
                "session_id": ctx.session_id,
                "round_number": round_number,
                "ledger_json": json.dumps(ledger.to_dict()),
                "budget_json": json.dumps(budget.to_dict()),
            }
        )

async def resume_session(session_id: str) -> ResearchContext | None:
    """Load last checkpoint and resume from the next round."""
    checkpoint = await load_last_checkpoint(session_id)
    if checkpoint is None:
        return None
    emit(SESSION_RESUMED, {"session_id": session_id, "from_round": checkpoint.round_number})
    return ResearchContext.from_checkpoint(checkpoint)
```

---

## 10. Research Memory System

Memory is what turns a one-shot pipeline into a long-running agent. Three layers:

### 10.1 Session Memory (Current Run)

Everything the system has done and learned in the current session. Stored in Postgres and queryable at any time.

```python
class SessionMemory:
    """Read/write interface to the current session's accumulated knowledge."""

    def get_confirmed_findings(self) -> list[Finding]:
        """All hypotheses confirmed in any round of this session."""

    def get_dead_ends(self) -> list[DeadEnd]:
        """All refuted hypotheses + their evidence, so later rounds don't repeat them."""

    def get_tried_tools(self, domain: str) -> set[str]:
        """Tools already called in this session, to guide diversity in later rounds."""

    def get_partial_findings(self) -> list[PartialFinding]:
        """Inconclusive hypotheses with confidence > 0.25, candidates for promotion."""

    def get_round_summaries(self) -> list[RoundSummary]:
        """Per-round: what was tried, what was found, what the marginal return was."""

    def summarize_for_hypothesis_generator(self) -> str:
        """Compact text summary suitable for inclusion in hypothesis generation prompt."""
```

### 10.2 Agent Working Memory (Per Sub-Agent)

Each sub-agent carries its own context window during execution. The think-act loop manages this explicitly — it does not rely on the LLM's context window to persist information.

```python
@dataclass
class AgentContext:
    hypothesis: RankedHypothesis
    prior: Prior
    peer_findings: list[PeerFinding]       # live peer results
    learned_from: str | None               # prior-round learning for this hypothesis
    results_so_far: list[StepResult]       # accumulated tool/code outputs THIS run
    steps_taken: int
    significance_status: SignificanceResult

    def to_prompt_summary(self, max_tokens: int = 2000) -> str:
        """
        Compact, token-bounded summary for inclusion in think prompts.
        Prioritizes: significance evidence > numeric findings > tool names run.
        Truncates old/redundant results when budget is tight.
        """
```

**Context window management:** As the agent runs more steps, older results are summarized or truncated. The most recent result and any significance evidence are always included verbatim. The agent never silently loses important context — the summarizer preserves the key numeric evidence.

### 10.3 Cross-Session Knowledge Base (Persistent)

Findings from completed sessions persist in a shared knowledge base. Future sessions on the same or related questions start with this prior work available.

```python
class CrossSessionKB:
    """
    Shared knowledge base that accumulates findings across all research sessions.
    Keyed by topic embedding — supports fuzzy lookup by semantic similarity.
    """

    async def write_finding(self, finding: Finding, session_id: str) -> None:
        """Write a confirmed finding to the KB with provenance."""
        embed = await embed(finding.text)
        await qdrant.upsert(
            collection="knowledge_base",
            points=[{
                "id": str(uuid4()),
                "vector": embed,
                "payload": {
                    "text": finding.text,
                    "evidence": finding.evidence,
                    "session_id": session_id,
                    "confidence": finding.confidence,
                    "created_at": datetime.utcnow().isoformat(),
                }
            }]
        )
        emit(MEMORY_KB_WRITTEN, {"finding": finding.text, "session_id": session_id})

    async def query(self, question: str, top_k: int = 10) -> list[KBFinding]:
        """Retrieve findings relevant to a question by semantic similarity."""
        embed = await embed(question)
        results = await qdrant.search(collection="knowledge_base", vector=embed, limit=top_k)
        return [KBFinding(**r.payload) for r in results]
```

**Integration with prior builder:** Before generating hypotheses, the prior builder queries the KB for findings from prior sessions. Confirmed findings from previous runs on the same topic are added to `established_facts` in the prior, so the new session starts further ahead.

---

## 11. Budget & Progress Management

### 11.1 Research Budget

```python
@dataclass
class ResearchBudget:
    # Hard limits
    max_rounds:            int    = 5
    max_hours:             float  = 1.0    # wall clock hours
    max_hypotheses_total:  int    = 50     # across all rounds
    max_cost_usd:          float  = 0.0   # 0 = no cost limit
    target_confirmed:      int    = 3      # stop if this many confirmed

    # Soft limits
    min_marginal_return:   float  = 0.05  # marginal progress threshold
    max_stale_rounds:      int    = 2     # stop after N rounds with no new confirmed

    # Per-agent
    agent_max_steps:       int    = 15    # max think-act steps per hypothesis
    agent_min_steps:       int    = 5     # minimum before allowing early stop
    agent_max_seconds:     int    = 300   # per-agent wall clock cap

    # Per-round
    max_hypotheses_per_round: int = 5
    max_seconds_per_round:    int = 600

    deadline: float = field(default_factory=lambda: time.monotonic() + 3600)

    def exhausted(self) -> bool:
        return (
            time.monotonic() >= self.deadline
            or self.rounds_completed >= self.max_rounds
            or self.hypotheses_tested >= self.max_hypotheses_total
        )

    def round_budget(self, round_number: int) -> RoundBudget:
        """Allocate per-round budget. Later rounds may get less if overall budget is tight."""
        remaining_seconds = max(0.0, self.deadline - time.monotonic())
        rounds_remaining = self.max_rounds - round_number
        return RoundBudget(
            max_seconds=min(self.max_seconds_per_round, remaining_seconds / max(1, rounds_remaining)),
            max_hypotheses=self.max_hypotheses_per_round,
            agent_budget=AgentBudget(
                max_steps=self.agent_max_steps,
                min_steps=self.agent_min_steps,
                deadline=time.monotonic() + min(self.agent_max_seconds, remaining_seconds / 2),
            )
        )
```

Environment variable overrides:

| Variable | Default | Purpose |
|---|---|---|
| `RESEARCH_MAX_ROUNDS` | `5` | Hard round cap |
| `RESEARCH_MAX_HOURS` | `1.0` | Wall-clock limit for the full session |
| `RESEARCH_TARGET_CONFIRMED` | `3` | Stop when this many hypotheses confirmed |
| `RESEARCH_MAX_HYPOTHESES` | `50` | Total hypothesis budget across rounds |
| `RESEARCH_MIN_MARGINAL_RETURN` | `0.05` | Diminishing-returns threshold |
| `AGENT_MAX_STEPS` | `15` | Per-agent think-act step cap |
| `AGENT_MIN_STEPS` | `5` | Minimum steps before agent may stop early |

For a **full-day run**, set `RESEARCH_MAX_HOURS=24`, `RESEARCH_MAX_ROUNDS=50`, `RESEARCH_TARGET_CONFIRMED=10`.

### 11.2 Progress Measurement

Marginal return is computed after each round to detect diminishing returns:

```python
def marginal_return(self, round_number: int) -> float:
    """
    Returns a 0–1 score measuring new scientific progress made in this round.
    Combines: new confirmed findings + hypothesis confidence improvement + new tools explored.
    """
    if round_number < 1:
        return 1.0
    prev = self.round_summaries[round_number - 1]
    curr = self.round_summaries[round_number]

    delta_confirmed = (len(curr.confirmed) - len(prev.confirmed)) / max(1, len(curr.confirmed) + 1)
    delta_confidence = _mean_confidence_delta(prev.inconclusive, curr.inconclusive)
    delta_coverage = _new_tool_coverage(prev.tools_used, curr.tools_used)

    return 0.5 * delta_confirmed + 0.3 * delta_confidence + 0.2 * delta_coverage
```

---

## 12. Evidence Accumulation & Significance Gate

### 12.1 Accumulated Evidence Across Rounds

Evidence is not just per-hypothesis per-round. It accumulates. If hypothesis H2 was promoted to H2-refined in round 2, the evidence from H2 round 1 carries forward into H2-refined round 2.

```python
class AccumulatedEvidence:
    """
    Cross-round evidence for a hypothesis lineage (original + all refinements).
    Statistical power grows as more data points are added.
    """
    lineage: list[str]                     # hypothesis_id chain
    all_metric_values: list[float]
    all_p_values: list[float]
    all_effect_sizes: list[float]
    combined_n: int                        # total observations across rounds

    def combined_significance(self) -> SignificanceResult:
        """Fisher's method for combining p-values across rounds."""
        if not self.all_p_values:
            return SignificanceResult(gate_passed=False, ...)
        chi2 = -2 * sum(math.log(p) for p in self.all_p_values if p > 0)
        df = 2 * len(self.all_p_values)
        combined_p = 1 - chi2_cdf(chi2, df)
        return SignificanceResult(
            gate_passed=combined_p < 0.05,
            p_value=combined_p,
            method="fisher_combined",
            ...
        )
```

### 12.2 Evidence Contract (Enforced)

This is the complete rule set enforced in `compute_verdict()`:

| Condition | Verdict | Confidence range |
|---|---|---|
| No metric-bearing steps | inconclusive | 0.0 |
| Metric steps ran but no baseline | inconclusive | 0.15–0.30 |
| Baseline present, no significance evidence | inconclusive | 0.25–0.40 |
| Significance gate passed, direction supports | confirmed | 0.50–0.95 |
| Significance gate definitively failed | refuted | 0.60–0.80 |
| Significance gate passed, direction opposes | refuted | 0.60–0.85 |

No LLM override of this contract. The verdict is computed from evidence fields, not inferred by language model.

---

## 13. Paper Writing Layer

### 13.1 Core Constraint

The methods section is **not written by an LLM**. It is compiled deterministically from `experiment_steps` rows in order, across all rounds. Every claim in the paper can be traced to a specific database row.

### 13.2 Multi-Round Paper Pipeline

The paper writer now operates over the full accumulated ledger from all rounds, not just one pass.

```
INPUTS:
  ├── research_sessions row + all session_checkpoint rows
  ├── all hypotheses rows across all rounds (with round_id, refinement_of)
  ├── all experiment_steps rows (ordered by round then hypothesis then step)
  ├── all tool_calls rows
  ├── all llm_calls rows
  └── MinIO artifacts (figures, output files)

PIPELINE:
  1. multi_round_trace_compiler()
     Group steps by round → hypothesis → step
     Build hypothesis lineage graph (original + refinements)
     Identify the "evolution story" of each finding across rounds
     → MultiRoundTraceReport

  2. section_generator()
     abstract      ← strongest confirmed findings across all rounds
     introduction  ← prior + motivation
     methods       ← DETERMINISTIC: per-round experiment summaries from trace
     results       ← per-round finding summaries + significance evidence
     discussion    ← cross-round synthesis: how did the investigation evolve?
     conclusion    ← confirmed findings + open questions
     references    ← papers from prior + any fetched during experiments

  3. figure_embedding()
  4. latex_render()
  5. emit(PAPER_READY)
```

---

## 14. Data Layer

### 14.1 Storage Components

| Store | Purpose | What lives here |
|---|---|---|
| **Postgres** | Source of truth | Sessions, hypotheses, experiment steps, tool calls, LLM calls, paper cache, citation graph, session checkpoints, research rounds |
| **Qdrant** | Vector search | Paper chunk embeddings, cross-session knowledge base embeddings |
| **Redis** | Task queue + event bus + agent peer channels | Celery queue, SSE pub/sub, agent-peer broadcast channels, query cache |
| **MinIO** | Binary artifacts | PDFs, figures, compiled LaTeX, experiment outputs |
| **SQLite** | BM25 index | Per-session inverted index over paper text |

### 14.2 Postgres Schema

```sql
-- Research sessions
CREATE TABLE research_sessions (
    id              UUID PRIMARY KEY,
    question        TEXT NOT NULL,
    status          TEXT NOT NULL,  -- pending | running | completed | failed
    stage           TEXT,
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
    status          TEXT
);

CREATE TABLE paper_citations (
    source_paper_id TEXT REFERENCES papers(id),
    cited_paper_id  TEXT,
    PRIMARY KEY (source_paper_id, cited_paper_id)
);

-- Research rounds (new — multi-round loop)
CREATE TABLE research_rounds (
    id              UUID PRIMARY KEY,
    session_id      UUID REFERENCES research_sessions(id),
    round_number    INT NOT NULL,
    status          TEXT NOT NULL,  -- running | completed | failed
    confirmed_count INT DEFAULT 0,
    refuted_count   INT DEFAULT 0,
    inconclusive_count INT DEFAULT 0,
    marginal_return FLOAT,
    budget_json     JSONB,          -- budget snapshot at start of round
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    UNIQUE (session_id, round_number)
);

-- Session checkpoints (new — resumability)
CREATE TABLE session_checkpoints (
    id              UUID PRIMARY KEY,
    session_id      UUID REFERENCES research_sessions(id),
    round_number    INT NOT NULL,
    ledger_json     JSONB NOT NULL,   -- full accumulated ledger at this point
    budget_json     JSONB NOT NULL,   -- budget state at this point
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (session_id, round_number)
);

-- Hypotheses
CREATE TABLE hypotheses (
    id              UUID PRIMARY KEY,
    session_id      UUID REFERENCES research_sessions(id),
    round_id        UUID REFERENCES research_rounds(id),  -- which round
    text            TEXT NOT NULL,
    test_methodology TEXT,
    scores_json     JSONB,
    rank            INT,
    status          TEXT,   -- pending | running | completed | failed | promoted | retired
    verdict         TEXT,   -- confirmed | refuted | inconclusive
    confidence      FLOAT,
    evidence_summary TEXT,
    key_finding     TEXT,
    tool_trace_id   TEXT,
    refinement_of   UUID REFERENCES hypotheses(id),   -- prior-round parent if promoted
    learned_from    TEXT,                              -- summary from prior round
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Full experiment trace
CREATE TABLE experiment_steps (
    id              UUID PRIMARY KEY,
    hypothesis_id   UUID REFERENCES hypotheses(id),
    step_type       TEXT NOT NULL,  -- tool_call | code_exec | llm_reasoning | think_decision
    step_index      INT,
    input_json      JSONB,
    output_json     JSONB,
    error_json      JSONB,
    duration_ms     INT,
    memory_mb       INT,
    timeout_sec     INT,
    summary         TEXT,
    significance_json JSONB,   -- new: p_value/effect_size/CI if this step produced them
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Tool calls (denormalized for convenience)
CREATE TABLE tool_calls (
    id              UUID PRIMARY KEY,
    step_id         UUID REFERENCES experiment_steps(id),
    hypothesis_id   UUID REFERENCES hypotheses(id),
    tool_name       TEXT NOT NULL,
    domain          TEXT NOT NULL,
    params_json     JSONB,
    result_json     JSONB,
    success         BOOLEAN,
    significance_capable BOOLEAN DEFAULT FALSE,  -- new
    produced_p_value FLOAT,                       -- new: extracted from result
    produced_effect_size FLOAT,                   -- new
    duration_ms     INT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Every LLM call stored verbatim
CREATE TABLE llm_calls (
    id              UUID PRIMARY KEY,
    session_id      UUID REFERENCES research_sessions(id),
    hypothesis_id   UUID,
    call_purpose    TEXT,   -- "hypothesis_generation" | "agent.decide_next_step" | etc.
    model           TEXT,
    prompt_text     TEXT NOT NULL,
    response_text   TEXT NOT NULL,
    input_tokens    INT,
    output_tokens   INT,
    duration_ms     INT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Cross-session knowledge base entries
CREATE TABLE kb_findings (
    id              UUID PRIMARY KEY,
    session_id      UUID REFERENCES research_sessions(id),
    hypothesis_id   UUID REFERENCES hypotheses(id),
    text            TEXT NOT NULL,
    evidence        TEXT,
    confidence      FLOAT,
    embedding_id    TEXT,   -- Qdrant point ID in "knowledge_base" collection
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Session budget tracking
CREATE TABLE session_budgets (
    id              UUID PRIMARY KEY,
    session_id      UUID REFERENCES research_sessions(id),
    config_json     JSONB NOT NULL,     -- initial budget config
    consumed_json   JSONB DEFAULT '{}', -- running totals: rounds, hypotheses, cost
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Agent memory state (per-agent working memory snapshots)
CREATE TABLE agent_memory (
    id              UUID PRIMARY KEY,
    hypothesis_id   UUID REFERENCES hypotheses(id),
    step_index      INT,
    significance_json   JSONB,   -- significance status at this step
    peer_findings_json  JSONB,   -- peer results received so far
    results_summary TEXT,        -- compact summary for context management
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Event log
CREATE TABLE events (
    id              UUID PRIMARY KEY,
    session_id      UUID REFERENCES research_sessions(id),
    round_id        UUID REFERENCES research_rounds(id),  -- new
    event_type      TEXT NOT NULL,
    source          TEXT,
    step            TEXT,
    hypothesis_id   UUID,
    parent_event_id UUID,
    payload_json    JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX events_session_idx ON events(session_id, created_at);
CREATE INDEX events_round_idx   ON events(round_id, created_at);
```

---

## 15. API & Streaming Layer

### 15.1 Endpoints

```
POST   /research                         Submit a research question → returns session_id
GET    /stream/{session_id}              SSE stream of all events
GET    /sessions/{session_id}            Full session state
GET    /sessions/{session_id}/prior      Prior JSON
GET    /sessions/{session_id}/rounds     All research rounds with summaries
GET    /sessions/{session_id}/hypotheses All hypotheses (filterable by round, verdict)
GET    /sessions/{session_id}/trace      Full experiment trace (all rounds, all steps)
GET    /sessions/{session_id}/paper      Paper PDF + .tex download URLs
GET    /sessions/{session_id}/memory     Session memory summary (confirmed findings, dead ends)
GET    /sessions/{session_id}/budget     Current budget status and consumption
GET    /sessions/{session_id}/llm-calls  All LLM prompts + responses
POST   /sessions/{session_id}/resume     Resume an interrupted long-running session
GET    /knowledge-base                   Query the cross-session KB
GET    /tools                            List all tools with specs
GET    /tools/{domain}                   Tools for a domain
GET    /tools/significance               All significance-capable tools
GET    /health                           Liveness check
```

### 15.2 POST /research

```json
{
  "question": "...",
  "config": {
    "max_rounds":           5,
    "max_hours":            1.0,
    "target_confirmed":     3,
    "max_hypotheses":       5,
    "agent_max_steps":      15,
    "paper_ttl_days":       30,
    "llm_model":            "gpt-4o",
    "plan_source":          "hybrid"
  }
}
```

---

## 16. Infrastructure & Setup

### 16.1 docker-compose.yml

Services: `frontend`, `api`, `orchestrator`, `worker`, `postgres`, `redis`, `qdrant`, `minio`.

Workers are scaled with `PROPAB_WORKERS` (default: 3). For long-running sessions, increase workers and Celery concurrency.

### 16.2 Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | LLM + embedding calls |
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `ollama` \| `gemini` |
| `LLM_MODEL` | `gpt-4o` | Model for orchestrator + agents |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder |
| `PROPAB_WORKERS` | `3` | Parallel experiment workers |
| `PROPAB_MAX_HYPOTHESES` | `5` | Hypotheses per round |
| `PAPER_TTL_DAYS` | `30` | Paper cache TTL |
| `SANDBOX_TIMEOUT_SEC` | `30` | Global sandbox timeout default |
| `SUB_AGENT_PLAN_SOURCE` | `hybrid` | `heuristic` \| `llm` \| `hybrid` |
| `SUB_AGENT_MAX_ROUNDS` | `3` | Heuristic plan rounds |
| `SUB_AGENT_TOOLS_PER_ROUND` | `3` | Tools selected per heuristic round |
| `SUB_AGENT_MAX_PLANNED_STEPS` | `8` | LLM plan max steps |
| `RESEARCH_MAX_ROUNDS` | `5` | Multi-round loop hard cap |
| `RESEARCH_MAX_HOURS` | `1.0` | Wall-clock budget hours |
| `RESEARCH_TARGET_CONFIRMED` | `3` | Confirmed findings to aim for |
| `RESEARCH_MAX_HYPOTHESES` | `50` | Total hypothesis budget |
| `RESEARCH_MIN_MARGINAL_RETURN` | `0.05` | Diminishing-returns threshold |
| `AGENT_MAX_STEPS` | `15` | Think-act steps per agent |
| `AGENT_MIN_STEPS` | `5` | Min steps before early stop allowed |

### 16.3 Setup

```bash
git clone https://github.com/propab/propab
cd propab
cp .env.example .env          # add your OPENAI_API_KEY
docker compose up
# → frontend: http://localhost:3000
# → api:      http://localhost:8000/docs
```

---

## 17. Failure Handling

| Failure | Detection | Recovery | Event |
|---|---|---|---|
| Literature fetch returns 0 papers | Empty results | Broaden query, retry once | `SESSION_FAILED` |
| Prior builder malformed JSON | JSON parse fails | Retry with schema-injected prompt (max 2x) | `LLM_PARSE_ERROR` |
| Think decision LLM parse failure | JSON parse fails | Fall back to heuristic next-step selection | `LLM_PARSE_ERROR` |
| Sub-agent exceeds step budget | steps >= agent_max_steps | Evaluate with accumulated evidence, mark incomplete | `AGENT_COMPLETED` |
| Sub-agent crashes | Celery exception | Hypothesis inconclusive, full traceback stored | `AGENT_FAILED` |
| Significance gate not passable | Max steps reached, no sig tool ran | Mandatory correction prompt or inconclusive | `SIG_GATE_FAILED` |
| All sub-agents inconclusive | Round ledger check | Emit warning, still continue to next round | `SYNTH_ALL_INCONCLUSIVE` |
| Round deadline exceeded | time.monotonic() check | Cancel remaining tasks, proceed with partial results | `BUDGET_EXHAUSTED` |
| Session interrupted | Process killed / infra failure | Checkpoint exists → resume from last round | `SESSION_RESUMED` |
| LaTeX compilation fails | pdflatex exit code != 0 | Return .tex with error attached | `PAPER_LATEX_COMPILED {success: false}` |
| LLM rate limit | HTTP 429 | Exponential backoff: 2s, 4s, 8s, 16s | `LLM_PROMPT` retry |

**Failure Principle:** A session never silently fails. If interrupted, it checkpoints. If a tool fails, the agent continues. If a round fails, the orchestrator proceeds. A partial result with honest failure documentation is more valuable than an aborted run.

---

## 18. Build Roadmap

### Phase 1 — Foundation (Complete)
- [x] Monorepo structure, Docker Compose, all services stubbed
- [x] Postgres schema + migrations (Alembic)
- [x] Event system: `emit()` → Redis → Postgres
- [x] SSE endpoint
- [x] arxiv API client + PDF parser
- [x] Hybrid retrieval: dense + sparse + citation graph + reranker
- [x] Prior builder
- [x] Hypothesis generator + ranker
- [x] Sub-agent harness + Celery
- [x] Domain routing
- [x] Tool registry + 40+ STEM tools
- [x] Sandbox executor with resource caps
- [x] Paper writer (trace compiler + section generator + LaTeX)
- [x] Evidence contract: metric-based verdict, confidence computation
- [x] Crash hardening: Q1/Q4 failure path resolved

### Phase 2 — Iterative Agent Core (Active)
- [ ] **Think-act sub-agent loop**: LLM decides next step after observing each result
- [ ] **Significance gate**: enforced per-hypothesis, mandatory before verdict
- [ ] **Question-aware domain routing**: research question injected alongside hypothesis
- [ ] **Significance-capable tool injection**: always include stat tools in domain cluster
- [ ] **Hypothesis quality gate**: reject generic fallback text, retry
- [ ] **Promoted hypotheses**: inconclusive with signal → refined version re-queued
- [ ] **Agent working memory**: `AgentContext.to_prompt_summary()` with token budget

### Phase 3 — Multi-Round Orchestrator (Next)
- [ ] `research_rounds` table + migrations
- [ ] `session_checkpoints` table + resumability
- [ ] Multi-round `research_loop()` with budget control
- [ ] `AccumulatedLedger` across rounds
- [ ] Diminishing-returns detection
- [ ] Round-aware hypothesis generator (prior round findings in prompt)
- [ ] Hypothesis deduplication across rounds
- [ ] `SYNTH_ALL_INCONCLUSIVE` warning event + handling
- [ ] Budget management API: `GET /sessions/{id}/budget`

### Phase 4 — Memory & Cross-Agent (Following)
- [ ] Cross-agent peer finding broadcast (Redis channels per agent)
- [ ] Agent think prompts include peer findings
- [ ] `CrossSessionKB`: Qdrant collection + write/query API
- [ ] KB query integrated into prior builder
- [ ] `agent_memory` table: per-step significance snapshots
- [ ] `kb_findings` table + `GET /knowledge-base` endpoint
- [ ] Cross-round evidence accumulation (Fisher's method for combined p-values)

### Phase 5 — Long-Running Mode & Scale (Future)
- [ ] Full-day run mode: `RESEARCH_MAX_HOURS=24`, `RESEARCH_MAX_ROUNDS=50`
- [ ] Thousands-of-hypotheses management: priority queue, aggressive deduplication
- [ ] Dataset connectors (UCI ML, NCBI, HuggingFace datasets)
- [ ] Claim grounding / hallucination guard (each paper claim traced to DB row)
- [ ] Ollama / local model support (chat adapter already written)
- [ ] Community tool plugin submission process

---

## 19. Repository Structure

```
propab/
├── docker-compose.yml
├── .env.example
├── ARCHITECTURE.md               ← this document
│
├── services/
│   ├── api/                      # FastAPI gateway
│   │   └── app/routes/
│   │       ├── research.py
│   │       ├── stream.py
│   │       ├── sessions.py
│   │       └── knowledge_base.py    # new: KB query endpoint
│   │
│   ├── orchestrator/             # Multi-round research loop
│   │   ├── research_loop.py      # multi-round explicit state machine
│   │   ├── round_loop.py         # new: single-round logic extracted
│   │   ├── accumulated_ledger.py # new: cross-round evidence tracking
│   │   ├── budget.py             # new: budget + progress management
│   │   ├── hypothesis_lifecycle.py # new: promote/retire logic
│   │   ├── intake.py
│   │   ├── literature.py
│   │   ├── hypotheses.py
│   │   ├── paper.py
│   │   └── ...
│   │
│   └── worker/                   # Celery think-act agents
│       ├── sub_agent_loop.py     # updated: think-act loop
│       ├── think_act.py          # new: decide_next_action, AgentContext
│       ├── significance.py       # new: SignificanceResult, check_significance
│       ├── agent_memory.py       # new: AgentContext.to_prompt_summary()
│       ├── peer_findings.py      # new: poll_peer_findings, broadcast
│       ├── domain_router.py      # updated: question-aware routing
│       └── ...
│
├── packages/
│   └── propab-core/propab/
│       ├── events.py             # updated: new event types
│       ├── types.py              # updated: Round, Budget, SignificanceResult types
│       ├── memory.py             # new: CrossSessionKB
│       ├── budget.py             # new: ResearchBudget, AgentBudget
│       └── tools/
│           ├── registry.py       # updated: get_cluster_with_significance()
│           └── ...
│
├── migrations/
│   └── versions/
│       ├── 001_initial.sql
│       ├── 002_research_rounds.sql    # new
│       ├── 003_session_checkpoints.sql # new
│       ├── 004_agent_memory.sql        # new
│       └── 005_kb_findings.sql         # new
│
└── frontend/
    └── src/pages/
        ├── Session.tsx           # updated: round progress, budget display
        ├── Results.tsx           # updated: multi-round results
        └── KnowledgeBase.tsx     # new: KB explorer
```

---

*This document is the authoritative architecture reference for Propab. Implementation decisions that diverge from this spec must be recorded here with rationale before the divergence is merged.*
