"""
Real LitQA2 evaluation — not a proxy.

``astabench.py``'s ``run_litqa2_proxy`` scores hand-built cases against
whatever this service already has indexed, using embedding similarity alone
(no LLM reasoning at all). This module is the honest upgrade: it loads
actual questions from ``futurehouse/lab-bench`` (LAB-Bench's public LitQA2
subset — the exact same dataset AstaBench's official harness wraps in
``asta-bench/astabench/evals/labbench/litqa2/task.py``, and it is *not*
gated), retrieves evidence live through this service's own `/prior` pipeline,
and then uses an LLM to answer from that evidence — matching how every real
leaderboard entry works (retrieve, then reason), not just nearest-neighbor
matching.

Scoring matches LAB-bench exactly: accuracy (fraction correct), coverage
(fraction of answers that aren't "Insufficient information"), precision
(fraction correct among the answers that weren't "Insufficient information").
See ``asta-bench/astabench/evals/labbench/litqa2/task.py`` for the reference
implementation this mirrors.
"""
from __future__ import annotations

import asyncio
import random
import re
import string
from typing import Any

import httpx

from services.literature.app.llm_client import gemini_generate
from services.literature.app.retriever.chunk_rag import rerank_chunks, retrieve_relevant_chunks

UNSURE_ANSWER = "Insufficient information to answer the question"
_LITQA2_PARQUET_URL = (
    "https://huggingface.co/datasets/futurehouse/lab-bench/resolve/main/LitQA2/train-00000-of-00001.parquet"
)


async def load_litqa2_sample(
    *, n: int, seed: int = 0, only_opensource: bool = True, http_timeout: float = 60.0
) -> list[dict[str, Any]]:
    """Download the real LitQA2 question set and return a reproducible random
    sample. Downloaded at call time rather than vendored into the repo — it's
    a third-party dataset (CC-BY-SA-4.0), not a fixture we own."""
    import io

    import pandas as pd

    async with httpx.AsyncClient(timeout=http_timeout, follow_redirects=True) as client:
        resp = await client.get(_LITQA2_PARQUET_URL)
        resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))
    if only_opensource:
        df = df[df["is_opensource"]]
    rng = random.Random(seed)
    indices = list(df.index)
    rng.shuffle(indices)
    sample = df.loc[indices[:n]]
    return [
        {
            "id": row["id"],
            "question": row["question"],
            "ideal": row["ideal"],
            "distractors": list(row["distractors"]),
            "sources": list(row["sources"]) if row["sources"] is not None else [],
            "key_passage": row.get("key-passage"),
        }
        for _, row in sample.iterrows()
    ]


def build_choices(ideal: str, distractors: list[str], rng: random.Random) -> tuple[list[str], int, int]:
    """Mirrors ``LabbenchQuestion.full_choices()`` in asta-bench exactly:
    ideal + distractors + an injected "unsure" option, shuffled, tracking
    both the correct-answer index and the unsure-option index post-shuffle."""
    choices = [ideal] + list(distractors) + [UNSURE_ANSWER]
    unsure_idx = len(choices) - 1
    perm = list(range(len(choices)))
    rng.shuffle(perm)
    shuffled = [choices[i] for i in perm]
    return shuffled, perm.index(0), perm.index(unsure_idx)


def format_choices(choices: list[str]) -> str:
    letters = string.ascii_uppercase
    return "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))


def build_prompt(question: str, choices: list[str], evidence: list[str]) -> str:
    # Calibration is the dominant accuracy lever here, and it points one way:
    # abstain as little as possible. The scored accuracy metric is
    # correct/total, and "Insufficient information" scores *zero* — a
    # guaranteed miss — whereas committing to a best-reasoned answer is right
    # some fraction of the time. Measured live (n=50, gemini-3.1-pro-preview,
    # CHANGELOG.md 0.6.0): the model abstained on 56% of questions at
    # precision 0.77, so overall accuracy was only 0.34 — the abstentions,
    # not wrong answers, were the ceiling. LAB-bench's own reference prompt
    # (Future-House/LAB-Bench/labbench/zero_shot.py) likewise instructs the
    # model to "always answer... even if you are unsure." So: reason hard
    # from evidence + domain knowledge, and commit to the single most
    # plausible real option; reserve "Insufficient information" for the rare
    # case where the options are genuinely indistinguishable given everything
    # known. An educated guess beats a guaranteed zero.
    evidence_block = (
        "\n\n".join(f"[{i + 1}] {e}" for i, e in enumerate(evidence)) if evidence else "(no evidence retrieved)"
    )
    return (
        "You are answering a multiple-choice question about scientific literature. "
        "Reason carefully from the evidence excerpts below AND your own scientific "
        "knowledge, exactly as an expert would if reading these papers.\n\n"
        "Answering policy — this is critical:\n"
        "- You MUST select one of the SUBSTANTIVE answer options. Do NOT select "
        "\"Insufficient information\" — treat it as not being a valid choice.\n"
        "- Every question has a correct answer among the substantive options. When the "
        "evidence is partial, indirect, or absent, use your scientific/domain knowledge, "
        "the plausibility of each option, and careful reasoning to identify the single "
        "MOST LIKELY answer. An educated best guess is always required and always beats "
        "abstaining (which scores zero).\n"
        "- Eliminate options that are implausible or contradicted, then choose the best of "
        "what remains — even a 55/45 lean is a decision worth committing to.\n\n"
        f"Evidence excerpts:\n{evidence_block}\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{format_choices(choices)}\n\n"
        "Think step by step, then respond with your final choice (a substantive option, "
        'never "Insufficient information") as exactly one JSON object on its own line at '
        'the end: {"answer": "<letter>"}'
    )


_QUERY_STOPWORDS_RE = re.compile(r"^(what|which|how|why|when|where|who|does|do|is|are|by|the|a|an|of|on|for|in)$", re.I)


async def reformulate_query_for_search(
    question: str, *, api_key: str, model: str, choices: list[str] | None = None, http_timeout: float = 25.0
) -> list[str]:
    """Generate several *targeted* search queries for finding the one specific
    source paper a LitQA2 question is about.

    This replaced (CHANGELOG 0.7.0) an earlier "extract keyword phrases, then
    OR them all into one query" design that was THE dominant blocker: OR-ing
    every keyword returned 500,000+ papers sorted by recency, so the specific
    target paper was never in the fetched top-N. Verified live: for a
    question whose source was an ``Acinetobacter lwoffii`` antibiotic-
    resistance paper, the broad-OR query missed it entirely, while a targeted
    ``"Acinetobacter lwoffii" antibiotic resistance evolution`` query (with
    the source's own PubMed relevance sort) returned it at **rank 1**.

    So this returns a list of **complete, focused query strings** — each one
    combines the most specific/rare entity in the question (a gene, organism,
    protein, method; multi-word names quoted) with 1-2 context words. The
    caller (``retriever/chunk_rag.retrieve_relevant_chunks``) issues each as
    its own separate search and unions the results, the way a real retrieval
    agent runs several precise searches rather than one broad one.

    ``choices``: the shuffled answer options (no correct-answer marker) — a
    real solver sees these, and a distinct entity named only in an option is
    legitimate search signal. Not answer leakage: at query time nothing
    distinguishes the correct option from a distractor.

    Falls back to a quoted-noun-phrase heuristic if the LLM call fails, so a
    flaky reformulation never blocks retrieval outright."""
    choices_block = f"\n\nAnswer choices (one is correct, do not assume which):\n{format_choices(choices)}" if choices else ""
    prompt = (
        "Generate 3 to 5 focused literature-search queries to find the ONE specific paper "
        "that answers the question below. Each query should be a short string combining the "
        "MOST SPECIFIC entity in the question (a gene, protein, organism, cell line, method, "
        "or named system) with one or two context words — the way an expert would search a "
        "database, not a keyword dump. Put multi-word entity names in double quotes. Make the "
        "queries specific enough to return a handful of papers, not thousands. Use distinct "
        "entities across the queries (including any named only in the answer choices) so "
        "together they cover different ways the source paper might be indexed.\n\n"
        'Respond with exactly one JSON object on a single line: {"queries": ["query1", "query2", ...]}\n\n'
        f"Question: {question}{choices_block}"
    )
    try:
        # Fail fast (no retry, tight timeout) to the heuristic fallback: the
        # fallback now produces a genuinely targeted query (quoted entity),
        # so a slow/hanging flash reformulation is not worth waiting on —
        # better a fast decent query than a 90s hang per question at n=100.
        raw = await gemini_generate(prompt, api_key=api_key, model=model, http_timeout=http_timeout, max_retries=0)
        parsed = _parse_query_list(raw)
        if parsed:
            return parsed
    except Exception:
        pass
    return _fallback_queries(question)


def _parse_query_list(raw: str) -> list[str]:
    """Extract the query strings from the LLM response, robust to truncation.
    Measured live: the flash model routinely returns a *truncated* JSON array
    (cut off mid-string before the closing ``]``), which a strict
    ``json.loads`` of a ``[...]`` match rejects — silently dropping a
    perfectly good set of targeted queries and falling back to a junk one.
    So: try strict JSON first, then fall back to pulling every complete
    double-quoted string that appears after ``"queries": [`` even if the
    array was never closed."""
    import json as _json

    m = re.search(r'"queries"\s*:\s*(\[.*?\])', raw, re.DOTALL)
    if m:
        try:
            queries = _json.loads(m.group(1))
            if isinstance(queries, list):
                cleaned = [str(q).strip() for q in queries if str(q).strip()]
                if cleaned:
                    return cleaned
        except Exception:
            pass
    # Truncation-tolerant: grab complete "..."-quoted items after the array open.
    start = re.search(r'"queries"\s*:\s*\[', raw)
    if start:
        tail = raw[start.end():]
        items = re.findall(r'"((?:[^"\\]|\\.)*)"', tail)
        cleaned = [i.replace('\\"', '"').replace("\\\\", "\\").strip() for i in items]
        cleaned = [c for c in cleaned if c]
        if cleaned:
            return cleaned
    return []


_CAP_ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:[ -][A-Z0-9][A-Za-z0-9]*)*)\b")
_ALLCAPS_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,}[A-Za-z0-9]*)\b")
_QUESTION_WORDS = {"how", "what", "which", "when", "where", "who", "why", "does", "do", "is", "are", "can", "in", "among", "by"}
_STOP_CONTENT = {"been", "have", "has", "these", "that", "with", "from", "into", "following", "lab", "used", "using", "long", "many", "much", "fraction", "percent"}


def _fallback_queries(question: str) -> list[str]:
    """Build a targeted query without the LLM: quote the most distinctive
    entity (all-caps gene symbols like HSPA5/BiP first, then other proper
    nouns — but never the sentence-initial question word) and pair it with a
    couple of content words. Far better than the old
    join-the-first-6-non-stopwords string, which produced un-findable junk
    like 'Acinetobacter lwoffii has been evolved lab'."""
    allcaps = [e for e in _ALLCAPS_RE.findall(question)]
    proper = [
        e for e in _CAP_ENTITY_RE.findall(question)
        if len(e) > 2 and e.split()[0].lower() not in _QUESTION_WORDS
    ]
    entities = allcaps + [p for p in proper if p not in allcaps]
    content = [
        w for w in re.findall(r"[A-Za-z0-9\-]+", question)
        if len(w) > 3 and w.lower() not in _QUESTION_WORDS and w.lower() not in _STOP_CONTENT
    ]
    if entities:
        entity = entities[0]
        extra = [w for w in content if w.lower() not in entity.lower()][:3]
        return [f'"{entity}" ' + " ".join(extra)]
    return [" ".join(content[:5])] if content else [question]


_JSON_ANSWER_RE = re.compile(r'"answer"\s*:\s*"?([A-Za-z])"?')
_TAG_ANSWER_RE = re.compile(r"\[ANSWER\]\s*([A-Za-z])\s*\[/ANSWER\]", re.I)
_BARE_LETTER_RE = re.compile(r"^\(?([A-Za-z])\)?[.:)]?\s*$")


def parse_answer_letter(raw_text: str, n_choices: int) -> int | None:
    """Parses either astabench's ``{"answer": "X"}`` format or LAB-bench's
    ``[ANSWER]X[/ANSWER]`` format (see asta-bench's ``mark_multichoice_answer``),
    with a bare-single-letter fallback for a plain-text reply."""
    for pattern in (_JSON_ANSWER_RE, _TAG_ANSWER_RE):
        m = pattern.search(raw_text)
        if m:
            idx = string.ascii_uppercase.find(m.group(1).upper())
            if 0 <= idx < n_choices:
                return idx
    m = _BARE_LETTER_RE.match(raw_text.strip())
    if m:
        idx = string.ascii_uppercase.find(m.group(1).upper())
        if 0 <= idx < n_choices:
            return idx
    return None


async def retrieve_evidence_documents(
    sources: dict[str, Any],
    *,
    search_terms: list[str],
    profile: dict[str, Any],
    max_docs_per_source: int = 5,
) -> tuple[list[str], list[str]]:
    """Search + fetch real documents directly (title + abstract/body text),
    bypassing this service's claims extractor entirely.

    Why: the claims extractor is a structured, citation-grounded extraction
    step tuned for arXiv's LaTeX theorem/lemma environments (see
    ``extractors/claims.py``). PubMed and bioRxiv are abstract-only sources
    (``extraction_method="abstract_only"``, see their ``fetch_full_text``) —
    measured live, running the full ``/prior`` claims pipeline against
    LitQA2's biology questions retrieved real papers (``papers_indexed`` > 0)
    but extracted zero ``established_facts`` from them, because an abstract's
    prose rarely contains the specific signal phrases ("we show", "it is
    known that", ...) the extractor looks for. For question-answering, the
    raw retrieved text *is* the evidence — a QA-answering LLM is a perfectly
    good extractor of "what does this abstract say," and real LitQA2 systems
    (PaperQA-style agents) work exactly this way: read title+abstract/full
    text directly, no intermediate claim-extraction layer.
    """
    # ``query`` is the primary term; the rest go into profile["search_terms"]
    # so pubmed.py/arxiv.py's existing "OR every term together" query
    # construction actually uses all of them as separate clauses, not just
    # the first. Dropping this (i.e. passing only search_terms[0] with an
    # empty search_terms list) was tried and measured live: it silently
    # discards most of the reformulated keywords and starves retrieval.
    query = search_terms[0] if search_terms else ""
    search_profile = dict(profile)
    search_profile["search_terms"] = list(profile.get("search_terms", []) or []) + search_terms[1:]
    relevant = {name: src for name, src in sources.items() if src.is_relevant(search_profile)}

    async def _search(name: str, src: Any) -> tuple[str, list[Any]]:
        try:
            docs = await src.search(query, search_profile)
        except Exception:
            docs = []
        return name, docs[:max_docs_per_source]

    search_results = await asyncio.gather(*(_search(n, s) for n, s in relevant.items()))

    async def _fetch(name: str, raw_doc: Any) -> str | None:
        try:
            full = await sources[name].fetch_full_text(raw_doc)
        except Exception:
            return None
        text = (full.body_text or "").strip()
        if not text:
            return None
        # Kept short deliberately: measured live, the final answer call (question +
        # choices + N evidence snippets) timed out on gemini-3-flash-preview far
        # more often than the much shorter reformulation call — a smaller prompt
        # is the direct lever available without infrastructure changes.
        return f"{full.title} ({full.source}, {full.year or 'n.d.'}): {text[:700]}"

    fetch_tasks = [_fetch(name, raw) for name, docs in search_results for raw in docs]
    snippets = [s for s in await asyncio.gather(*fetch_tasks) if s]
    sources_with_hits = sorted({name for name, docs in search_results if docs})
    return snippets, sources_with_hits


_SUFFICIENT_RE = re.compile(r'"sufficient"\s*:\s*(true|false)', re.I)
_FOLLOWUP_RE = re.compile(r'"follow_up_query"\s*:\s*"([^"]*)"')


async def judge_evidence_sufficiency(
    question: str, choices: list[str], evidence: list[str], *, api_key: str, model: str, http_timeout: float = 45.0
) -> dict[str, Any]:
    """Real retrieval agents (ReAct and similar) don't answer after one
    search — they read what came back, decide whether it actually settles
    the question, and if not, search again with a refined query. This is
    that judgment step: given the current evidence, is it enough, and if
    not, what's a better thing to search for next. Defaults to "sufficient"
    on any parse/API failure — a broken judgment call should stop the loop
    and fall through to answering with whatever evidence already exists,
    not retry indefinitely or crash the question."""
    evidence_block = "\n\n".join(f"[{i + 1}] {e}" for i, e in enumerate(evidence)) if evidence else "(none yet)"
    prompt = (
        "You are deciding whether the evidence below is enough to confidently answer the "
        "question. If it is not — e.g. it's off-topic, or it's relevant but doesn't contain "
        "the specific fact/number needed — propose ONE short search phrase that would more "
        "directly find the missing information.\n\n"
        f"Evidence so far:\n{evidence_block}\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{format_choices(choices)}\n\n"
        'Respond with exactly one JSON object on a single line: '
        '{"sufficient": true or false, "follow_up_query": "short phrase or empty string"}'
    )
    try:
        raw = await gemini_generate(prompt, api_key=api_key, model=model, http_timeout=http_timeout)
        sufficient_m = _SUFFICIENT_RE.search(raw)
        followup_m = _FOLLOWUP_RE.search(raw)
        sufficient = sufficient_m.group(1).lower() == "true" if sufficient_m else True
        follow_up = followup_m.group(1).strip() if followup_m else ""
        return {"sufficient": sufficient, "follow_up_query": follow_up}
    except Exception:
        return {"sufficient": True, "follow_up_query": ""}


async def answer_one_question(
    pipeline: Any,
    *,
    question: str,
    ideal: str,
    distractors: list[str],
    profile: dict[str, Any],
    domain_id: str,
    rng: random.Random,
    google_api_key: str,
    llm_model: str,
    answer_model: str = "",
    depth: str = "standard",
    n_evidence: int = 6,
    n_answer_evidence: int = 20,
    max_rounds: int = 2,
) -> dict[str, Any]:
    """Retrieve real evidence for ``question`` (blind — the source paper is
    never told to the retriever, exactly like the real benchmark), then have
    an LLM answer from that evidence. Scored with LAB-bench's is_correct/
    is_sure semantics: an unparseable answer counts as neither correct nor
    sure (matching ``score_litqa2`` in asta-bench's task.py).

    Retrieval is now multi-round, not one-shot: search, judge whether the
    evidence actually settles the question (``judge_evidence_sufficiency``),
    and if not, search again with a refined query — this is the actual
    mechanism ReAct-style agents use (read, decide, refine, repeat), not
    just a single blind search-then-answer. Chunks accumulate across rounds
    and are re-ranked against the question once at the end
    (``retriever/chunk_rag.rerank_chunks``), so a later round's results
    compete on relevance with earlier ones rather than displacing them
    outright.

    Evidence is chunk-level retrieval ranked by similarity to the actual
    question (``retriever/chunk_rag.py``) — the third evidence strategy this
    module has tried, and empirically the best of the three measured so far:

    1. Raw truncated document text (``retrieve_evidence_documents``, kept
       below as the last-resort fallback): always the first ~700 characters
       regardless of where in the document the answer actually is.
    2. Citation-grounded ``established_facts`` via the LLM claim locator:
       100% citation-verified, but isolated single sentences with no
       surrounding paragraph — measured live (CHANGELOG.md 0.4.0), this
       *reduced* accuracy/precision relative to (1), because QA needs the
       context a lone sentence strips away.
    3. Chunk-level RAG (this): keeps paragraph context like (1) intact,
       while ranking by relevance to the question like neither (1) nor (2)
       does — chunks are scored against the question itself, not the
       document's position or a claim-worthiness judgment.

    ``domain_id``/``depth`` are accepted for interface symmetry with the
    rest of the pipeline but unused by the chunk-RAG path, which talks to
    sources directly rather than going through ``/prior``."""
    _ = domain_id, depth
    answer_model = answer_model or llm_model
    choices, target_idx, unsure_idx = build_choices(ideal, distractors, rng)
    # Chunk-ranking query = question + the real answer options (minus the
    # "insufficient information" option). The answer-bearing chunk often
    # shares its value with a choice ("4 weeks") but not with the question
    # ("how long do neurons survive"), so folding choices into the chunk BM25
    # query surfaces it — a surgical fix for the "paper found but answer chunk
    # not in top-k" failure mode, without the evidence-dilution that made
    # feeding whole papers regress (CHANGELOG 0.8.0).
    chunk_query = question + " " + " ".join(c for i, c in enumerate(choices) if i != unsure_idx)
    # Reformulation uses the pro model, not flash: query quality directly
    # determines whether the one source paper is found at all, and the flash
    # path proved flaky here (intermittently slow / truncated JSON → junk
    # fallback queries). Pro returns clean complete JSON reliably in ~10s —
    # the single highest-leverage place to spend a pro call besides the answer.
    search_terms = await reformulate_query_for_search(
        question, api_key=google_api_key, model=answer_model, choices=choices, http_timeout=40.0
    )

    all_chunks: list[Any] = []
    seen_chunks: set[tuple[str, str]] = set()
    sources_with_hits: set[str] = set()
    search_terms_by_round = [search_terms]

    for round_i in range(max_rounds):
        round_chunks, hit = await retrieve_relevant_chunks(
            pipeline.ctx.sources, pipeline.ctx.embedder,
            question=question, search_terms=search_terms, profile=profile,
            top_k=n_answer_evidence, ranker="bm25", chunk_query=chunk_query,
        )
        sources_with_hits.update(hit)
        for c in round_chunks:
            key = (c.source, c.text)
            if key not in seen_chunks:
                all_chunks.append(c)
                seen_chunks.add(key)

        if round_i == max_rounds - 1 or not all_chunks:
            break

        ranked_so_far = await rerank_chunks(pipeline.ctx.embedder, question, all_chunks, n_evidence, ranker="bm25")
        judgment = await judge_evidence_sufficiency(
            question, choices, [c.format() for c in ranked_so_far],
            api_key=google_api_key, model=llm_model,
        )
        if judgment["sufficient"] or not judgment["follow_up_query"]:
            break
        search_terms = [judgment["follow_up_query"]]
        search_terms_by_round.append(search_terms)

    # Feed the frontier answer model a generous slice of top chunks — pro
    # models have large context, and the answer-bearing chunk could be
    # anywhere in a full-text paper, so recall on the evidence set matters
    # more than keeping the prompt small (that constraint only applied to the
    # flaky flash path).
    ranked = await rerank_chunks(pipeline.ctx.embedder, chunk_query, all_chunks, n_answer_evidence, ranker="bm25")
    evidence = [c.format() for c in ranked]
    if not evidence:
        evidence, sources_with_hits_fallback = await retrieve_evidence_documents(
            pipeline.ctx.sources, search_terms=search_terms_by_round[0], profile=profile
        )
        evidence = evidence[:n_evidence]
        sources_with_hits.update(sources_with_hits_fallback)
    prompt = build_prompt(question, choices, evidence)
    # Final answer uses the frontier reasoning model (answer_model), not the
    # cheap flash model used for reformulation/judging above. Longer timeout:
    # pro models reason for longer, and there's no embedding-burst contention
    # now that ranking is local BM25.
    # 180s, not 120: the answer prompt asks the pro model to reason step by
    # step, which produces longer responses — measured live, a 120s cap +
    # concurrency-6 load caused ~14% of answers to time out (each a scored
    # miss), pure lost accuracy that had nothing to do with answer quality.
    # One retry inside gemini_generate on top of this.
    raw = await gemini_generate(prompt, api_key=google_api_key, model=answer_model, http_timeout=180.0)
    picked_idx = parse_answer_letter(raw, len(choices))

    if picked_idx is None:
        is_correct, is_sure = False, False
    else:
        is_sure = picked_idx != unsure_idx
        is_correct = picked_idx == target_idx

    return {
        "question": question,
        "search_terms_by_round": search_terms_by_round,
        "n_rounds": len(search_terms_by_round),
        "choices": choices,
        "target_idx": target_idx,
        "unsure_idx": unsure_idx,
        "picked_idx": picked_idx,
        "picked_text": choices[picked_idx] if picked_idx is not None else None,
        "is_correct": bool(is_correct),
        "is_sure": bool(is_sure),
        "n_evidence": len(evidence),
        "sources_consulted": sorted(sources_with_hits),
        "raw_llm_output": raw[:300],
    }


def score_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """LAB-bench's exact metrics, computed the same way
    ``asta-bench``'s ``score_litqa2`` scorer aggregates them."""
    n = len(results)
    if n == 0:
        return {"n": 0, "accuracy": None, "coverage": None, "precision": None}
    n_correct = sum(1 for r in results if r["is_correct"])
    n_sure = sum(1 for r in results if r["is_sure"])
    n_correct_and_sure = sum(1 for r in results if r["is_correct"] and r["is_sure"])
    return {
        "n": n,
        "accuracy": round(n_correct / n, 4),
        "coverage": round(n_sure / n, 4),
        "precision": round(n_correct_and_sure / max(1, n_sure), 4),
    }
