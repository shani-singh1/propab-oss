# Changelog

Eval scores before/after each significant change. Full score history:
`artifacts/astabench_literature_scores.json` (append-only).

## 0.9.0 — breaking the 0.71 plateau: deep-read the source paper (+5-7pp)

Continuing the loop toward the AstaBench 0.84 baseline. The starting point was
a stubborn **0.71 (n=100)** plateau, and the story here is mostly about *not*
being fooled by it — three separate changes each looked like a win at n=50 and
were confirmed as noise at n=100 before the one real lever landed.

**Diagnosis first (the instrumentation that made this tractable).** Rather than
guess, the eval was instrumented to record, per question, which papers reached
the answer prompt (`retrieved_titles`/`retrieved_urls`) and the score run cross-
checked those against LitQA2's known source DOI and gold `key_passage`. Two
facts fell out and reframed the whole problem:

1. **The source paper is retrieved in ~78% of cases** — search is NOT the main
   bottleneck. Only ~4 of ~16 failures per 50 are true retrieval misses.
2. **The dominant failure is "paper retrieved, answered wrong" (~12 of 16).**
   And the single strongest predictor of a correct answer is *best-single-chunk
   overlap* with the gold passage: **0.74 on correct answers vs 0.36 on wrong**.
   Union overlap (passage present *somewhere* across all chunks) barely
   separates them (0.89 vs 0.64) — so the problem is that the answer-bearing
   passage, even when retrieved, arrives *fragmented* across a soup of chunks
   drawn from ~12 different papers, and the model then reasons from parametric
   priors ("by homology…") instead of the paper's actual reported result.

**Three measured nulls (documented so they're not re-tried).** Each was A/B'd at
n=50, looked good, and evaporated at n=100 — the exact small-sample trap:

| lever | n=50 | n=100 | verdict |
|---|---|---|---|
| chunk size 900→1600 | 0.74 | 0.70 | null (reverted) |
| choice-aware doc-fetch ranking | 0.68 | — | flat, recall 39→40 only (reverted) |
| evidence-over-priors answer prompt | 0.72 | 0.69 | null (reverted) |

The lesson (again): **n=50 has ±7pp; only n=100 counts.** Retrieval-recall and
prompt tuning could not move the plateau, which pointed the finger squarely at
*how the retrieved source paper is presented to the model*.

**The win — deep-read the single most-relevant paper.** The leaderboard leaders
(AI2's ReAct agent at 0.82, FutureHouse's PaperQA2) don't answer from a cross-
paper chunk mix; they identify the source paper and read *it* deeply. Ported to
this pipeline: the top-ranked fetched document (already the most likely source
by title+abstract relevance) now gets a large, **guaranteed** chunk budget
(`deep_read_k=10`) that leads the evidence and is never filtered out by the
global top-k, while the other papers stay shallow (a few chunks each) as backup
coverage. This concentrates depth on ONE paper — categorically different from
the 0.8.0 experiment that fed the top *three* papers in full and regressed to
0.48 through dilution; here the depth is on a single paper and relevance-ranked.

Confirmed on **two independent 100-question samples**, which is what separates
it from the three nulls above:

| | Accuracy | Coverage | Precision |
|---|---|---|---|
| chunk-soup baseline (0.8.0), seed 0 | 0.71 | 0.97 | 0.73 |
| **deep-read (0.9.0), seed 0** | **0.76** | 0.97 | 0.78 |
| **deep-read (0.9.0), seed 1** | **0.78** | 0.97 | 0.80 |

**+5-7pp accuracy AND +5-7pp precision, holding across seeds.** The precision
jump is the mechanism showing through: when the model commits, it is now right
far more often, because it is reading the actual source paper rather than
guessing around a fragmented passage. Coverage is unchanged (still ~forced-
answer). Project arc: 0.44 → 0.60 → 0.71 → **0.77 (deep-read)**, now inside the
0.76-0.80 band and closing on the 0.84 AstaBench baseline.

**Honest remaining gap.** ~0.77 vs 0.84. What's left is the ~22% of questions
where the source paper isn't surfaced by any public API at all (the retrieval-
coverage ceiling of 0.8.0, unchanged here) plus a residual reasoning-error floor
on genuinely ambiguous options — both requiring capability beyond single-pass
retrieval + one answer call (a snippet index or a true multi-hop read-search
agent), which is a separate build, not another tuning pass.

## 0.8.0 — breaking the 0.60 ceiling: read the pattern, force the answer (+11pp)

**Context: accuracy was pinned at 0.60 (n=100) across several retrieval
changes.** This entry is an honest improvement *loop* — diagnose, change one
thing, A/B at n=50, confirm at n=100, commit wins, revert losses — including
the dead ends, because the dead ends are what pointed at the real lever.

**Two changes that regressed and were reverted (measured, not guessed):**
- *Whole-document evidence* (feed the pro model the top 3 papers in full
  instead of BM25-picked chunks): **0.48** at n=50, down from ~0.60. Lesson:
  the constraint is not evidence *quantity* — more text diluted the signal
  with off-topic/wrong-paper content, and precision dropped. Reverted.
- *Self-consistency* (sample the answer 3× at temperature, majority vote):
  **0.46**. For a well-calibrated frontier model at temp 0, adding temp-0.5
  samples and voting added noise rather than removing it. Reverted.
  (Also caught a process error here: I'd been treating 12pp swings on n=50,
  which carries ±7pp, as real. Re-confirmed the committed baseline reproduces
  at **0.68** on n=50 / 0.60 on n=100 before continuing — n=100 is the number
  to trust.)

**Choice-aware chunk ranking (kept).** Rank chunks against the question PLUS
the answer options, not the question alone: the answer-bearing chunk often
shares its value with a choice ("4 weeks") but no words with the question
("how long do neurons survive"), so folding the choices into the chunk BM25
query surfaces it — surgical, no evidence dilution. Document ranking stays
question-only so choices don't bias *which* paper is judged the source. Net
at n=100: accuracy-neutral but precision up (0.79→0.82) — a real quality gain
that set up the win below.

**The confirmed retrieval ceiling (why more retrieval stopped helping).**
Tested the actual hard-to-find source papers against every public source —
PubMed, Europe PMC (including its full-text search), Semantic Scholar,
OpenAlex — and **none surface the exact paper in their top results.** LitQA2
questions quote specifics from the paper *body*; every public API's relevance
ranking puts other papers first. Widening the net (5 query variants, 16 docs)
was flat on accuracy and slower — reverted. This is the gap a proprietary
full-text snippet index (AI2) or a heavy multi-hop agent (FutureHouse's
PaperQA2) closes; single-pass public-API retrieval caps coverage around 0.78.

**The win, found by reading the pattern (forced answer).** Every retrieval
improvement raised precision (→0.82-0.84) but *lowered* coverage (→0.73):
with sharper evidence the well-calibrated pro model correctly recognizes the
genuinely-unfindable questions and abstains — and each abstention is a
guaranteed zero on the accuracy metric. So the lever was never more
retrieval; it was to stop abstaining. Changed the answer prompt to forbid
"Insufficient information" entirely — the model must eliminate implausible
options and commit to the single most likely substantive answer, always. The
coverage×precision math predicted the payoff and the measurement matched it:

| | Accuracy | Coverage | Precision |
|---|---|---|---|
| calibrated abstention (0.7.0) | 0.60 | 0.78 | 0.79 |
| **forced answer (0.8.0)** | **0.71** | 0.97 | 0.73 |

**+11pp accuracy at n=100 (±5pp)** — coverage to 0.97, precision drops as
expected, net strongly positive because forced best-guesses replaced
guaranteed zeros. This lands in FutureHouse's (0.72) territory, on public
infrastructure. Project arc: 0.44 (flash, one-shot) → 0.60 (targeted search)
→ **0.71 (forced answer)**, vs the 0.84/0.94/0.89 AstaBench baseline.

**Honest remaining gap.** 0.71 vs 0.84. Precision is now the axis (0.73 vs
0.94): the forced guesses on unfindable-paper questions are right less often
than the baseline's evidence-backed answers. Closing that needs better
retrieval coverage of the hard papers (the ceiling above) — i.e. the
proprietary-index / heavy-agent capability, a substantial separate build,
not prompt tuning.

## 0.7.0 — the real blocker was query construction (targeted search, +8pp)

**Reframe, prompted by the user refusing the "proprietary index" excuse:**
FutureHouse scored 0.72 and AI2 scored 0.82 on public infrastructure with
the same model class — so 0.52 was an engineering failure, not a structural
ceiling. Diagnosed the actual failures per-question: 28 of 100 were
abstentions ("insufficient information"), and a live probe showed why — the
source paper wasn't in the retrieved evidence *at all* for those. Found the
cause and it was mine:

**The blocker: OR-ing every reformulated keyword into one query.** Measured
live against real PubMed: the broad-OR query for an ``Acinetobacter lwoffii``
antibiotic-resistance question returned **572,021 papers sorted by recency**,
so the specific source paper was never in the fetched top-N. The *same
paper* was returned at **rank 1** by a targeted ``"Acinetobacter lwoffii"
antibiotic resistance evolution`` query with PubMed's relevance sort. Pure
engineering bug.

**Fixes:** ``reformulate_query_for_search`` now produces several **complete
targeted query strings** (quoted entity + context), not a keyword list to OR;
``retriever/chunk_rag`` issues each as its own search and unions results.
PubMed: ``sort=relevance`` + AND context terms instead of OR-broadening.
Europe PMC: fixed a double-quoting bug that silently broke targeted queries.
Reformulation moved to the **pro model** (reliable clean JSON in ~10s) after
the flash path proved intermittently slow/truncated → junk-fallback queries;
parse made truncation-tolerant; heuristic fallback rewritten to quote the
rarest entity (gene symbol / proper noun).

**Measured, n=100 seed-42 (±5pp), before → after the search fix:**

| | Accuracy | Coverage | Precision | Timeouts |
|---|---|---|---|---|
| 0.6.0 (broad-OR search) | 0.52 | 0.66 | 0.79 | 6 |
| **0.7.0 (targeted search)** | **0.60** | **0.78** | 0.77 | 1 |

**+8pp accuracy, via exactly the predicted mechanism:** coverage rose
0.66→0.78 because the targeted search finds the source paper far more often,
so fewer questions abstain for lack of evidence. Against the 0.84/0.94/0.89
baseline. Full project arc: 0.44 (flash, one-shot) → 0.52 (pro model +
anti-abstention) → **0.60 (targeted search)**.

**Remaining gap, honestly.** 0.60 vs 0.84, still below FutureHouse (0.72) and
AI2 (0.82) — more real engineering left, not a wall. Two open axes: coverage
0.78 vs 0.89 (~11% of source papers still not found — preprints, very
specific entities), and **precision 0.77 vs 0.94** (when we commit, wrong 23%
vs the baseline's 6%). The precision gap is now the larger opportunity: if
answers on retrieved papers matched the baseline, 0.78 × 0.94 ≈ 0.73. That's
a retrieval-*precision*/chunk-selection problem — retrieving the right paper
but not surfacing the exact answer-bearing chunk (a table number, a specific
results sentence), or occasionally answering from a similar-but-wrong paper —
the next concrete target.

## 0.6.0 — the two things that actually move LitQA2: a frontier answer model
and document-precision retrieval

**Reframe, prompted by the user pushing back on infrastructure tinkering:**
the timeout fix (0.5.0) only ever raised the floor — it turned auto-wrong
timeouts into real answers. The ceiling was always ``correct-when-answered``,
which sat at **0.61** regardless of how many sources or how much full text
was added. 0.61 → 0.84 is a *reasoning-and-retrieval-precision* problem, not
an infrastructure one. Two changes attack it directly:

**1. Frontier answering model (`gemini-3.1-pro-preview`).** The single
biggest lever, and one hiding in plain sight the whole time: every LitQA2
answer this project ever produced was generated by `gemini-3-flash-preview`
— a *flash* model — while the AstaBench baseline it's chasing is Claude Opus
4.7 at `xhigh` reasoning. You cannot out-reason a frontier-model baseline
with flash, no matter how good the evidence. Checked what the API key
actually exposes: `gemini-3.1-pro-preview` (and `gemini-2.5-pro`,
`deep-research-pro-preview`) are all available — same weight class as the
baseline. `config.py` now has a separate `answer_model` used *only* for the
final answer; the cheap sub-tasks (query reformulation, sufficiency judging)
stay on flash to control latency/cost. Immediate validation signal (n=5):
**precision 1.0** — every confident answer correct — where flash had been
guessing wrong routinely.

**2. Document-precision retrieval (two-stage ranking + per-document chunk
cap), fixing a bug found by looking at the actual chunks.** LitQA2 questions
are each about ONE specific paper. The naive "fetch everything, chunk
everything, take the global top-k chunks" flow had a fatal failure mode,
confirmed live on the Ddd1-deaminase question: **12 of 20 evidence chunks
came from a single lexically-similar-but-wrong long paper**, while the actual
source paper contributed 1 chunk — so the pro model correctly abstained
("insufficient information") because the answer genuinely wasn't in the
evidence it was handed. Rebuilt `retriever/chunk_rag.retrieve_relevant_chunks`
into two stages: (a) rank *candidate documents* by BM25 relevance of
title+abstract to the question and only fetch full text for the top few —
document-level precision before spending fetch budget; (b) rank chunks
*within each document* and cap at `max_chunks_per_doc`, so no single paper
can dominate the evidence set. Re-probed the same question: the 12/20
concentration is gone, evidence is now diverse across the papers actually
worth reading.

**3. BM25 local chunk ranking (`rank_chunks_bm25`), replacing dense Gemini
embedding in the QA hot path.** Not just a speed fix — it removes the root
cause of the ~28% `ReadTimeout` rate that capped every prior measurement.
Dense ranking fired up to ~90 concurrent `embedContent` HTTP calls per
question; that burst was the contention source (0.5.0 diagnosed the symptom,
this removes the cause). BM25 is zero-network, instant, and genuinely strong
for a task that turns on specific named entities/genes/numbers — lexical
overlap finds the chunk that literally contains them. Dense ranking stays
available via `rerank_chunks(..., ranker="embedding")` for the `/prior`
semantic-dedup path where contention isn't the constraint. Validation (n=5):
**zero timeouts**, down from 30-40%.

**4. Concurrent question processing (`scripts/run_litqa2_real.py`).**
Questions now run N-at-a-time (bounded semaphore) with a per-question
deterministic RNG seeded from the question id (so concurrency doesn't make
choice-shuffling order-dependent, and runs stay reproducible). This makes
`n=50`/`n=100` measurements feasible in reasonable wall time — the only way
past the `n=25` noise floor (±10pp) that made every single-run comparison in
0.3.0-0.5.0 statistically meaningless.

**5. The biggest single win: stop abstaining (answer-prompt calibration).**
The first n=50 run with the pro model measured **0.34 / 0.44 / 0.77**
(accuracy/coverage/precision) — and the shape was the whole lesson. Precision
0.77 means the model was *right when it committed*; coverage 0.44 means it
abstained ("Insufficient information") on 56% of questions. Key realization:
**for the accuracy metric, abstaining is strictly worse than guessing** —
accuracy is correct/total, an abstention scores a guaranteed zero, whereas a
reasoned best-guess is right some fraction of the time. The baseline answers
89% of questions (coverage 0.89) for exactly this reason. The pro model was
*too* well-calibrated for this scoring. Rewrote `build_prompt` to commit to
the single most plausible option using evidence + domain knowledge, reserving
abstention for genuinely-indistinguishable cases only. Clean A/B on the
**same** n=50 seed-42 sample (only the prompt changed):

| | Accuracy | Coverage | Precision |
|---|---|---|---|
| well-calibrated abstention | 0.34 | 0.44 | 0.77 |
| **commit-to-best-answer** | **0.54** | 0.70 | 0.77 |

**+20 points of accuracy from one prompt change**, with precision holding at
0.77 (the extra answers were mostly right). This is the largest single
improvement of the project.

**Progression (accuracy/coverage/precision):** flash + one-shot retrieval
~0.34-0.44 (n=25, noisy) → pro model + document-precision retrieval,
well-calibrated abstention: 0.34 (n=50) → + commit-to-best-answer prompt:
0.54 (n=50) → + answer-timeout fix (180s): 0.58 (n=50).

**Definitive headline, n=100 seed-42 (±5pp, the most trustworthy measurement
this project has produced): 0.52 / 0.66 / 0.79** (accuracy/coverage/
precision), 6% timeouts. The n=50 read (0.58) and n=100 read (0.52) are
consistent within their error bars; the true value sits around **0.52-0.55**,
and n=100 is the number to trust. Against the **0.84 / 0.94 / 0.89**
leaderboard baseline (ReAct + Claude Opus 4.7, official test split).

**Honest bottom line.** This is a real ~+8-11pp over the flash baseline on
trustworthy measurement — the pro answer model and (dominant) anti-abstention
calibration did the work, with BM25/zero-timeouts and document-precision
retrieval as necessary enablers. It does **not** reach or beat 0.84. The
remaining ~0.30 gap is now cleanly attributable to **retrieval recall**:
precision 0.79 says the model answers correctly when the source paper's
content reaches it, but coverage 0.66 says ~34% of the time the exact
one-paper-per-question source isn't found by public-API keyword search or
isn't ranked high enough to fetch. Closing that is a document-recall problem
(query expansion, a persistent full-corpus index, more/better sources), which
is precisely the axis where the top baselines' proprietary snippet index over
a curated corpus has a structural advantage over public-API search — the
honest next frontier, not more answer-side tuning.

**Where the remaining gap lives, now unambiguous:** the binding constraint is
**retrieval recall** — coverage 0.70 with precision 0.77 means that when the
answer-bearing content reaches the model it usually answers correctly, but
~30% of the time the exact source paper (LitQA2 is one-paper-per-question)
either isn't found by public-API keyword search or isn't ranked high enough
to be fetched. This is precisely the axis where a proprietary full-corpus
snippet index (what the top baselines have) beats public-API search, and
it's the honest next frontier: better document recall (query expansion,
more sources, a persistent index) rather than more answer-side tuning.

## 0.5.0 — real retrieval infrastructure (PMC full-text, Europe PMC, chunk RAG,
multi-round search) + the honest statistical limit of n=25

**Why:** pushed back on treating "AI2 has proprietary retrieval infra" as a
terminal excuse — other teams without that access (FutureHouse, Elicit,
SciSpace, You.com, Perplexity) compete on the same leaderboard on public/
commercial APIs, and even AI2's own tools likely only have privileged depth
on Semantic Scholar specifically, not PubMed Central, full-text table
extraction, or multi-round search. Those are addressable with real
engineering, not access. This entry is that engineering.

**Real API keys wired.** `SEMANTIC_SCHOLAR_API_KEY`/`NCBI_API_KEY` added to
`.env`. Same bug class as `google_api_key` earlier: `config.py` had no
unprefixed alias for either, so they'd never have been read (`LITERATURE_`
prefix would've been required) — fixed alongside wiring them in.

**PMC full-text fetching (`sources/pubmed.py`, new `sources/_pmc.py`).**
Confirmed live against real articles: PubMed's abstract-only path was
capping evidence at ~2-4K characters; PubMed Central's OA subset serves full
JATS XML for the same paper (measured: 16,049 characters real full text vs.
a ~2K abstract, for a paper this session's own LitQA2 questions reference).
Not every PMC-indexed article allows this — many journals opt into PMC
without granting bulk-download rights, detected via an explicit marker
NCBI's `efetch` returns instead of a `<body>`, handled by falling back to
the abstract rather than guessing. Table extraction (`<table-wrap>` →
pipe-delimited text) added in the same pass, since LitQA2 answers are
routinely a number that lives in a table, never in prose.

**New source: `sources/europepmc.py`.** `bioRxiv`'s `search()` has returned
`[]` all session — the public bioRxiv API has no free-text search endpoint
at all, a known, documented gap never closed until now. Europe PMC is a
real relevance-ranked full-text search engine covering PMC + bioRxiv +
medRxiv + MEDLINE in one index, no key required, and its results carry a
PMCID directly (skips the id-conversion round trip `pubmed.py` needs).
Confirmed live: found the exact source paper for a real LitQA2 question on
the first search, fetched 33,324 characters of real full text.

**New: `retriever/chunk_rag.py` — chunk-level embedding retrieval.** The
core architectural upgrade this entry is really about. Prior evidence
strategies were: (1) raw truncated document text (first ~700 characters
regardless of where the answer lives) or (2) isolated single-sentence
claims via the LLM locator (0.4.0 — citation-verified but stripped of
context, measured to *reduce* accuracy). This is PaperQA2-style RAG: chunk
full documents into paragraph-sized pieces, embed every chunk with Gemini,
rank by cosine similarity to the *actual question* — keeps paragraph
context intact like (1) while ranking by relevance like neither (1) nor (2)
did.

**Multi-query reformulation from question + choices
(`reformulate_query_for_search`).** Real LitQA2 solvers see the shuffled
answer choices, not just the question — entity names in *any* option
(correct or distractor; nothing distinguishes them at reformulation time)
are legitimate retrieval signal a real agent would use too. Not answer
leakage: the model has no way to know which option is correct when
generating search terms.

**Multi-round retrieval loop (`judge_evidence_sufficiency` +
`chunk_rag.rerank_chunks`).** The actual mechanism ReAct-style agents use
that this service's one-shot search-then-answer never did: retrieve, ask
an LLM whether the evidence actually settles the question, and if not,
search again with a refined query, accumulating and re-ranking the
combined chunk pool before a final answer (capped at 2 rounds).

**A real infrastructure finding, verified rather than assumed:** isolated
Gemini calls with realistic prompt sizes (~9K characters, matching the
actual answer-with-evidence prompt) succeeded 100% of the time in direct
testing — `gemini-3-flash-preview` is not an inherently broken model.
Reproducing the *same* call under synthetic concurrent load (30 simultaneous
embedding calls, matching what one retrieval round actually fires)
increased latency (12s vs. 4-8s) without an outright failure. The
persistent ~30-40% `ReadTimeout` rate measured across live eval runs this
session is best explained by sustained concurrent HTTP load during
retrieval (search + full-text fetch + chunk embedding, all firing at once,
now compounded by multiple sequential LLM calls per question from the
multi-round loop) — most plausibly local network/connection contention in
this environment, not a fundamental reliability problem with the model or
architecture. Mitigated (not eliminated) by capping chunks per round
(`max_docs_per_source` 5→3, `max_total_chunks=60` before embedding —
previously an unbounded burst, since a single 15-30K character PMC
full-text document alone produces 15-30+ chunks).

**The honest statistical read, which matters more than any single number
here.** Ten `n=25` live runs across this session and the previous one
produced accuracy values of 0.44, 0.24, 0.28, 0.44, 0.32, 0.44, 0.44, 0.32,
0.44 (excluding one early run invalidated by the over-cautious-prompt bug
in 0.3.0, since-fixed). At `n=25`, the binomial standard error for an
accuracy near 0.4 is ≈ ±10 percentage points (1 SE) — **the entire observed
0.24-0.44 spread is consistent with a single true accuracy of roughly
0.35-0.40 and no real per-change effect at all.** The mode (0.44, 5 of 9
valid runs) is a better point estimate than any individual run, including
this entry's own final combined-everything measurement (0.44 / 0.72 /
0.6111). **This means most of the single-intervention comparisons drawn
earlier in this session (citation-fix "hurt" accuracy, Europe PMC alone
"hurt" accuracy, etc.) cannot be reliably attributed to those specific
changes** — they are equally consistent with sampling noise. A trustworthy
verdict on any single architectural change needs either a substantially
larger sample (`n=100` → ±5pp SE, `n=400` → ±2.5pp) or multiple repeated
runs per configuration to estimate variance directly; continuing to
compare single `n=25` runs against each other is not a reliable signal and
should stop.

**Where this leaves the real number:** best current estimate ≈ 0.35-0.44
accuracy against the real leaderboard baseline of 0.84 (ReAct + Claude Opus
4.7, official test split). The gap has not closed with a good margin, and
this entry's honest conclusion is that it will not close further from
continued single-shot `n=25` comparisons — the next real step is either (a)
a properly-sized measurement to get a trustworthy number before deciding
what to try next, or (b) accepting that closing the remaining gap requires
retrieval-infrastructure depth (a persistent index, not one built fresh per
question — `qdrant_backend`/`postgres_backend` are built but off by
default) that goes beyond what a single session's engineering can close.

## 0.4.0 — LLM claim location (code-only verbatim) + a negative result

**The mandate:** fix citation verification to ≥90% first, in isolation,
before touching anything else — separate claim *location* (an LLM finds
which sentence) from *verbatim extraction* (code alone pulls the exact
string from that location; the LLM never supplies the quoted text, even if
it echoes one back). Then, and only then, check whether the LitQA2 gap was
downstream of citation quality.

**What was already true, and what wasn't:** the core `/prior` pipeline had
no LLM in its citation path at all before this — `extractors/claims.py` and
`sources/_latex.py` are pure regex/structural extraction, verbatim is
always `text[start:end]` off the source. Citation-verification rate was
already ~100% (see 0.1.0/0.2.0 entries). What was actually broken was
**recall on abstract-only sources**: PubMed/bioRxiv prose rarely contains
the exact signal phrases ("we show", "it is known that", ...) the regex
scan requires, so `established_facts` came back empty for those sources —
this is the same gap `litqa2_live.py`'s `retrieve_evidence_documents` was
built to route around by bypassing claim extraction entirely (see 0.3.0).

**New: `extractors/llm_claim_locator.py`.** Code splits a document into a
numbered sentence list (`split_sentences` — same paragraph-scoped splitter
as `claims.py`, so it can't splice across a structural seam). An LLM is
given *only* the numbered list and asked which indices state a claim, plus
a type/status judgment — explicitly instructed not to restate the sentence.
The response is parsed for indices only; `verbatim` is always
`sentences[index]`, a pure code-side lookup. Even if a model ignores
instructions and includes a paraphrased "text" field in its response
anyway (tested for directly —
`test_llm_claim_locator.py::test_ignores_llm_supplied_text_field`, a model
does exactly this and the paraphrase is discarded), it is never read.
Wired into `retriever/query.py::process_document` as a fallback: only runs
when regex extraction found nothing, so LaTeX-rich arXiv papers are
unaffected and cost/latency only applies where it's needed. Gated on
`ctx.llm_api_key` (unset ⇒ identical behavior to before).

New shared `app/llm_client.py` holds the Gemini call (moved out of
`evaluator/litqa2_live.py`, which now imports it — one implementation, two
callers, both non-core-citation-path uses).

**Also fixed while measuring:** `evaluator/metrics.py::_external_id_for` had
no case for `pubmed` — it fell through to `source_doi or source_url`, which
`sources/pubmed.py`'s `fetch_full_text` cannot resolve (it needs a bare
PMID). Every PubMed claim was silently "unverifiable" (excluded from the
rate's denominator, not counted as a failure) rather than actually
re-checked — invisible until claims from PubMed existed to check, which
they didn't until this entry.

**Measured live (real papers, real Gemini calls, not mocked):** a `/prior`
run against real PubMed/bioRxiv/Semantic Scholar results for a cancer-
immunology query produced 27 claims (up from ~0 for these sources before —
recall fix confirmed), 25 of them via the new locator.
`citation_verification_rate` on those 25: **25/25 (100%)**. Combined with
the 2 regex-extracted claims in the same run: **27/27 (100%)**. This
confirms the location/verbatim-separation design empirically, not just via
the mocked unit tests — see `scripts/verify_llm_locator_citations.py`.

**Then reran LitQA2 with this fixed evidence path** (`answer_one_question`
now uses `pipeline.build_prior()`'s `established_facts` as primary evidence,
falling back to raw document text only when that's empty — same 25-question
sample, same seed, directly comparable to the 0.3.0 numbers):

| Metric | Before (0.3.0, raw abstract text) | After (0.4.0, located claims) |
|---|---|---|
| Accuracy | 0.44 | **0.32** |
| Coverage | 0.64–0.68 | 0.68 |
| Precision | 0.65–0.69 | **0.47** |

**This is a real negative result, not a bug to chase.** The hypothesis —
"the LitQA2 gap is downstream of citation/extraction quality, so fixing
extraction should raise both metrics together" — is not supported by this
measurement. Citation *fidelity* was never the problem (it was already
~100% before this entry); citation *fabrication* was never happening. What
changed is evidence *granularity*: `established_facts` are isolated single
sentences, deduped across however many documents contributed claims, with
no surrounding paragraph; the raw-abstract-snippet approach it replaced
handed the answering LLM a full ~700-character contiguous excerpt with
context intact (what organism, what condition, what experiment) even though
that text was never checked against any per-sentence citation contract.
Precision dropping more than accuracy is the tell: the model is answering
just as often (coverage held), just more often wrong — consistent with
losing disambiguating context, not with losing correct information. Kept
both evidence paths in the code (`established_facts` primary,
`retrieve_evidence_documents` fallback) rather than reverting, since the
citation-quality fix is real, valuable, and correct on its own terms
(recall + fidelity for `/prior`, independent of this one eval); the honest
conclusion is that it isn't the fix for the LitQA2 gap, and the LitQA2 gap
remains what 0.3.0 already identified (retrieval infrastructure, single-
round search) plus, now additionally: evidence context granularity.

## 0.3.0 — real LitQA2 evaluation against the actual AstaBench leaderboard

**Why:** `astabench.run_litqa2_proxy` only ever scored this service against
its own hand-built cases with embedding similarity — never against real
LitQA2 data, and never against a real published baseline. This entry closes
that gap: found the actual live AstaBench leaderboard, pulled real baseline
numbers, downloaded the real (public, non-gated) LitQA2 question set, built
an LLM-answering path this service didn't have before, and ran it.

**The real baseline** (pulled from the public `allenai/asta-bench-results`
HuggingFace dataset — the leaderboard's actual raw submission data, not a
summary page — via `Ai2_ReAct_2026-04-21T23-37-05.json`, a ReAct agent using
`anthropic/claude-opus-4-7` at `reasoning-effort=xhigh`):

| Metric | ReAct + Claude Opus 4.7 (official test split) |
|---|---|
| LitQA2_FullText accuracy | 0.84 |
| LitQA2_FullText precision | 0.94 |
| LitQA2_FullText coverage | 0.89 |
| PaperFindingBench adjusted F1 | 0.37 (0.43 for the specialized Asta Paper Finder agent) |
| ScholarQA_CS2 global avg | 0.86 (0.90 for Asta Scholar QA w/ Claude 4.6) |

**New: `evaluator/litqa2_live.py` + `scripts/run_litqa2_real.py`.** Downloads
real questions from `futurehouse/lab-bench` (LAB-Bench's public LitQA2
subset — the same dataset AstaBench's official harness wraps in
`asta-bench/astabench/evals/labbench/litqa2/task.py`; the *task-definition*
dataset `allenai/asta-bench` that maps questions to the official dev/test
split is gated, so this run draws from the full 199-question public set,
not necessarily the identical test-split partition — disclosed here, not
hidden). For each question: reformulates it into search keywords with
Gemini, retrieves real documents from PubMed/bioRxiv/Semantic Scholar/
Crossref, then has Gemini answer from that evidence — matching how every
real leaderboard entry works (retrieve, then reason), unlike the
embedding-similarity-only proxy. Scored with LAB-bench's own metrics
(accuracy/coverage/precision), including its "Insufficient information"
option and its is_correct/is_sure semantics.

**Real bugs found and fixed via live runs against real questions** (the
mocked test suite structurally cannot catch any of these — they only show
up against real API responses and real model behavior):

1. **Full-sentence queries return nothing.** PubMed's esearch (and, more
   subtly, Semantic Scholar) implicitly ANDs every word of a query together;
   a raw 20-word question matched zero abstracts (measured: 0 hits). 3-4
   substantive keyword phrases matched 6. Fixed by reformulating each
   question into search terms with Gemini before retrieval
   (`reformulate_query_for_search`) — first pass concatenated the terms into
   one string and lost this benefit entirely (still 0 hits); the actual fix
   was keeping them as a list so `pubmed.py`/`arxiv.py`'s existing
   "OR every term together" query construction treats each as its own
   clause (confirmed live: PubMed correctly scopes OR per-clause even when
   one clause is a whole dead-weight sentence).
2. **The claims extractor is the wrong evidence source for QA.** Running
   the full `/prior` pipeline against these questions found real papers
   (`papers_indexed > 0`) but extracted zero `established_facts` — PubMed/
   bioRxiv are abstract-only sources, and an abstract's prose rarely
   contains the specific signal phrases ("we show", "it is known that", ...)
   `extractors/claims.py` looks for. Added `retrieve_evidence_documents`,
   which hands the LLM raw retrieved title+abstract text directly — a
   QA-answering LLM is a perfectly good "extractor" of what an abstract
   says; real LitQA2 systems work exactly this way.
3. **Miscalibrated prompt caused 21/21 abstentions despite real evidence.**
   Modeled after typical RAG caution ("if unsure, say so"), the first
   answering prompt told the model to prefer "Insufficient information"
   over an uncertain guess. Result: 0% accuracy, 0% coverage, every single
   answer abstaining — even questions with 7-8 retrieved evidence snippets.
   Checked LAB-bench's own reference zero-shot prompt
   (`Future-House/LAB-Bench/labbench/zero_shot.py`): it instructs the model
   to *"always answer... even if you are unsure"* and never mentions the
   unsure option at all — it exists to catch blind guessing, not to be a
   safe default. Rewrote the prompt to ask for the best-supported inference
   from evidence, reserving abstention for genuinely unrelated evidence.
   This one change took accuracy from 0.0 to 0.44.
4. **A generous single-attempt LLM timeout is worse than a short one with a
   retry — but only up to a point.** 40% of a 25-question run hit
   `httpx.ReadTimeout` with no retry, silently scored as wrong+not-sure.
   Adding retry-with-backoff to `gemini_generate` looked obviously right,
   but the first attempt (2 retries, 120s each) measured *worse* — each
   slow attempt ate the entire per-question wall-clock budget by itself, so
   a retry never got to run before the outer timeout fired first. Fixed by
   shortening each attempt (45s) and cutting to 1 retry, so multiple real
   attempts fit inside the same overall budget instead of one very patient
   one. Reduced (didn't eliminate) timeout-driven failures.

**Honest final number** (`services/literature/scripts/run_litqa2_real.py 25 42`,
real questions, live retrieval, live Gemini answering — reproduced twice at
0.44 after the fixes above; two earlier attempts at 0.24 and 0.28 came from
timeout-tuning regressions documented above, not from retrieval/prompt
changes, and are kept in `artifacts/astabench_literature_scores.json` rather
than deleted):

| Metric | This service (n=25, live) | ReAct + Claude Opus 4.7 (official, n≈500+) |
|---|---|---|
| Accuracy | 0.44 | 0.84 |
| Coverage | 0.64–0.68 | 0.89 |
| Precision | 0.65–0.69 | 0.94 |

**This does not close the gap to the real leaderboard, and it is not
expected to with this architecture.** The honest structural reasons, not
excuses to wave away:

- **Retrieval infrastructure is fundamentally weaker.** The official harness
  gives agents `make_asta_mcp_tools` — AI2's proprietary, pre-built
  full-text snippet search index over a curated paper corpus. This service
  hits public PubMed/bioRxiv/Semantic Scholar/Crossref APIs directly with a
  single blind keyword search per question. No amount of prompt tuning
  closes a retrieval-infrastructure gap.
- **One search round vs. an agentic loop.** ReAct-style agents issue
  multiple tool calls, refining queries based on what the first round
  returns. This evaluator does one reformulate → one search → one answer,
  by design (it's testing this service's retrieval quality, not building a
  general research agent).
- **A "-preview" model tier.** `gemini-3-flash-preview` (matching the rest
  of Propab's config) showed persistent timeout instability under this
  workload (30-40% of calls even after tuning) that a more mature model
  tier likely wouldn't — this is model/infra reliability, not something
  fixable in this service's code.

Where real, fixable engineering gains are still on the table: multi-round
retrieval (search again using terms from the first round's results),
including the answer choices (not just the question) in query reformulation
(a real agent sees both), and a non-preview model tier for this eval path.
None of these were attempted here — see "Iterate to close the gap" in this
project's task history for what's still open.

## 0.2.0 — Gemini embeddings + AstaBench proxy tuned into range

**Why:** the rest of Propab already runs on Gemini embeddings in production
(`EMBED_PROVIDER=google` / `EMBED_MODEL=gemini-embedding-2` in the root
`.env`, consumed by `packages/propab-core/propab/embeddings.py`) — this
service defaulting to a weaker offline hashing embedding meant its semantic
retrieval/novelty logic was measurably worse than the rest of the system for
no good reason. Fixed:

- `indexer/embeddings.py`: added a `google`/`gemini` provider calling the
  same `v1beta/models/{model}:embedContent` endpoint, same request shape,
  same 429/503 retry-with-backoff as `propab.embeddings._google_embed` — so
  both call sites behave identically against the same API. `embed_provider`
  now defaults to `"google"`, `embed_model` to `"gemini-embedding-2"` (measured
  live: 3072-dim vectors), and a new `google_api_key` setting reads the
  **unprefixed** `GOOGLE_API_KEY`/`GEMINI_API_KEY` env vars (not
  `LITERATURE_GOOGLE_API_KEY`) so it shares one key with the rest of Propab.
  Falls back to the offline hashing embedding exactly as before when no key
  is configured — nothing about the zero-infrastructure dev/CI story changed.
- Per-text embedding calls (a `/prior` response can carry 50+ claims) now run
  concurrently under a semaphore (`max_concurrency=8`, default) instead of
  fully sequentially — this was a real latency problem waiting to happen once
  a real embedding API was in the default path, not just a nice-to-have.
- `evaluator/astabench.run_litqa2_proxy`: reworked the scoring from "average
  the top-5 retrieved claims into one evidence blob, embed that blob, compare
  options to the blob" to "retrieve top-10 claims, score each option by its
  *max* cosine similarity against any single retrieved claim." The blob
  approach diluted signal across unrelated claims; max-pooling rewards an
  option that closely restates one specific indexed claim, which is what
  actually separates a correct answer from a same-domain distractor.

**Measured (real Gemini embeddings, `services/literature/scripts/seed_artifacts.py`):**
citation-verification rate held at **100%** (40/40 on a broader
Sidon+cap-set+AP-free index — see `artifacts/literature_coverage.json`).
AstaBench LitQA2 proxy went **0.33 (1/3, offline embeddings) → 0.875 (7/8,
Gemini embeddings + max-pool scoring)** on an 8-case hand-built set that
deliberately includes hard near-miss distractors (swapped attribution, swapped
construction name, "gap already closed" false-closure claims), not just
easy/unrelated ones. Full progression, including two intermediate runs, is in
`artifacts/astabench_literature_scores.json` (append-only — nothing overwritten).

The one remaining miss ("who first studied the Sidon density question" —
picked Behrend 1946 over the correct Erdos-Turan 1941) is a genuine
attribution nuance: both are real, closely-related early papers in the same
history, and all three options scored within 0.03 of each other. Diagnosed by
checking the retrieved evidence directly (not by inspection of the metric),
and left as-is rather than hand-tuned away — hand-tuning to beat one's own
8-question eval is exactly the kind of overfitting `agent3.md` warns against
("Do not optimize for these benchmarks — they are measurement instruments").
Separately, an earlier run of the same case set caught a *real* retrieval gap
(cap-set question scored ~0.02 across all options — nothing relevant had been
retrieved at all) that traced back to the `/prior` call using a narrowly
Sidon-focused research question that under-indexed the other two seed-paper
topics; broadening that one question fixed it. Both are logged here because
they're the actual lesson: a narrow index-priming question silently starves
whatever topics it didn't cover, independent of embedding quality.

**Tests:** added `tests/test_embeddings.py` (provider selection, Gemini
request shape/headers via `httpx.MockTransport`, 429-retry, offline fallback
on persistent error) and `tests/test_astabench.py` (max-pool scoring
correctness, empty-index fallback, `record_score` append-not-overwrite).
71 tests total (up from 59), still fully offline/mocked.

## 0.1.0 — initial build

Full service skeleton stood up in one pass rather than the staged week-by-week
order in `agent3.md`, since the whole surface area was needed for the tests
to mean anything (a `/prior` endpoint that can't run `/novelty` against its
own index doesn't prove much).

**Sources:** arXiv (LaTeX e-print primary path with balanced-brace environment/
footnote/caption/bibliography/`\cite`-sentence extraction, PDF-text fallback via
`pypdf`, disk cache keyed by arxiv id), OEIS (search + cached-sequence tabulation
lookup), Semantic Scholar (citation graph, two-level citation-depth crawl),
MathOverflow/StackExchange (accepted-answer known-vs-open classification),
zbMATH, PubMed, bioRxiv, Crossref.

**Extractors:** claims (structural LaTeX environments + linguistic signal-phrase
scan over body/footnotes/captions + proof-body quantitative-sentence mining +
bibliography-annotation extraction), tables (LaTeX `tabular` + free-text
enumerated lists, appendix/supplementary flagging, row-count fidelity check),
open problems (explicit markers + conjecture environments + open-signal
sentences, computational-approachability heuristic), contradictions (numeric
bound-interval disjointness across embedding-similarity clusters), gaps
(proven-vs-conjectured bound pairing).

**Indexer:** offline hashing embedding (no API key/network needed) + OpenAI
embedding option; in-memory vector/structured stores (default) + Qdrant/Postgres
backends behind the same interface. (Google/Gemini embeddings became the
default in 0.2.0 — see below.)

**Retriever:** question decomposition, multi-source parallel search/fetch,
claim dedup by embedding similarity, the six-step novelty-check algorithm
(tabulation short-circuit → semantic search → scope match → implication
check → conservative verdict, biased toward "uncertain"), gap mapper ranked
by approachability/age/alignment.

**Evaluator:** citation-verification-rate spot-check (re-fetches sources,
confirms verbatim substring match — the primary health metric), domain-eval
runner (novelty-verdict accuracy against a hand-built test set), AstaBench
LitQA2-shaped proxy runner (documented as a stand-in for the official
`inspect_ai` harness, which needs gated dataset access this environment
doesn't have — see `README.md`).

**Tests:** 59 tests, all offline (`httpx.MockTransport` for HTTP, in-memory/
offline backends for storage/embeddings).

**Live validation (2026-07-05):** ran `/prior` end-to-end against real arXiv,
OEIS, Semantic Scholar, MathOverflow, and zbMATH for math_combinatorics
(`services/literature/scripts/smoke_run.py`) — 16 papers indexed, 32
established facts, 5 open gaps, 3 tabulated sequences. This surfaced and
fixed two real citation-fidelity bugs the offline test suite couldn't catch:

1. `sources/oeis.py` assumed the OEIS search API returns `{"results": [...]}`;
   it actually returns a bare JSON list on success or literal `null` on no
   match. Fixed with `_oeis_results()`; regression tests added.
2. Bibliography-annotation claims (`\cite{...}` sentences) were rewriting the
   citation marker to a synthetic `"[cite]"` placeholder in the stored
   `verbatim` — which broke the verbatim requirement outright (the "quote"
   didn't exist in the source anymore) and, separately, `claims.py`'s and
   `_latex.py`'s sentence-splitters could run a "claim" across a paragraph
   break (e.g. straight through this service's own injected
   `"Answer (score N):"` marker between a MathOverflow question and its
   answer), producing claims that spliced unrelated text together. Fixed by
   (a) keeping the raw `\cite{...}` markup in both the stored verbatim and
   `body_text` instead of stripping/replacing it, and (b) scoping sentence
   extraction to one paragraph at a time in both `claims.py` and `_latex.py`.

Citation verification rate after the fix, re-measured live against the same
30-claim sample (`evaluator/metrics.citation_verification_rate`):
**30/30 verifiable, 30/30 verified → 100%** (see
`artifacts/literature_coverage.json`). This is one run on one domain, not a
statistically robust estimate — the number that matters is that the fix
closed a real gap between "looks right" and "is right."

AstaBench LitQA2 proxy (`evaluator/astabench.run_litqa2_proxy`), 3 hand-built
cases against the same math_combinatorics index: 1/3 correct (33%) — the
offline hashing embedding is a weak semantic signal, exactly as expected;
this is a real, honest first data point in `artifacts/astabench_literature_scores.json`,
not a benchmark result to be proud of. Swapping in real (OpenAI) embeddings
via `LITERATURE_EMBED_PROVIDER=openai` should improve this substantially —
untested here (no API key in this environment).

Gap mapper run live for two domains: math_combinatorics found 200+ candidate
open-problem statements from arXiv/MathOverflow searches
(`artifacts/domain_gap_maps/math_combinatorics.json`); network_diffusion
found 0 in the same pass (`artifacts/domain_gap_maps/network_diffusion.json`)
— physics-style papers apparently don't phrase results with the
"Problem:"/"Open problem:" markers `open_problems.py` looks for as often as
math papers do. Logged here rather than silently accepted: the fix, if this
domain becomes a priority, is a signal-phrase set tuned for physics/network
science phrasing, not more math-specific markers.

**Domain plugins:** `literature_profile()` implemented for all seven —
math_combinatorics, genomics, mandrake, materials, enzyme_kinetics,
graph_invariants, network_diffusion. Seed papers and OEIS tabulation ids
were verified against live Crossref/arXiv/OEIS lookups during this build,
not guessed (see each plugin's `literature_profile()` for the exact DOIs/
arXiv ids/OEIS sequence numbers used).

**Integration:** `DomainPlugin.literature_profile()` added to
`packages/propab-core/propab/domain_modules/base.py` (default: empty profile,
falls back to keyword search) and `literature_service_url` added to
`propab.config.Settings` — the two changes to existing code permitted by this
service's ownership contract. Wiring `campaign_loop.py`/`verdict_pipeline.py`
to actually call this service is out of scope here (owned by
`services/orchestrator/`, currently mid-flight from other work) — the
`/prior` and `/novelty` contracts are ready for that integration whenever it
happens.
