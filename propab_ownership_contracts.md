# Propab — Component Ownership Contracts

This document defines ownership, responsibility boundaries, and health
metrics for every major Propab component. It exists for one reason: so
that when something goes wrong, you can say "verification artifact
rejection rate dropped from 94% to 60%" instead of "Propab isn't working."

Every component has exactly one owner (the thing it is responsible for),
one thing it explicitly never owns (the boundary it must not cross), one
input contract (what it requires to function), one output contract (what
it guarantees to produce), and one health metric (the number that tells
you whether this component is doing its job).

When you add a new component, write its entry in this document before
writing its code. The contract defines the code, not the other way around.

---

## The rule this document enforces

If debugging a problem requires looking at more than one component's
health metric to find the root cause, either the contracts are wrong or
the component boundaries are wrong. Fix the contracts, not the debugging
process.

---

## Component contracts

---

### Literature Layer

**Owns:** structured knowledge about what is known, contested, and unknown
in a research domain, sourced from published papers.

**Never owns:** hypotheses, verification of claims, campaign state.

**Input contract:**
- A research question (string)
- A domain identifier (string)
- Optional: a date range for recency filtering

**Output contract:**
- A structured prior object containing:
  - `established_facts`: list of attributed claims with paper citations
  - `open_gaps`: list of explicitly unresolved questions from the literature
  - `contradictions`: list of pairs of conflicting claims with citations
  - `dead_ends`: list of approaches the literature has tried and ruled out
- Every claim in established_facts and contradictions must have at least
  one real, verifiable citation (not inferred, not hallucinated)
- Citation format must include enough information to retrieve the paper
  (title + authors + year minimum, DOI preferred)

**Health metric:** Citation verification rate — the fraction of claims in
`established_facts` whose cited paper actually contains the attributed
claim, spot-checked by retrieving the abstract and confirming relevance.
Target: ≥ 90%. If this drops below 80%, the literature layer is
fabricating or misattributing claims and its output cannot be trusted as
a campaign prior.

**What a drop in this metric means:** The LLM extracting claims from
abstracts is hallucinating citations or attributing claims to the wrong
paper. Fix: tighten the extraction prompt or add a citation spot-check
pass before returning output.

---

### Hypothesis Generator

**Owns:** producing candidate hypotheses for a campaign to test, conditioned
on what is already known and what has already been tried.

**Never owns:** testing hypotheses, verifying claims, updating beliefs,
deciding which hypothesis to test next (that is the Campaign Manager's job).

**Input contract:**
- The research question (string)
- The current belief state (active beliefs, closed beliefs)
- The literature prior (output of Literature Layer)
- Dead ends from the current campaign (list of failed hypothesis texts)

**Output contract:**
- A list of candidate hypotheses, each with:
  - `statement`: the specific claim, stated precisely enough to be testable
  - `test_methodology`: what test would falsify this claim
  - `population`: which dataset/domain/context this claim applies to
  - `claimed_generalization`: what broader class this claim belongs to
  - `ood_test`: what held-out or cross-group test would validate transfer
  - `expected_failure_mode`: the most likely reason this hypothesis fails
- Duplicate rate: the fraction of new candidates with ≥ 85% text similarity
  to any candidate already in the campaign tree must be < 10% per batch

**Health metric:** Duplicate rate per synthesis round. Target: < 10%.
If this exceeds 20%, the generator is recycling — producing minor
rewrites of the same idea rather than genuinely new candidates. This
is the most reliable early signal that the campaign has exhausted its
current search direction and needs either a new direction (from synthesis)
or a belief state update.

**What a drop in this metric means (duplicate rate rising):** Synthesis
is not providing enough variety in the belief state to drive diverse
generation. Either beliefs are too narrow (only one strong belief,
no rivals) or the generator is not reading the belief state correctly.
Check whether the active belief set has at least two distinct rivals.

---

### Campaign Synthesis

**Owns:** updating beliefs based on completed experiment results, deciding
the current campaign direction, identifying when a direction is exhausted.

**Never owns:** running experiments, verifying claims, generating the
detailed text of hypotheses.

**Input contract:**
- All completed nodes since the last synthesis pass, each with:
  - The hypothesis that was tested
  - The full evidence blob
  - The verdict (confirmed / refuted / inconclusive) and verdict reason
- The current active belief state (≤ 3 active BeliefObjects)
- The closed beliefs record
- The pinned campaign context (research question, human messages)
- The previous synthesis pass's critical experiment and its result

**Output contract:**
- An updated belief state with:
  - Each active belief's confidence updated (strong / weak / unclear)
  - Each active belief's supporting_nodes and contradicting_nodes updated
    to include only nodes whose evidence actually bears on that belief's
    specific claim (Evidence Binding — see below)
  - Any belief that has been weakened past the abandonment threshold
    moved to closed beliefs with an explicit reason
  - At most 3 active beliefs at any time
- A named critical experiment for this round: the single test most likely
  to discriminate between the current active beliefs
- An exhaustion judgment: whether this direction appears exhausted
  (only meaningful after 3+ synthesis rounds on the same direction)

**Health metric:** Belief stability — the fraction of beliefs that
persist with the same statement across consecutive synthesis rounds
without being silently rewritten. Target: a belief's core claim should
not change by more than paraphrase between rounds; if a belief's statement
changes substantially, that should be recorded as belief closure plus
new belief creation, not silent mutation.

Companion metric: citation integrity — the fraction of nodes in
`supporting_nodes` and `contradicting_nodes` that actually bear on the
belief's stated claim (verified by checking claim_type and scope overlap).
This was the 94.5% fabrication problem. Target: ≥ 95% of citations are
relevant. If this drops below 85%, Evidence Binding is not working.

**What a drop in citation integrity means:** Synthesis is attaching
convenient nodes to beliefs rather than relevant ones. The Evidence
Binding check at write time is failing or has been bypassed. Check whether
`evidence_binding_audit` is being called at belief update time.

---

### Worker / Experiment Executor

**Owns:** executing one hypothesis test — running the code, calling the
tools, collecting raw evidence, returning a verdict.

**Never owns:** deciding what to test, updating beliefs, orchestrating
the campaign, deciding whether a result is scientifically significant
at the campaign level.

**Input contract:**
- One hypothesis with its full scope fields populated
- The domain plugin to use for verification
- A compute budget (max wall-clock time, max tool calls)

**Output contract:**
- A raw evidence blob containing everything the domain plugin's verifier
  produced: metric values, statistical test results, LOFO statistics
  (if applicable), verification method, number of verified_true and
  verified_false steps
- A verdict from `run_verdict_pipeline`: confirmed / refuted / inconclusive
- A verdict reason: the specific reason for the verdict (not a generic label)
- All of the above within the compute budget — a timed-out worker must
  return inconclusive with reason "timeout", not hang

**Health metric:** Experiment success rate — the fraction of dispatched
hypotheses that return a definitive verdict (confirmed or refuted) rather
than inconclusive due to execution failure (timeout, tool error, sandbox
crash). Target: ≥ 70% of hypotheses should reach a definitive verdict.
If this drops below 50%, experiments are failing before producing any
evidence, and the campaign is burning budget without learning anything.

Note: inconclusive because the evidence genuinely doesn't support a clear
conclusion is expected and correct. Inconclusive because of tool failure,
timeout, or code error is what this metric tracks — these are different
failure modes and should be logged separately.

**What a drop in this metric means:** Tool errors or sandbox timeouts are
dominating. Check the worker error logs for the most common failure reasons.
The most likely causes: domain plugin not correctly wired (tool calls failing),
sandbox network issues, compute budget too tight for the hypothesis complexity.

---

### Verification Pipeline

**Owns:** taking raw evidence from a worker and determining whether it
supports the hypothesis at the level required for confirmation.

**Never owns:** generating evidence (that is the worker's job), deciding
what to test (that is the campaign manager's job), updating beliefs (that
is synthesis's job).

**Input contract:**
- A raw evidence blob from the worker
- The domain plugin's confirmation criteria for this campaign
- The hypothesis's stated claim type (deterministic / statistical / lofo)

**Output contract:**
- A verdict: confirmed / refuted / inconclusive
- A confidence: float 0.0–1.0
- A verdict reason: specific, human-readable, naming which gate decided
  the verdict and why
- The evidence type that was detected: deterministic / lofo / statistical /
  unknown (so downstream components can interpret the verdict correctly)

**Health metric:** Artifact gate precision — the fraction of "confirmed"
verdicts that survive when independently audited by the same label-shuffle
permutation test used in the Mandrake and materials campaigns. Target:
≥ 90% of confirmed verdicts survive independent audit. If this drops,
the artifact gate is passing findings that don't generalize.

This metric requires periodic sampling — you cannot run a full permutation
audit on every hypothesis in real time. Run it at campaign end on all
confirmed findings from that campaign, the same way the falsification
map audits were run. Log the result as a campaign-level metric.

**What a drop in this metric means:** The artifact gate is too permissive
for the current domain. Either the domain's confirmation criteria are
miscalibrated, or the domain's artifact models are missing the dominant
failure mode. Run the dead-finding diagnosis bench on the failed audits
to identify which artifact type is being missed.

---

### Campaign Manager / Orchestrator

**Owns:** scheduling which hypothesis to test next, managing the worker
pool, deciding when a campaign is done, managing the campaign lifecycle.

**Never owns:** scientific reasoning about what the results mean (that
is synthesis's job), running experiments (that is the worker's job),
generating hypothesis text (that is the generator's job).

**Input contract:**
- A campaign configuration: research question, domain, budget, stopping rules
- A stream of completed worker results
- Synthesis output: updated beliefs, new frontier candidates, exhaustion signal

**Output contract:**
- A continuously populated frontier of candidates ready to dispatch
- Worker utilization: no worker should be idle when the frontier has
  candidates and budget remains
- A campaign stop event with an explicit, enum-valued stop reason when
  any stopping condition is met
- A resumable campaign state persisted to Postgres after every synthesis
  round — the campaign must be resumable after any service restart

**Health metric:** Worker utilization — the fraction of time workers are
actively running experiments versus idle waiting for the frontier to refill.
Target: ≥ 80% utilization during active campaign phases. Drops below 60%
indicate synthesis is not producing candidates fast enough to keep workers
busy, or frontier deduplication is rejecting too many candidates.

Companion metric: stop reason accuracy — every campaign stop event must
have a non-null, meaningful stop reason. If more than 5% of campaigns
stop with a generic or null reason, the lifecycle management is broken.

**What a drop in worker utilization means:** Either synthesis trigger
frequency is too low (workers finish before the next synthesis pass runs),
or duplicate rejection is too aggressive (synthesis produces candidates
but they're all rejected as duplicates). Check the synthesis trigger
multiplier and the duplicate rejection rate in the same time window.

---

### Evidence Binding

**Owns:** ensuring that any citation — any node listed as supporting
or contradicting a belief, any evidence attributed to a mechanism —
actually bears on the specific claim being cited.

**Never owns:** deciding what the evidence means, generating beliefs,
running experiments.

**Input contract:**
- A citing object: a belief, mechanism, or diagnosis with a list of
  node references in a `supporting_nodes` or `contradicting_nodes` field
- The cited nodes: the actual hypothesis text, evidence blob, and
  claim_type of each node being cited

**Output contract:**
- A binding verdict for each citation: bound (relevant) / unbound (irrelevant)
- A binding rejection count for the operation: how many citations were
  rejected
- For rejected citations: the reason (claim type mismatch, scope mismatch,
  verdict direction mismatch)
- Citations that fail binding are removed before the citing object is
  persisted. An empty, honest citing object is acceptable. A populated,
  false one is not.

**Health metric:** Binding rejection rate — the fraction of proposed
citations that fail the relevance check. Target: < 5% in a well-functioning
campaign. If this rises above 15%, synthesis is proposing many irrelevant
citations, which indicates either the belief statements are too vague
(they can plausibly be "supported" by evidence about unrelated topics)
or synthesis is not reading belief statements carefully enough.

Note: a rejection rate of exactly 0% over many rounds is also suspicious —
it may mean the binding check is not actually running. Log a warning if
binding is called 50+ times with zero rejections.

**What a drop in this metric (rate rising) means:** Synthesis is attaching
nodes to beliefs based on superficial relevance (same domain, same
direction of verdict) rather than genuine claim overlap. Tighten the
subject-extraction logic in the binding check or make belief statements
more specific so irrelevant nodes cannot appear relevant.

---

## How to use this document

**When adding a new component:** write its contract entry here before
writing any code. If you cannot write the entry — if you do not know
what it owns, what it never owns, its input/output contracts, and its
health metric — you do not yet understand what the component should do.
Clarify first.

**When debugging a problem:** start with the health metrics. Identify
which metric is out of range. That component is the root cause. Do not
look at other components until this one's metric is explained.

**When a metric is out of range:** do not look at metrics across multiple
components first. Fix the component whose metric is failing. If fixing
it requires changing another component's behavior, that is a contract
violation — the contracts need updating, not the components.

**When reviewing a PR:** check whether the PR changes behavior that
crosses a component's "never owns" line. If it does, it is the wrong
fix in the wrong place. The fix should be in the component that owns
the behavior being changed.

---

## Health metric tracking

These metrics should be logged to Postgres at the appropriate granularity:

| Metric | When to log | Table |
|--------|-------------|-------|
| Literature citation verification rate | Per literature prior build | campaign_literature_priors |
| Hypothesis duplicate rate | Per synthesis round | campaign_synthesis_events |
| Belief citation integrity | Per synthesis round | campaign_synthesis_events |
| Belief stability | Per synthesis round | campaign_synthesis_events |
| Worker experiment success rate | Per campaign | research_campaigns |
| Verification artifact gate precision | Per campaign (post-audit) | campaign_audit_results |
| Worker utilization | Per campaign | research_campaigns |
| Evidence binding rejection rate | Per synthesis round | campaign_synthesis_events |

A campaign dashboard should display the current value of each metric
for any running campaign, and flag any metric that is out of its target
range. This replaces "Propab failed" with "which component's number is
wrong" as the first question anyone asks when something goes wrong.