# Adding a domain to Propab

This guide walks through adding a new scientific domain plugin from scratch.
The **genomics** plugin (`packages/propab-core/propab/domain_modules/genomics/`)
is the worked example throughout — it was built to prove the interface without
touching campaign logic in core.

After following this guide you should be able to launch a dry-run campaign,
pass CI preflight, and keep hypothesis routing at zero mismatches for your corpus.

---

## What you are building

A domain is a self-contained package under:

```text
packages/propab-core/propab/domain_modules/<your_domain>/
  __init__.py
  plugin.py          # DomainPlugin subclass (required)
  adapter.py         # data loading / feature computation (typical)
  verifier.py        # run_verification + classify_verdict (typical)
  problems.py          # literature prior, seed hypotheses (optional)
  routing_inspector.py # offline routing corpus (required for CI)
  tests/               # plugin-local unit tests (recommended)
```

Core Propab resolves domains only through `DomainPlugin` methods and the
registry. It must not import your dataset, feature names, or thresholds directly.

---

## Step 0 — Pick an id and register

1. Choose a stable `domain_id` (lowercase, underscores), e.g. `genomics`.
2. Register the plugin in `domain_modules/registry.py` inside `_ensure_loaded()`:

```python
from propab.domain_modules.genomics.plugin import GenomicsPlugin
# ...
register_plugin(GenomicsPlugin())
```

This is the **only** intentional core change when adding a domain. Everything
else lives under your domain directory.

3. Add a top-level test file so CI discovers your plugin, e.g.
   `tests/test_genomics_plugin.py` (imports from your package, runs preflight
   and routing corpus).

4. If your corpus should run in the global routing gate, wire it in
   `scripts/inspect_hypothesis_routing.py` (genomics is merged alongside
   math combinatorics artifacts).

---

## Step 1 — Implement `DomainPlugin`

The ABC lives in `propab/domain_modules/base.py`. Subclass it in `plugin.py`.

### Identity attributes

| Attribute | Purpose | Genomics example |
|-----------|---------|------------------|
| `domain_id` | Registry key | `"genomics"` |
| `display_name` | Human label | `"Genomics (GTEx cross-tissue expression)"` |
| `version` | Semver string | `"1.0"` |
| `scope_question_markers` | Looser keywords for scope templates | `"gene expression"`, `"gtex"`, … |
| `artifact_question_markers` | Vocabulary for artifact-gate routing | `"housekeeping"`, `"tau index"`, … |

### `matches(question, payload) -> bool`

Return whether this plugin owns a campaign. Explicit signals (`domain`,
`domain_profile` on the payload, or `[domain_profile:genomics]` in the question)
are handled by the registry **before** `matches` is called.

Genomics accepts explicit profile, or ≥2 scope markers in the question:

```python
def matches(self, *, question: str = "", payload: dict | None = None) -> bool:
    if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "genomics":
        return True
    q = (question or "").lower()
    hits = sum(1 for m in self.scope_question_markers if m in q)
    return hits >= 2 or "[domain_profile:genomics]" in q
```

**Tip:** Keep detection heuristics in the plugin, not in orchestrator or worker code.

### `scope_template() -> dict[str, str] | None`

Default OOD-scope fields when the LLM omits them. Genomics returns population,
holdout design, and expected failure modes:

```python
def scope_template(self) -> dict[str, str]:
    return {
        "population": "GTEx v8 subset: 1000 variable genes × 10 tissues",
        "distribution": "Leave-one-tissue-out holdout across tissue types",
        "claimed_generalization": "Expression pattern survives held-out tissue",
        "expected_failure_modes": "Tissue-label leakage; housekeeping-only tautologies",
        "ood_test": "Leave-tissue-out LOFO + tissue-label shuffle null p<0.05",
    }
```

### `available_features() -> list[str]`

**Required.** Names the worker may select for verification. Genomics exposes
gene-level features computed in `adapter.py`:

```python
KNOWN_FEATURES = (
    "expression_variance",
    "tissue_specificity_tau",
    "mean_expression",
    "cv_across_tissues",
)

def available_features(self) -> list[str]:
    return list(KNOWN_FEATURES)
```

### `run_verification(hypothesis, evidence, features) -> dict`

Run the domain experiment and return an **evidence dict** the verdict pipeline
can interpret. Delegate to `verifier.py`:

```python
def run_verification(self, hypothesis, evidence=None, features=None):
    spec = GenomicsExperimentSpec.from_hypothesis(hypothesis)
    if features:
        spec = GenomicsExperimentSpec(feature_subset=list(features), target_column=spec.target_column)
    return run_genomics_experiment(spec)
```

### `classify_verdict(hypothesis_text, result) -> (verdict, rationale, confidence)`

Map evidence to `confirmed` / `refuted` / `inconclusive`. Genomics requires
LOFO R², tissue-label shuffle p-value, and `verified_true_steps`:

```python
def classify_verdict(self, hypothesis_text, result):
    lofo = float(result.get("lofo_r2") or 0.0)
    p = float(result.get("label_shuffle_null_p") or 1.0)
    steps = int(result.get("verified_true_steps") or 0)
    if steps >= 1 and lofo >= 0.15 and p < 0.05:
        return "confirmed", f"LOFO R²={lofo:.3f}, tissue-shuffle p={p:.3f}", 0.88
    # ...
```

### `confirmation_criteria() -> dict`

Tell core what “confirmed” means for your domain (holdout type, null test, min steps):

```python
def confirmation_criteria(self) -> dict:
    return {
        "min_metric_steps_for_confirm": 2,
        "min_confidence": 0.85,
        "requires_holdout": True,
        "holdout_type": "leave_tissue_out",
        "null_test": "tissue_label_shuffle",
        "verification_type": "statistical",
    }
```

### `preflight() -> PreflightResult`

Check dataset access, feature computability, and latency **before** a campaign
starts. Return `PreflightResult(passed=False, reason=...)` on failure.

See [Writing preflight](#step-3--writing-preflight) below.

### Optional but useful overrides

| Method | When to override |
|--------|------------------|
| `literature_prior(question)` | Seed facts / dead ends for campaign start |
| `implementable_methodologies()` | Filter LLM methodologies to implemented paths |
| `hypothesis_on_topic(text, methodology)` | Reject cross-domain leakage (e.g. Sidon sets in genomics) |
| `extract_numerical_seeds(confirmed_nodes)` | Feed next-campaign numerical seeds into lifetime store |
| `belief_promotion_threshold()` | Domain-specific belief promotion rules |
| `domain_profile()` | Link to artifact-gate `DomainProfile` if you use artifact models |

Defaults in `base.py` are safe no-ops; override only what you need.

---

## Step 2 — Writing a verifier

Put experiment logic in `verifier.py`, not in `plugin.py`. The evidence dict
must use **stable field names** that core and replay tooling already understand
where possible.

### Recommended evidence fields

| Field | Meaning | Used by |
|-------|---------|---------|
| `verified_true_steps` | Count of substantive verification steps that passed | Verdict pipeline, confirmation criteria |
| `verified_false_steps` | Steps that falsified the claim | Replay / audit |
| `verification_method` | Short id, e.g. `leave_tissue_out` | Logging, health metrics |
| `metric_name` | Canonical metric id (for routing inspector) | Routing corpus |
| `metric_value` | Scalar result | Claim parsers, artifacts |
| `discovery_worthy` | False for trivial rediscoveries | Synthesis filtering |
| `trivial_rediscovery` | True when result is known baseline | Belief promotion |
| `notes` | Human-readable summary | Forensics |

Statistical domains often also return:

- `lofo_r2`, `label_shuffle_null_p`, `label_shuffle_null_p95` (genomics)
- `n_genes`, `n_features`, `feature_subset` (provenance)

Math combinatorics returns `metric_name` like `sidon_ratio_to_sqrt_n` or
`ap_free_density_sweep` plus sweep tables in nested structures.

### Genomics verifier sketch

```python
def run_genomics_experiment(spec: GenomicsExperimentSpec) -> dict:
    # load frame → build X, y, tissue labels
    lofo_r2, nulls = _tissue_label_shuffle_null(X, y, tissues)
    return {
        "lofo_r2": lofo_r2,
        "label_shuffle_null_p95": float(np.percentile(nulls, 95)),
        "label_shuffle_null_p": float(np.mean([1 for n in nulls if n >= lofo_r2])),
        "verification_method": "leave_tissue_out",
        "verified_true_steps": 1 if lofo_r2 > 0 and label_shuffle_null_p < 0.05 else 0,
        # ...
    }
```

### Lazy imports

Import heavy dependencies (`pandas`, `sklearn`, `pymatgen`, …) **inside**
methods, not at module top level in `plugin.py`. Registration must stay cheap so
one broken domain does not break others.

---

## Step 3 — Writing preflight

Preflight answers: “Can we run a representative experiment in reasonable time
with the data available?”

Genomics preflight:

1. Imports numpy/scipy (declares dependency surface).
2. Loads the GTEx subset via `GenomicsAdapter`.
3. Runs one LOFO experiment with two features.
4. Fails if elapsed time exceeds 60 seconds or load raises.

```python
def preflight(self) -> PreflightResult:
    try:
        t0 = time.time()
        adapter = GenomicsAdapter()
        df = adapter.load_frame()
        run_genomics_experiment(GenomicsExperimentSpec(feature_subset=["expression_variance", "mean_expression"]))
        elapsed = time.time() - t0
        if elapsed > 60:
            return PreflightResult(False, f"LOFO too slow: {elapsed:.1f}s", {"elapsed_sec": elapsed})
        return PreflightResult(True, "GTEx subset loaded, LOFO preflight ok", {...})
    except Exception as exc:
        return PreflightResult(False, f"genomics preflight failed: {exc}")
```

Return `details` with counts (genes, tissues, elapsed seconds) — they appear in
dry-run reports and debugging.

---

## Step 4 — Building the routing corpus

The routing inspector checks that **hypothesis text → verifier → metric** stays
consistent without launching a full campaign. Each domain should ship:

- `routing_inspector.py` with `inspect_routing()`, `inspect_corpus()`, and
  `ROUTING_CORPUS: list[dict]`.
- Corpus entries as dicts with at least `statement` / `text` and
  `test_methodology`.

### How many entries? What variety?

| Guideline | Rationale |
|-----------|-----------|
| **≥20 entries** for a new domain | CI genomics test asserts `total >= 20` |
| Cover each verifier branch | e.g. single-point vs sweep, each feature family |
| Include negative / edge phrasing | Methodology keywords that must not mis-route |
| Include off-domain wording in inspector tests | Ensure `hypothesis_on_topic` rejects leakage |

Genomics corpus (`g01`–`g22`) spans housekeeping vs tissue-specific claims,
different feature subsets, LOFO vs shuffle-null wording, and tau/CV targets.

### `inspect_routing` pattern

```python
def inspect_routing(hypothesis, *, dry_run_experiment=False):
    features = _infer_features(hypothesis)
    invalid = [f for f in features if f not in KNOWN_FEATURES]
    routing_ok = not invalid and bool(features)
    out = {
        "domain": "genomics",
        "resolved_verifier": "genomics_lofo",
        "resolved_features": features,
        "expected_metric_name": "lofo_r2",
        "actual_metric_name": "lofo_r2",
        "routing_ok": routing_ok,
    }
    if dry_run_experiment:
        result = run_genomics_experiment(...)
        out["routing_ok"] = routing_ok and "lofo_r2" in result
    return out
```

For math-style domains, also set `metric_match` by comparing
`expected_metric_name` to the dry-run experiment’s `metric_name`.

Wire your corpus into `scripts/inspect_hypothesis_routing.py` so
`--corpus --require-zero-mismatches` includes your entries.

---

## Step 5 — Validation checklist

Run these before opening a PR. CI runs the same gates on every push to `main`.

### 1. Unit tests

```bash
pytest tests/test_<your_domain>_plugin.py -q
pytest tests/ -q --tb=short
```

Test count must not decrease.

### 2. Domain preflight

```bash
python -c "
from propab.domain_modules.<your_domain>.plugin import <YourPlugin>
r = <YourPlugin>().preflight()
assert r.passed, r.reason
print('ok:', r.details)
"
```

Or rely on CI’s loop over `all_plugins()` (stubs `enzyme_kinetics` and
`graph_invariants` are skipped).

### 3. Routing corpus (zero mismatches)

```bash
python scripts/inspect_hypothesis_routing.py --corpus --require-zero-mismatches
```

Exit code must be `0`. Fix mismatches in your domain’s `routing_inspector.py`
or verifier — not by special-casing core.

### 4. Campaign dry-run

```bash
python scripts/start_v1_frontier_campaign.py --domain genomics --dry-run
```

Expect JSON with `preflight_passed: true` and routing checks passing.

### 5. The falsifiability test

After your branch is complete:

```bash
git diff --name-only origin/main | grep -v domain_modules/<your_domain> | grep -v registry.py | grep -v tests/test_
```

**Goal:** no changes outside your domain folder except:

- `domain_modules/registry.py` (one import + register line)
- `scripts/inspect_hypothesis_routing.py` (corpus merge)
- `tests/test_<your_domain>_plugin.py` (CI discovery)
- docs / backlog updates

If you need to patch orchestrator, worker, or belief code for your domain, the
`DomainPlugin` interface is missing something — extend the ABC with a default
implementation instead of hardcoding domain logic in core.

---

## Common mistakes

### Domain coupling in core

**Wrong:** adding `if "gtex" in question` in `campaign_synthesis.py`.  
**Right:** implement `matches()` and `hypothesis_on_topic()` on the plugin.

### Vague claim types

Hypotheses like “gene X is important” without population, holdout, or metric
fail scope gates and produce inconclusive verdicts. Use `scope_template()` and
require LOFO / null-test language in methodologies.

### Methodology keywords mis-routing experiments

Math combinatorics learned this the hard way: methodology containing
`upper bound` must not trigger asymptotic sweep routing for single-`n` AP-free
claims. Base routing decisions on the **statement**, not concatenated
methodology text, when keywords overlap.

### Missing `verified_true_steps`

Core confirmation logic counts metric steps. Always set `verified_true_steps` (and
optionally `verified_false_steps`) in evidence, even for statistical domains.

### Heavy imports at plugin import time

**Wrong:** `import pymatgen` at top of `plugin.py`.  
**Right:** import inside `run_verification` / `preflight`.

### No routing corpus

Without offline routing tests, verifier refactors silently break hypothesis
classification until a full campaign run — too late and too expensive.

### Skipping preflight

Campaigns that fail mid-loop after consuming budget are costly. Preflight should
catch missing data files, slow paths, and broken feature columns.

---

## Genomics file map (quick reference)

| File | Role |
|------|------|
| `plugin.py` | `DomainPlugin` wiring |
| `adapter.py` | Synthetic GTEx subset, feature computation, cache path |
| `verifier.py` | LOFO ridge, tissue-label shuffle null, verdict classifier |
| `problems.py` | Literature prior for campaign seeding |
| `routing_inspector.py` | 22-entry corpus, feature inference |
| `tests/test_plugin.py` | Local unit tests |
| `tests/test_genomics_plugin.py` | CI entry point at repo root |

---

## Checklist summary

- [ ] Create `domain_modules/<id>/` with `plugin.py`, verifier, adapter
- [ ] Register in `registry.py`
- [ ] Implement `available_features`, `run_verification`, `preflight`
- [ ] Add `routing_inspector.py` with ≥20 corpus entries
- [ ] Add `tests/test_<id>_plugin.py`
- [ ] Wire corpus into `inspect_hypothesis_routing.py`
- [ ] `pytest tests/ -q` passes
- [ ] `--corpus --require-zero-mismatches` exits 0
- [ ] `start_v1_frontier_campaign.py --domain <id> --dry-run` passes
- [ ] Falsifiability test: no unintended core diffs

When all boxes are checked, your domain is ready for frontier campaigns and
external contributors can follow the same path without reading Propab internals.
