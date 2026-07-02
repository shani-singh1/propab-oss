# V1 Literature Prior (side experiment)

Standalone literature layer for validating whether a structured prior changes what Propab proposes on day one. **Not wired into core services** — delete `scripts/v1_literature_prior/` and the three entry scripts to remove entirely.

## What it does

Input: research question + domain  
Output: structured JSON with:

1. **Established facts** — specific attributed claims (not summaries)
2. **Contested claims** — where papers contradict each other
3. **Open gaps** — questions literature flags as unresolved

Uses arXiv + Semantic Scholar APIs only. Optional LLM extraction via `OPENAI_API_KEY` (falls back to abstract heuristics without it).

## Materials domain (running campaign)

```bash
# Build prior (~2–5 min depending on API rate limits)
python scripts/build_v1_literature_prior.py --domain materials

# Also emit Propab-shaped prior_json for campaign 2
python scripts/build_v1_literature_prior.py --domain materials \
  --campaign-prior-out artifacts/v1_literature_prior/materials_prior_for_campaign.json
```

## A/B test workflow (fixes.md)

| Step | Command |
|------|---------|
| Campaign 1 (no prior) | Running now — `d7afdf9c…` |
| Snapshot baseline hypotheses | `python scripts/snapshot_v1_campaign_hypotheses.py --campaign-id d7afdf9c-...` |
| Build literature prior | `python scripts/build_v1_literature_prior.py --domain materials` |
| Campaign 2 (with prior) | Launch after campaign 1; inject prior (see below) |
| Compare | `python scripts/compare_v1_literature_campaigns.py --baseline-snapshot ... --with-prior-snapshot ...` |

## Injecting prior into campaign 2

The core campaign loop currently rebuilds literature via `build_prior()` on fresh starts. For the V1 test without core changes:

1. Build `--campaign-prior-out` artifact
2. After launching campaign 2, write `prior_json` to Postgres **before** the built-in prior phase completes, **or**
3. Add a one-line optional `literature_prior` field to `CampaignRequest` when ready to productize

Until injection is wired, the prior artifact still supports manual review: compare its claims to hypotheses proposed in campaign 1.

## Files

| Path | Role |
|------|------|
| `domains.py` | Per-domain search queries + extraction focus |
| `fetch.py` | arXiv + Semantic Scholar fetch |
| `extract.py` | LLM / heuristic claim extraction |
| `build.py` | Orchestration |
| `compare.py` | Hypothesis A/B comparison |
