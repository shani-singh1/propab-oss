# Campaign Era Partitioning

Separate experience by architecture era. Older campaigns must not dominate newer search behavior.

## Eras

| Era | Label | Description |
|-----|-------|-------------|
| 0 | `pre_fix_contaminated` | Pre-literature fixes; contaminated priors; ML drift |
| 1 | `literature_claim_typing` | Literature repaired; claim typing; frontier fixes |
| 2 | `replay_simulator` | Replay + simulator calibration |
| 3 | `operator_credit` | Operator credit assignment |
| 4 | `db_backed_traces` | DB-backed traces; current architecture |

## Components (P0–P6)

- **P0** `CampaignEra` — era definitions with architecture features
- **P1** `CampaignEraMetadata` — commit, simulator version, trace source, policy generation
- **P2** `trust_weight(campaign)` — recent eras dominate; old eras decay
- **P3** Era-local statistics — `P(success|operator,state)` per era
- **P4** Cross-era comparison — operator ranking stability
- **P5** Gold corpus — latest-architecture demo/training campaigns only
- **P6** Experience archive — historical campaigns excluded from policy training

## Usage

```bash
docker compose up -d postgres migrate

# Partition 55 campaigns + select gold corpus
python scripts/partition_campaign_eras.py --with-credits

# Operator credit uses gold corpus for priors (default)
python scripts/run_operator_credit.py --all-db
```

## Outputs

| File | Contents |
|------|----------|
| `data/lifetime_knowledge/campaign_eras.json` | Full partition + era stats + cross-era comparison |
| `data/lifetime_knowledge/gold_corpus.json` | 5–10 highest-quality latest-era campaigns |
| `artifacts/campaign_era_partition.json` | Human-readable partition report |

## Design principle

> Trust 5 campaigns from the latest architecture more than 50 campaigns from obsolete ones.

Gold corpus drives operator priors. Archive campaigns remain for analysis but do not train the current system.
