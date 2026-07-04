# Agent 2 Engineering Log

## 2026-07-04 — Bootstrap (T1-003)

Started Agent 2 platform engineering loop. Delivered `scripts/engineering_status.py` for one-command engineering visibility.

## 2026-07-04 — T1-001 complete (Lifetime LWW → Postgres) — `945a901`

Alembic migrations for lifetime knowledge tables with upsert semantics. `propab/lifetime_postgres.py` routes `KnowledgeGraph`, `MetaScienceLedger`, and related stores through Postgres when `lifetime_store_backend=postgres`. Concurrent claim test passes (`tests/test_lifetime_postgres_concurrent.py`). JSON files remain read-only archives.

## 2026-07-04 — T1-002 complete (Genomics domain plugin) — `945a901`

Built `domain_modules/genomics/` end-to-end (adapter, verifier, preflight, 22-entry routing corpus). Preflight ~10s; dry-run passes. Zero core changes outside the plugin module.

## 2026-07-04 — Tier 2 platform batch — `945a901`

- T2-001: `docker-compose.prod.yml` with healthchecks, env-driven secrets, no source mounts
- T2-002: CI routing inspector + domain preflight gate in `.github/workflows/ci.yml`
- T2-003: `scripts/compare_campaigns.py`
- T2-004: `scripts/health_dashboard.py`
- T2-005: `scripts/setup_and_verify.sh` + README Getting Started

## 2026-07-04 — T3-001 complete (AP-free corpus + validation) — `945a901`, `1803b20`

20 AP-free hypotheses in `ap_free_corpus.json`; routing merged into inspector (341 total). AP-free single-point routing fix in `1803b20`. Preflight sweep to n=50,000 recorded in `artifacts/ap_free_preflight_n50000.json` (density 0.02048 at n=50k, 14.7s).

## 2026-07-05 — T3-002 through T3-005 — `614d258`

- **T3-002:** `domain_profiles/econometrics.py` with within-group R² panel FE gate. DiscoveryBench eval (`scripts/run_econometrics_discoverybench_eval.py`, 2026-07-05): 3 econometrics validation samples with `[domain_profile:econometrics]`, 20 min budget each → **mean HMS=0.0** (0 confirmed nodes; abstain policy). Gate correctness: `tests/test_econometrics_profile.py`. Full HMS lift requires campaigns that confirm panel-FE hypotheses within budget — tracked in `artifacts/econometrics_discoverybench_hms.json`.
- **T3-003:** Theme rules → domain plugins; Mandrake contrarian reset + `finding_audit` relocated.
- **T3-004:** Full enzyme kinetics plugin (BRENDA-style adapter, EC-class LOFO, 20-entry routing corpus).
- **T3-005:** Full graph invariants plugin (SNAP-style adapter, cross-family verifier, 20-entry routing corpus).
- CI updated to preflight all domains (no stub skip). Routing corpus **341/341** verified.

## 2026-07-04 — Tier 4 external readiness — `f4cfbff`, `10db19f`

- T4-001: OpenAPI route docs + `docs/api_reference.md`
- T4-002: `docs/adding_a_domain.md` (genomics worked example)
- T4-003: README refresh for external audience
- T4-004: `docs/operator_runbook.md`

## 2026-07-05 — Loop closure (artifact maintenance)

Updated `BACKLOG.md` commit hashes, `docs/component_map.md` (enzyme/graph active, T3-003 items no longer deferred), `engineering_status.py` (341 corpus from cached artifact, stub list removed), and exit-criteria artifacts for AP-free n=50k sweep and econometrics DiscoveryBench HMS.
