# Demo Execution Plan

**Objective:** Prove capability, not improve architecture.

## P0 — Architecture Freeze

**Allowed:** bug fixes, instrumentation, benchmark adapters, visualization

**Forbidden:** new layers, policy evolution, bandits, simulator work, search changes

See also: [campaign era partitioning](./campaign_era_partitioning.md) — gold corpus only for priors.

## P1 — Gold Corpus Only

| Set | Count | Use |
|-----|-------|-----|
| **Gold corpus** | 7 Era-4 campaigns | Training, priors, demo benchmark |
| **Archive** | 48 campaigns | Historical analysis only |

Old campaigns do **not** affect priors (`use_gold_priors=True` in credit cycle).

## P2 — Benchmark Domain (one only)

**Domain:** `graphs_sis_contagion`

**Question:** Which structural properties of complex networks most strongly determine contagion spreading speed under competing diffusion models?

**Why this domain:**
- Objective metric (`final_outbreak_fraction`, `improvement_pct`)
- Cheap verification (`numeric_summary`, `literature_baseline_compare`)
- Reproducible (fixed question, shared baseline `3351d2ab`)
- Understandable to outsiders

## P3 — Benchmark Harness

```
demo/benchmark/
  domain.py    — single domain config
  baseline.py  — (via gold.py) baseline campaign metrics
  metric.py    — objective metrics
  verifier.py  — cheap verification checks
  report.py    — demo asset reports
  gold.py      — gold corpus enforcement
  load.py      — Postgres loaders
```

## P4 — Pilot Runs (10–20 min)

Find bugs, not discoveries. No code changes once pilots stabilize.

```bash
docker compose up -d
python scripts/run_demo_pilot.py --minutes 15
```

## P5 — Main Runs (2–4 h)

Parallel, no architecture modifications during runs.

```bash
python scripts/run_demo_main.py --count 2 --hours 3
python scripts/monitor_campaign.py --state-file artifacts/demo/main_latest.json
python scripts/build_demo_assets.py
```

## P6 — Demo Assets

```bash
python scripts/build_demo_assets.py
```

Outputs: `artifacts/demo/demo_report.md` + `demo_report.json`

Shows: **Question → Hypothesis tree → Verification → Finding → Improvement over baseline**

## Success Criterion

People remember **"What did Propab discover?"** — not **"How is Propab implemented?"**
