# Propab operator runbook

For production deployments using `docker-compose.prod.yml` or a custom stack.
Assumes Postgres, Redis, Qdrant, MinIO, API, orchestrator, and Celery worker are
all running.

---

## Service map

| Service | Port (default) | Role |
|---------|----------------|------|
| **api** | 8000 | HTTP + SSE; delegates campaigns to orchestrator |
| **orchestrator** | 8010 | Campaign loop, synthesis, beliefs, paper |
| **worker** | — | Celery: one hypothesis verification per task |
| **postgres** | 5432 | Campaign state, events, health metrics, lifetime knowledge |
| **redis** | 6379 | Celery broker + SSE pub/sub |
| **qdrant** | 6333 | Literature chunk embeddings |
| **minio** | 9000 | Object storage (PDFs, artifacts) |
| **migrate** | — | One-shot Alembic `upgrade head` on deploy |
| **frontend** | 3000 | Dashboard (optional for headless ops) |

Health checks:

```bash
curl -s http://localhost:8000/health | jq .
docker compose ps
docker compose -f docker-compose.prod.yml ps
```

---

## First-time / after deploy

```bash
cp .env.example .env          # fill API keys and secrets
docker compose up -d          # dev
# or
docker compose -f docker-compose.prod.yml up -d

docker compose run --rm migrate
python scripts/setup_and_verify.sh   # full gate (Linux/macOS)
```

Verify domain preflights:

```bash
python scripts/engineering_status.py
python scripts/start_v1_frontier_campaign.py --domain genomics --dry-run
```

---

## Reading the health dashboard

```bash
python scripts/health_dashboard.py --once
python scripts/health_dashboard.py --interval 30
```

The dashboard reads `artifacts/v1_frontier_campaign_latest.json` for the active
campaign id, then polls `GET /campaigns/{id}`.

| Signal | Healthy | Investigate when |
|--------|---------|------------------|
| `inconclusive_rate` | < 50% | Verifier routing wrong or hypotheses off-domain |
| `beliefs_active` | ≥ 1 after synthesis | Synthesis empty / evidence binding rejecting everything |
| `stop_reason` | non-null when completed | Null stop = abnormal termination |
| API unreachable | — | Container down, wrong port, or DB connection failure |

Postgres health metrics (literature citation rate, binding rejection rate, etc.)
are documented in [`propab_ownership_contracts.md`](../propab_ownership_contracts.md).

---

## Common failure modes

### Campaign stuck in `active` but no progress

**Symptoms:** `elapsed_sec` flat; event counts not increasing; worker queue idle.

**Checks:**

```bash
docker compose logs worker --tail 100
docker compose logs orchestrator --tail 100
docker compose exec redis redis-cli LLEN celery
```

**Fixes:**

1. Restart worker: `docker compose restart worker`
2. Restart orchestrator (campaign resumes from checkpoint if orchestrator was the blocker):
   `docker compose restart orchestrator`
3. Confirm `CELERY_BROKER_URL` and `REDIS_URL` match across worker and orchestrator
4. Inspect `GET /campaigns/{id}/resume-readiness` — fix belief backfill issues before resume

### Worker not receiving tasks

**Symptoms:** Celery queue empty but hypotheses pending; worker logs show no tasks.

**Checks:**

- Worker container healthy and same Redis URL as orchestrator
- `docker compose logs worker` for import errors at startup
- Domain plugin import failure in registry (one broken plugin is skipped, but worker code must load)

**Fix:** Rebuild worker image after code deploy; `docker compose build worker && docker compose up -d worker`

### Synthesis empty too early

**Symptoms:** `beliefs_active: 0`; high `falsifiability_rejected_count` in synthesis metrics.

**Checks:**

- `GET /campaigns/{id}` → `belief_state.active_beliefs`
- Event log for `synthesis.*` events with rejection reasons

**Fixes:**

- Domain preflight may have passed but hypotheses are off-topic — tighten question / use `[domain_profile:...]`
- Evidence binding rejecting citations — check citation integrity metric
- Contrarian resume: `POST .../resume` with `belief_reset: "contrarian"`

### Orchestrator 502 from API

**Symptoms:** `POST /campaigns` returns 502.

**Fix:** Ensure `ORCHESTRATOR_URL` points to reachable orchestrator and
`ORCHESTRATOR_INTERNAL_TOKEN` matches on both sides.

### Domain preflight failed at launch

**Symptoms:** Campaign never starts; dry-run shows `preflight_passed: false`.

**Fix:** Run plugin preflight directly:

```python
from propab.domain_modules.registry import get_domain_plugin
print(get_domain_plugin("genomics").preflight())
```

Fix data paths under `PROPAB_DATA_DIR`, install missing deps, or reduce dataset size.

### Routing corpus regression after deploy

```bash
python scripts/inspect_hypothesis_routing.py --corpus --require-zero-mismatches
```

Must exit 0 before promoting a release (also enforced in CI).

---

## Stop a campaign cleanly

There is no public `POST /campaigns/{id}/stop` endpoint. Preferred approaches:

1. **Let budget exhaust** — set a low `compute_budget_hours` on resume.
2. **Orchestrator restart** — campaign checkpoints to Postgres; status may remain
   `active` until the loop observes shutdown. Check `stop_reason` after restart.
3. **Manual DB update** (last resort):

```sql
UPDATE research_campaigns
SET status = 'budget_exhausted', stop_reason = 'operator_halt'
WHERE id = '<campaign_id>';
```

Then verify workers are not still processing queued hypothesis tasks for that campaign.

---

## Resume after service restart

Campaign state lives in Postgres (`research_campaigns`, `events`). After API or
orchestrator restart:

```bash
curl -s http://localhost:8000/campaigns/<id>/resume-readiness | jq .
curl -X POST http://localhost:8000/campaigns/<id>/resume \
  -H 'Content-Type: application/json' \
  -d '{"compute_budget_hours": 4.0}'
```

Lifetime knowledge (cross-campaign compounding) uses `LIFETIME_STORE_BACKEND`:

- `json` — files under `PROPAB_DATA_DIR/lifetime_knowledge/` (tests, single-node)
- `postgres` — upsert tables from migration `20260704130000` (production default in compose)

---

## Database maintenance

### What grows

| Table / area | Growth driver |
|--------------|---------------|
| `events` | Every LLM call, tool step, verdict |
| `llm_calls` | Prompt/response audit trail |
| `research_campaigns` | One row per campaign + large JSON tree |
| `lifetime_knowledge_*` | Cross-campaign claims, seeds, policies |
| `campaign_synthesis_events` | Health metrics per synthesis round |
| Qdrant / MinIO | Literature embeddings and PDFs |

### Pruning (dev / stale campaigns)

Keep production pruning conservative — export forensic JSON first.

```sql
-- Example: delete events for a abandoned dev campaign (replace id)
DELETE FROM events WHERE session_id = '<uuid>';
DELETE FROM research_campaigns WHERE id = '<uuid>';
DELETE FROM research_sessions WHERE id = '<uuid>';
```

Alembic is the only schema migration path:

```bash
docker compose run --rm migrate
# or locally:
python -m alembic upgrade head
```

---

## Logs

```bash
docker compose logs -f api orchestrator worker
docker compose logs worker --since 30m | rg -i "error|traceback|failed"
```

| Log pattern | Meaning |
|-------------|---------|
| `Orchestrator unreachable` | API cannot reach orchestrator URL |
| `DOMAIN_PREFLIGHT_FAILED` | Campaign refused at launch |
| `evidence_binding` / `falsifiability` | Belief rejected at synthesis write time |
| Celery `Task ... raised` | Worker verification exception — check hypothesis + domain verifier |

---

## Production checklist

- [ ] Secrets only via env (never in images)
- [ ] `docker compose -f docker-compose.prod.yml` healthchecks green
- [ ] Migrations applied
- [ ] `LIFETIME_STORE_BACKEND=postgres` on api/orchestrator/worker
- [ ] Routing corpus 0 mismatches
- [ ] `ORCHESTRATOR_INTERNAL_TOKEN` set
- [ ] Backups for Postgres volume and MinIO bucket

---

## Related docs

- [API reference](./api_reference.md)
- [Adding a domain](./adding_a_domain.md)
- [ARCHITECTURE.md](../ARCHITECTURE.md)
