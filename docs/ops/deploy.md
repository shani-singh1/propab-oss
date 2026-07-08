# Deploy & Ops Runbook

How the local Propab stack is deployed, kept in sync with `main`, and monitored.
This exists because the deployed images used to drift behind `main` on every
merge (a manual rebuild that was sometimes skipped — which once silently broke
telemetry), and because the infra services crashed and stayed down ~2h
**unnoticed**. The tooling below makes redeploys reliable/loud and makes a
crashed service self-heal and get flagged.

- **Stack**: `docker-compose.yml` — `api:8000`, `orchestrator:8010`, `worker`
  (celery), `literature:8020`, `postgres:15`, `redis:7`, `qdrant`, `minio`, plus
  the one-shot `migrate` (`alembic upgrade head`) and `frontend:3000`.
- **DB creds**: `propab:propab` on `postgres:5432` (db `propab`).
- **Target**: local dev deployment (no cloud). Commands run from the repo root.

---

## THE RULE: rebuild after every backend-changing merge

Until real CD exists, **the deployed containers do not update themselves.** A
merge to `main` that changes any backend code, migration, Docker/compose file,
or dependency is **not live** until the images are rebuilt and the stack is
brought back up.

> After merging backend changes to `main`: `git pull` then run
> `scripts/redeploy.sh`. That is the whole rule. Skipping it is how the running
> orchestrator ended up without the new telemetry hook/migration.

Frontend-only or docs-only merges do not require a rebuild of the backend
images (the `frontend` image still needs a rebuild for frontend changes).

---

## Redeploy: `scripts/redeploy.sh`

The one command that stops the drift. It rebuilds the app images, brings the
stack up (which runs the `migrate` service), then runs a **post-deploy health
check** and exits **non-zero with a clear message** if anything is wrong.

```bash
git checkout main && git pull
scripts/redeploy.sh
```

What it does, in order:

1. **Preflight** — `docker compose config -q` must pass.
2. **Rebuild** app images (`api orchestrator worker literature migrate`).
3. **`docker compose up -d`** — starts/updates every service and runs the
   one-shot `migrate` (`alembic upgrade head`).
4. **Post-deploy health check** (all three must pass):
   - every long-running service is `running`, and every service that declares a
     healthcheck reports `healthy` (waits up to `HEALTH_TIMEOUT`, default 180s);
   - the API returns **HTTP 200** on `GET /campaigns`;
   - Alembic migrations are **at head** (applied revision == `alembic heads`) —
     this is the direct check for "images lagging `main`".

If any check fails it prints the failing lines and exits non-zero.

### Flags

| Flag | Effect |
|------|--------|
| _(none)_ | rebuild + `up -d` + health check (the normal redeploy) |
| `--no-build` | skip the rebuild, still `up -d` + health check |
| `--check` | **health check only** — no build, no restart (safe on a live stack) |
| `--dry-run` | print what it *would* do and run only the read-only checks |
| `-f FILE` | forward a `-f` compose file to `docker compose` (repeatable) |

### Env

| Var | Default | Meaning |
|-----|---------|---------|
| `API_BASE_URL` | `http://localhost:8000` | base URL for the `/campaigns` probe |
| `HEALTH_TIMEOUT` | `180` | seconds to wait for services to become healthy |

Verify a running stack without touching it:

```bash
scripts/redeploy.sh --check
```

---

## Continuous monitoring: `scripts/health_monitor.py`

Standalone, **read-only** status/alert tool. Use it to catch the "crashed and
stayed down unnoticed" failure mode. It exits non-zero when unhealthy, so it
drops straight into `watch`/cron.

```bash
python scripts/health_monitor.py                 # compact dashboard
python scripts/health_monitor.py --json          # machine-readable
watch -n 60 'python scripts/health_monitor.py || echo "PROPAB ALERT"'
```

It checks:

1. **Docker services** — each long-running service is Up; healthchecked ones
   (`api`, `orchestrator`, `postgres`, `redis`, `qdrant`, `minio`) are `healthy`.
2. **API** — `GET /health` and `GET /campaigns` both return 200.
3. **Stuck campaigns** — a campaign that is `active` in `research_campaigns` but
   has emitted **no new event** for the last N minutes (`--stuck-minutes`,
   default 30). This flags a run that silently died even while the containers
   look up.

Useful flags: `--stuck-minutes N`, `--api-base URL`, `--database-url URL`,
`--no-docker` / `--no-api` / `--no-db` (run a subset), `--json`.
The DB URL defaults to `DATABASE_URL_SYNC` / `DATABASE_URL`, falling back to
`postgresql://propab:propab@localhost:5432/propab`. It accepts SQLAlchemy-style
URLs (`+asyncpg` / `+psycopg` are stripped automatically). Requires
`psycopg` for the campaign check; without it the docker/API checks still run.

> Run `health_monitor.py` from the deploy directory (the repo root that owns the
> running compose project) so `docker compose ps` resolves to the right project.

---

## Self-healing resilience (in `docker-compose.yml`)

The compose file is configured so a crashed infra service **auto-recovers** —
the ~2h silent crash would now self-heal:

- **`restart: unless-stopped`** on every long-running service (`api`,
  `orchestrator`, `worker`, `literature`, `postgres`, `redis`, `qdrant`,
  `minio`) — Docker restarts them on crash and on host/daemon reboot.
- **`healthcheck:`** blocks on `api`/`orchestrator` (HTTP `/health`),
  `postgres` (`pg_isready`), `redis` (`redis-cli ping`), `qdrant` (`/healthz`),
  and `minio` (`/minio/health/live`).
- **`depends_on … condition: service_healthy`** so `api`/`orchestrator`/`worker`
  wait for `postgres` and `redis` to be *healthy* before starting — this is why
  the api used to crash-loop on `redis:6379` at boot; it now waits.

The `migrate` service is intentionally `restart: "no"` (one-shot; it runs
`alembic upgrade head` and exits). `docker-compose.prod.yml` carries the same
resilience patterns.

Validate compose changes without starting anything:

```bash
docker compose config -q
```

---

## Typical flows

**Deploy latest `main`:**
```bash
git checkout main && git pull
scripts/redeploy.sh
```

**Is the running stack healthy right now?**
```bash
scripts/redeploy.sh --check        # one-shot pass/fail
python scripts/health_monitor.py   # dashboard incl. stuck campaigns
```

**A service crashed** — Docker restarts it automatically
(`restart: unless-stopped`). Confirm recovery with `health_monitor.py`; if it
does not come back, inspect logs: `docker compose logs -f <service>`.
