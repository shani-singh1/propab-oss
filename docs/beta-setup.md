# Propab beta setup & troubleshooting

A "clone → run → launch your first campaign" guide for beta users. For
production operations (scaling, DB maintenance, resuming, pruning) see the
[operator runbook](./operator_runbook.md).

---

## 1. Prerequisites

- **Docker + Docker Compose** (Docker Desktop on macOS/Windows, or Docker
  Engine + compose plugin on Linux). No GPU required.
- **One LLM API key** — OpenAI **or** Google Gemini. That's the only secret you
  must provide.
- Free local ports: `3000` (frontend), `8000` (api), `8010` (orchestrator),
  `8020` (literature), `5432` (postgres), `6379` (redis), `6333` (qdrant),
  `9000`/`9001` (minio). All are published to `localhost` by `docker-compose.yml`.

---

## 2. First run

```bash
git clone https://github.com/shani-singh1/propab-oss.git
cd propab-oss                     # keep this directory name (see gotcha #5 below)
cp .env.example .env
```

Edit `.env` and set your provider + key. The **only** required choice:

**Option A — OpenAI (matches the defaults, simplest):**
```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
EMBED_PROVIDER=openai
```

**Option B — Google Gemini:**
```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=AIza...
EMBED_PROVIDER=google          # important: reuse the same key for embeddings
```

Then bring the stack up:

```bash
docker compose up          # add -d to run detached
```

This builds and starts `api`, `orchestrator`, `worker`, `literature`,
`frontend`, plus `postgres`, `redis`, `qdrant`, `minio`. A one-shot `migrate`
service runs `alembic upgrade head` first; every app service waits for it to
finish. First build pulls base images and compiles the scientific Python stack,
so it can take several minutes.

Once up:

- Dashboard → <http://localhost:3000>
- API health → <http://localhost:8000/health>
- Interactive API docs → <http://localhost:8000/docs>
- Literature service health → <http://localhost:8020/health>

---

## 3. Launch your first campaign

Use the dashboard, or `POST /campaigns` directly. A pure-computation domain like
`math_combinatorics` is the safest first run — it needs no mounted datasets:

```bash
curl -X POST http://localhost:8000/campaigns \
  -H 'Content-Type: application/json' \
  -d '{
        "question": "Which additive combinatorics constructions maximize Sidon density under greedy search? [domain_profile:math_combinatorics]",
        "compute_budget_hours": 1.0,
        "policy_mode": "accepted",
        "max_hypotheses": 40
      }'
```

Response:

```json
{ "campaign_id": "…", "stream_url": "/stream/…", "status": "started" }
```

Request-body fields (validated by the API):

| Field | Required | Default | Notes |
|-------|----------|---------|-------|
| `question` | yes | — | min length 8. Add `[domain_profile:<id>]` to pin the domain. |
| `compute_budget_hours` | no | `4.0` | range `0.1`–`168.0` |
| `policy_mode` | no | `accepted` | `accepted` \| `candidate` |
| `max_hypotheses` | no | none | range `1`–`500` |

Watch it:

```bash
# live event stream (SSE)
curl -N http://localhost:8000/stream/<campaign_id>

# snapshot: tested/confirmed/refuted counts + event_counts_by_type
curl -s http://localhost:8000/campaigns/<campaign_id> | jq .summary
```

Other domains: `genomics` (synthetic fallback when no GTEx data is mounted),
`materials`, `network_diffusion`. `enzyme_kinetics` and `graph_invariants` are
stubs. See the [API reference](./api_reference.md) for monitoring and resuming.

---

## 4. Health checks

```bash
docker compose ps                       # all services Up / healthy?
curl -s http://localhost:8000/health    # api liveness + build metadata
docker compose logs -f api orchestrator worker
```

Expected: `postgres`, `redis`, `minio`, `api`, `orchestrator` show `healthy`;
`migrate` shows `Exited (0)`; `qdrant`, `literature`, `frontend`, `worker` show
`Up` (they have no in-container healthcheck — see gotcha #4).

---

## 5. Troubleshooting (common first-run failures)

### 1. Missing / invalid LLM API key
**Symptom:** campaign starts but stalls with 0 hypotheses; `orchestrator`/`worker`
logs show `401`/`invalid api key`/auth errors from OpenAI or Google.
**Fix:** confirm the key in `.env` matches `LLM_PROVIDER`, then recreate the
containers so the new value is picked up:
```bash
docker compose up -d --force-recreate orchestrator worker api
```
`.env` changes are read by compose at container-create time, not on the fly.

### 2. Provider / key mismatch (silent quality drop)
**Symptom:** stack runs, but literature retrieval and dedup seem weak, or logs
mention embedding fallback.
**Cause:** `EMBED_PROVIDER` points at a provider whose key you didn't set, so
embeddings silently fall back to non-embedding ranking.
**Fix:** set `EMBED_PROVIDER` to match the key you provided (`openai` with
`OPENAI_API_KEY`, `google` with `GOOGLE_API_KEY`).

### 3. Port already in use
**Symptom:** `docker compose up` fails with `bind: address already in use` /
`ports are not available` on 3000, 8000, 5432, etc.
**Fix:** stop the conflicting process, or remap the host side of the port in a
local override. Find the offender:
```bash
# Linux/macOS
lsof -i :8000
# Windows (PowerShell)
Get-NetTCPConnection -LocalPort 8000
```

### 4. A service shows "unhealthy" or won't come up
**Checks:**
```bash
docker compose ps
docker compose logs <service> --tail 100
```
- `qdrant` has **no** healthcheck by design (its image ships no curl/wget), so it
  shows `Up`, not `healthy` — dependents wait on `service_started`. This is
  expected, not a failure.
- If `api`/`orchestrator` never reach `healthy`, they usually can't reach
  Postgres/Redis — check that `migrate` exited 0 and `postgres` is `healthy`.

### 5. Sandbox image not found (code steps fail)
**Symptom:** worker logs show it cannot find image `propab-oss-worker:latest`
when running a code step; hypotheses that need code execution fail.
**Cause:** the sandbox image name is derived from the Compose **project name**,
which defaults to the repo directory name. `SANDBOX_IMAGE` defaults to
`propab-oss-worker:latest`, which only matches if the directory is `propab-oss`.
**Fix:** keep the clone directory named `propab-oss`, **or** set
`COMPOSE_PROJECT_NAME=propab-oss` in `.env`, **or** set `SANDBOX_IMAGE` to the
image Compose actually built (`docker images | grep worker`).

### 6. Worker can't spawn sandboxes (Docker socket)
**Symptom:** worker logs show Docker connection/permission errors when starting a
sandbox container.
**Cause:** the worker mounts `/var/run/docker.sock` to run sandbox containers on
the host daemon (Docker-out-of-Docker). This works on Docker Desktop and standard
Linux Docker; it will not work on rootless/remote daemons without extra setup.
**Fix:** run on a daemon that exposes `/var/run/docker.sock`, or ensure your user
can access it.

### 7. Empty campaign / generation stalled
**Symptom:** campaign is `active` but `event_counts_by_type` isn't growing and
`total_tested` stays at 0.
**Checks:**
```bash
docker compose logs orchestrator --tail 100     # prior build / expansion
docker compose logs worker --tail 100           # task pickup + code steps
docker compose exec redis redis-cli LLEN celery # queued but unprocessed tasks?
```
**Likely causes:** LLM key/quota (see #1), `DOMAIN_PREFLIGHT_FAILED` (the domain
is underpowered — use `math_combinatorics` for a first run), or a stuck worker
(`docker compose restart worker`). The campaign checkpoints to Postgres, so
restarting `orchestrator`/`worker` resumes from the last checkpoint.

### 8. Frontend loads but shows no data / API errors in the browser
**Cause:** the frontend is a static build with the API URL baked in at build time
(`VITE_API_BASE=http://localhost:8000`). It expects the API reachable at
`localhost:8000` **from your browser**.
**Fix:** open the dashboard on the same machine running Docker. To serve it to
another host, rebuild the frontend with a different `VITE_API_BASE` build arg.

---

## 6. Stopping & cleaning up

```bash
docker compose down             # stop containers, keep data volumes
docker compose down -v          # also delete postgres/qdrant/minio/propab data
```

---

## 7. Security note for beta users

- The real `.env` is git-ignored and is **not** copied into the built images
  (the Dockerfiles copy only source dirs), so your keys are not baked into
  images. Still, treat `.env` as a secret and never commit it.
- Default credentials in compose (`postgres/propab`, `minio propab/propab_secret`)
  are for local use only. Change them (via `docker-compose.prod.yml` +
  `.env`) before exposing any port beyond localhost.
- Local dev has **no API auth**. Do not expose port 8000 publicly without a
  reverse proxy; set `ORCHESTRATOR_INTERNAL_TOKEN` for any shared deployment.

---

## Related docs

- [README quickstart](../README.md#quickstart)
- [API reference](./api_reference.md) — monitor, resume, retrieve findings
- [Operator runbook](./operator_runbook.md) — production ops & failure modes
- [ARCHITECTURE.md](../ARCHITECTURE.md)
