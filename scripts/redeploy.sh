#!/usr/bin/env bash
# redeploy.sh — reliable, LOUD redeploy of the Propab stack.
#
# Purpose: stop the deployed images from drifting behind `main`. Rebuilds the
# app images, brings the stack up (which runs the one-shot `migrate` service),
# and then runs a POST-DEPLOY HEALTH CHECK. If anything is wrong — a service is
# not Up/healthy, the API does not return 200 on /campaigns, or migrations are
# not at head — it prints a clear message and exits NON-ZERO.
#
# Usage:
#   scripts/redeploy.sh                 # rebuild + up -d + health check
#   scripts/redeploy.sh --no-build      # up -d + health check (skip rebuild)
#   scripts/redeploy.sh --check         # health check ONLY (read-only, no build/up)
#   scripts/redeploy.sh --dry-run       # print what it WOULD do, run read-only checks only
#   scripts/redeploy.sh -f docker-compose.yml -f docker-compose.prod.yml ...
#
# Env:
#   API_BASE_URL   Base URL for the API health probe (default http://localhost:8000)
#   HEALTH_TIMEOUT Seconds to wait for services to become healthy (default 180)
#
# Notes:
#   --check / --dry-run never rebuild or restart anything, so they are safe to
#   run against a live stack.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"

DO_BUILD=1
DO_UP=1
DRY_RUN=0
COMPOSE_ARGS=()   # extra args forwarded to `docker compose` (e.g. -f FILE)

# Long-running services that must be Up after a deploy.
LONG_RUNNING=(api orchestrator worker literature postgres redis qdrant minio)
# Subset that declare a healthcheck and must report "healthy".
HEALTHCHECKED=(api orchestrator postgres redis qdrant minio)

# ---- pretty logging -------------------------------------------------------
c_red()  { printf '\033[31m%s\033[0m\n' "$*"; }
c_grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
c_ylw()  { printf '\033[33m%s\033[0m\n' "$*"; }
step()   { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()     { c_grn "    OK  $*"; }
fail()   { c_red "    FAIL $*"; }

die() { c_red ""; c_red "REDEPLOY FAILED: $*"; c_red ""; exit 1; }

# ---- arg parsing ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) DO_BUILD=0; shift ;;
    --check)    DO_BUILD=0; DO_UP=0; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -f|--file)  COMPOSE_ARGS+=(-f "$2"); shift 2 ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

dc() { docker compose "${COMPOSE_ARGS[@]}" "$@"; }

run() {
  # Run a command, or just echo it under --dry-run.
  if [[ "$DRY_RUN" -eq 1 ]]; then
    c_ylw "    [dry-run] $*"
  else
    "$@"
  fi
}

# ---- preflight ------------------------------------------------------------
step "Preflight"
command -v docker >/dev/null 2>&1 || die "docker not found on PATH"
dc config -q || die "docker compose config is INVALID — fix docker-compose.yml first"
ok "docker compose config valid"

# ---- build ----------------------------------------------------------------
if [[ "$DO_BUILD" -eq 1 ]]; then
  step "Rebuilding app images (api orchestrator worker literature migrate)"
  run dc build api orchestrator worker literature migrate
  ok "images built"
else
  step "Skipping rebuild (--no-build/--check)"
fi

# ---- up (runs migrate) ----------------------------------------------------
if [[ "$DO_UP" -eq 1 ]]; then
  step "Bringing stack up (docker compose up -d — runs migrate)"
  run dc up -d --remove-orphans
  ok "compose up issued"
else
  step "Skipping 'up' (--check) — health-checking the live stack only"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  step "Dry-run: running read-only health checks against the current stack"
fi

# ===========================================================================
# POST-DEPLOY HEALTH CHECK
# ===========================================================================
FAILED=0

# 1) every long-running service is Up; every healthchecked one is healthy.
step "Health check 1/3: service states"

svc_cid() { dc ps -q "$1" 2>/dev/null | head -n1; }

svc_state() {
  # echoes "<status> <health>" for a service ("" health if none)
  local cid; cid="$(svc_cid "$1")"
  [[ -z "$cid" ]] && { echo "missing none"; return; }
  docker inspect -f '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid" 2>/dev/null || echo "missing none"
}

is_healthchecked() {
  local s; for s in "${HEALTHCHECKED[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1
}

# Wait loop: give healthchecked services up to HEALTH_TIMEOUT to converge.
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
while true; do
  pending=0
  for svc in "${LONG_RUNNING[@]}"; do
    read -r status health <<<"$(svc_state "$svc")"
    if [[ "$status" != "running" ]]; then pending=1; fi
    if is_healthchecked "$svc" && [[ "$health" != "healthy" && "$health" != "none" ]]; then pending=1; fi
  done
  if [[ "$pending" -eq 0 ]]; then break; fi
  if [[ "$(date +%s)" -ge "$deadline" ]]; then break; fi
  sleep 3
done

for svc in "${LONG_RUNNING[@]}"; do
  read -r status health <<<"$(svc_state "$svc")"
  if [[ "$status" != "running" ]]; then
    fail "$svc is '$status' (expected running)"; FAILED=1
  elif is_healthchecked "$svc" && [[ "$health" != "healthy" && "$health" != "none" ]]; then
    fail "$svc is running but health='$health' (expected healthy)"; FAILED=1
  else
    ok "$svc: $status${health:+ ($health)}"
  fi
done

# 2) API returns HTTP 200 on /campaigns.
step "Health check 2/3: API GET ${API_BASE_URL}/campaigns"
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "${API_BASE_URL}/campaigns" || echo 000)"
if [[ "$code" == "200" ]]; then
  ok "/campaigns -> 200"
else
  fail "/campaigns -> $code (expected 200)"; FAILED=1
fi

# 3) migrations are at head (no drift).
step "Health check 3/3: alembic migrations at head"
# `alembic current` (applied) must include the `alembic heads` revision.
if [[ "$DRY_RUN" -eq 1 ]]; then
  c_ylw "    [dry-run] would compare 'alembic current' vs 'alembic heads' in the migrate image"
else
  cur="$(dc run --rm --no-deps --entrypoint alembic migrate current 2>/dev/null | grep -oE '^[0-9a-f]{6,}' | head -n1 || true)"
  head="$(dc run --rm --no-deps --entrypoint alembic migrate heads 2>/dev/null | grep -oE '^[0-9a-f]{6,}' | head -n1 || true)"
  if [[ -z "$cur" || -z "$head" ]]; then
    fail "could not determine alembic current/head (cur='$cur' head='$head')"; FAILED=1
  elif [[ "$cur" == "$head" ]]; then
    ok "migrations at head ($head)"
  else
    fail "migration DRIFT: applied=$cur head=$head — run migrate"; FAILED=1
  fi
fi

# ---- verdict --------------------------------------------------------------
if [[ "$FAILED" -ne 0 ]]; then
  die "one or more post-deploy health checks failed (see FAIL lines above)"
fi

step "All post-deploy health checks passed"
c_grn "Propab stack is Up, healthy, serving /campaigns, and at migration head."
