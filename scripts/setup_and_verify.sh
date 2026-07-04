#!/usr/bin/env bash
# Complete setup verification for a new environment (Agent 2 T2-005)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "1. Checking environment variables..."
required=(DATABASE_URL REDIS_URL PROPAB_DATA_DIR)
for v in "${required[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    echo "Missing required env: $v (see .env.example)" >&2
    exit 1
  fi
done

echo "2. Starting services..."
docker compose up -d

echo "3. Running migrations..."
docker compose run --rm migrate

echo "4. Running tests..."
python -m pytest tests/ -q

echo "5. Checking routing corpus..."
python scripts/inspect_hypothesis_routing.py --corpus --require-zero-mismatches

echo "6. Engineering status..."
python scripts/engineering_status.py --quick

echo "7. Domain dry-run (genomics)..."
python scripts/start_v1_frontier_campaign.py --domain genomics --dry-run

echo "Setup verified. Propab is ready."
