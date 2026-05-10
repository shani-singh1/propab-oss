# Apply migrations/006_campaigns.sql to an existing Postgres volume.
# Docker init scripts only run on first DB creation; use this after upgrading.
param(
    [string]$ComposeDir = (Split-Path -Parent $PSScriptRoot)
)
Set-Location $ComposeDir
Get-Content "$ComposeDir\migrations\006_campaigns.sql" -Raw | docker compose exec -T postgres psql -U propab -d propab
Write-Host "Done. Verify: docker compose exec -T postgres psql -U propab -d propab -c '\dt research_campaigns'"
