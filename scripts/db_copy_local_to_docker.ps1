param(
    [string]$SourceHost = "host.docker.internal",
    [int]$SourcePort = 5432,
    [string]$SourceDb = "geoai",
    [string]$SourceUser = "postgres",
    [Parameter(Mandatory = $true)]
    [string]$SourcePassword,
    [string]$TargetDb = "geoai",
    [string]$TargetUser = "postgres",
    [int]$TargetPort = 5433,
    [string]$BackupDir = ".docker_db_backups",
    [switch]$CleanTarget
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI is not available on PATH."
}

New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$dumpPath = Join-Path $BackupDir "$SourceDb-$timestamp.dump"

$env:GEOAI_DB_PORT = "$TargetPort"

Write-Host "Starting Docker PostGIS..."
docker compose up -d db
if ($LASTEXITCODE -ne 0) {
    throw "Docker PostGIS did not start. Port $TargetPort may already be allocated. Try: .\scripts\db_copy_local_to_docker.ps1 -SourcePassword '<password>' -TargetPort 5434"
}

Write-Host "Waiting for Docker PostGIS health check..."
$health = ""
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    $health = docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}" geoai-postgis 2>$null
    if ($health -eq "healthy") {
        break
    }
}

if ($health -ne "healthy") {
    throw "Docker PostGIS did not become healthy. Last container status: $health"
}

Write-Host "Creating dump from local PostgreSQL source: ${SourceHost}:$SourcePort/$SourceDb"
docker compose exec -T `
    -e PGPASSWORD="$SourcePassword" `
    db pg_dump `
    -h "$SourceHost" `
    -p "$SourcePort" `
    -U "$SourceUser" `
    -d "$SourceDb" `
    --format=custom `
    --no-owner `
    --no-acl `
    --file="/tmp/$SourceDb.dump"
if ($LASTEXITCODE -ne 0) {
    throw "pg_dump failed. Check that the source database is reachable from Docker at ${SourceHost}:$SourcePort and that the PostgreSQL client version matches the source server."
}

Write-Host "Copying dump to $dumpPath"
docker cp "geoai-postgis:/tmp/$SourceDb.dump" "$dumpPath"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to copy dump from the Docker container."
}

Write-Host "Restoring dump into Docker PostgreSQL target: $TargetDb"
$restoreArgs = @(
    "-U", $TargetUser,
    "-d", $TargetDb,
    "--no-owner",
    "--no-acl"
)

if ($CleanTarget) {
    $restoreArgs += @("--clean", "--if-exists")
}

$restoreArgs += "/tmp/$SourceDb.dump"
docker compose exec -T db pg_restore @restoreArgs
if ($LASTEXITCODE -ne 0) {
    throw "pg_restore failed. The dump was kept at: $dumpPath"
}

Write-Host "Done. Docker database connection:"
Write-Host "PG_CONN=postgresql://${TargetUser}:<password>@localhost:$TargetPort/$TargetDb"
Write-Host "Local backup dump kept at: $dumpPath"
