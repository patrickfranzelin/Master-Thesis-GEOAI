# Docker PostGIS Workflow

This repository supports two Docker database workflows:

- `data-db`: run the public GHCR image that already contains the thesis database dump.
- `db`: start an empty PostGIS database and copy from a local PostgreSQL source.

## Run Public Data Image

```powershell
docker compose up -d data-db
$env:PG_CONN="postgresql://postgres:geoai@localhost:5434/geoai"
```

This uses:

```text
ghcr.io/patrickfranzelin/master-thesis-geoai-data:latest
```

The image restores the embedded PostGIS dump on first startup.

## Start Docker PostGIS

This starts an empty PostGIS database for local copy/import workflows.

```powershell
docker compose up -d db
```

The Docker database listens on local port `5433` so it does not collide with a local PostgreSQL server on `5432`.

If `5433` is already used, choose another host port:

```powershell
$env:GEOAI_DB_PORT="5434"
docker compose up -d db
```

Use this connection string for the pipeline:

```powershell
$env:PG_CONN="postgresql://postgres:geoai@localhost:5433/geoai"
```

For a real password, copy `.env.docker.example` to `.env.docker.local`, keep that local file out of git, and start Compose with:

```powershell
docker compose --env-file .env.docker.local up -d db
```

## Copy Local DB Into Docker

This copies from the host PostgreSQL database into Docker. It does not delete or modify the source database.

```powershell
.\scripts\db_copy_local_to_docker.ps1 -SourcePassword "<local-db-password>"
```

If the Docker database already has old `src` schema data and you want to replace only the Docker copy:

```powershell
.\scripts\db_copy_local_to_docker.ps1 -SourcePassword "<local-db-password>" -CleanTarget
```

If local port `5433` is already allocated, pass the same alternate port to the copy script:

```powershell
.\scripts\db_copy_local_to_docker.ps1 -SourcePassword "<local-db-password>" -TargetPort 5434
$env:PG_CONN="postgresql://postgres:geoai@localhost:5434/geoai"
```

The script also keeps a local dump under `.docker_db_backups/`.

## GitHub

Commit the Docker/config/scripts to GitHub, not the database volume or raw data dumps. Database content should stay in:

- the Docker named volume `geoai_postgis_data`
- local dump files under `.docker_db_backups/`
- an external artifact store if you need to share large data

GitHub is appropriate for code and reproducible setup, not for large PostGIS data or local credentials.
