# GeoAI PostGIS Data Image

This image packages the thesis PostGIS dump into a runnable database container. On first startup, the official PostgreSQL entrypoint restores `geoai.dump` into the configured database.

The dump itself is not tracked by git. Before building, copy the verified local dump into this directory:

```powershell
Copy-Item .docker_db_backups\geoai-20260625_114034.dump docker\postgis-data\geoai.dump
```

Build and tag for GitHub Container Registry:

```powershell
docker build -t ghcr.io/patrickfranzelin/master-thesis-geoai/postgis-data:latest docker/postgis-data
docker push ghcr.io/patrickfranzelin/master-thesis-geoai/postgis-data:latest
```

After pushing, make the package public in GitHub Packages if external users should be able to pull it without authentication.

Run directly:

```powershell
docker run --name geoai-postgis-data -p 5434:5432 `
  -e POSTGRES_DB=geoai `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=geoai `
  ghcr.io/patrickfranzelin/master-thesis-geoai/postgis-data:latest
```

Connection string:

```text
postgresql://postgres:geoai@localhost:5434/geoai
```
